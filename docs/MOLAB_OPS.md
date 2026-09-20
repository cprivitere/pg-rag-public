# molab platform ops — how the hosted notebook actually works

Operational knowledge for `notebooks/molab-mirror/notebook.py` (the
PG-RAG chat on molab) that doesn't fit the pairing protocol in the
`molab-notebook` skill. Read this before touching the notebook's
platform-coupled parts: `setup`, `rag_index` cache paths, model identity.

## What molab is

[molab](https://molab.marimo.io) is marimo's hosted notebook service
(by marimo HQ, the same team as marimo OSS). A "sandbox" is one notebook
session: a container with a marimo server at
`https://sb-<id>.sb.molab.run/`, guarded by a 64-hex bearer token.
Identifying traits:

- URL pattern `sb-*.sb.molab.run` (the `sb-` prefix is stable; the hex id
  changes whenever the sandbox is recreated).
- Auth is per-sandbox, rotated on every recreate. No stable project-level
  credentials.
- Sandboxes are **ephemeral by default**: idle timeouts, and every
  lifecycle event that restarts the container is a recreate.

## Sandbox lifecycle (what causes recreation)

| Event | Result |
|---|---|
| GPU attach / detach | Recreate at a NEW url + new token (old URL 410s) |
| Sandbox shutdown/restart in UI | Recreate at a NEW url + new token |
| Session reconnect (browser refresh) | Same sandbox, marimo renames the session id (internal only) |
| Idle | Frozen, then closed after timeout — notebook state lost unless committed to the HF bucket or repo |

Corollaries:

- Never treat a sandbox as durable storage. Durable artifacts live in
  `hf://buckets/Nubula/paddock/` (see below) or in this repo.
- The connect block the user pastes is the ONLY credential transport.
  `molab-connect.ps1 -Pasteline … -Save` caches it per sandbox id.

## Hardware

- Default: CPU-only (20 vCPU / 160 GiB RAM, no CUDA, no `/dev/nvidia*`,
  no `nvidia-smi`, no `libcuda`). Image torch is CPU-only
  (`2.14.0+cpu` at time of writing).
- GPU tier: attach via the notebook specs button in the UI header —
  RTX PRO 6000 Blackwell (96 GiB). Attaching recreates the sandbox.
- Post-GPU image: CUDA 13.0 runtime + **CPU-only wheel of torch 2.11**
  (`2.11.0+cu130`). This is why the notebook's `setup` cell exists: it
  repairs the sandbox venv to `torch==2.14.0+cu132` in a SUBPROCESS
  (`uv pip install --torch-backend=auto`) so the kernel never imports a
  mismatched torch.
- **After a repair, the session must be restarted from the UI** so the
  kernel re-imports torch fresh. The setup cell prints this reminder.
  Verify with: `/tmp/uv-venv/bin/python -c "import torch;
  print(torch.__version__, torch.cuda.is_available())"` → expect
  `2.14.0+cu132 True`.

## Auto-start limitation (verified in marimo 0.24.0 source)

molab sandboxes set `auto_instantiate=false` server-side and STRIP that
key from the notebook's PEP 723 header on upload — a notebook cannot
override it (security policy). Consequences:

- Cells are **loaded** on open but not run; the model is not resident
  until someone clicks run-all once.
- There is no supported way to have the chat hot on open. First-boot
  UX is: attach GPU → (env-repair fires if needed) → restart session →
  run-all → model downloads (~2 min, cached per-sandbox after) → chat
  live for the session's lifetime (~12 h max, 90-min idle close).

## Data: the HF bucket

- Bucket: `hf://buckets/Nubula/paddock/` (HuggingFace *Buckets* product,
  `hf://buckets/<org>/<name>/...` scheme — NOT `hf://datasets/`).
- `documents.json` (~262k docs, ~170 MB) is the same artifact the
  local `pgrag build-documents` emits. Publishing is **explicit**:
  after a local rebuild run `mise upload-docs` (writes
  `data/documents.json` → the bucket with your `HF_TOKEN`, then
  verifies remote size == local). A rebuild never auto-publishes — a
  bad local rebuild must not silently become the molab corpus on next
  sandbox boot.
- The notebook's `rag_index` downloads the bucket file and caches it to
  workspace `data/documents.json`; the next boot after an upload picks
  up the new corpus automatically.
- Historical: `Nubula/paddock/training/` holds unsloth JSONLs from the
  retired distillation route (removed in `4f95adc`). Untouched, but
  nothing in the current pipeline reads them.
- Access from the sandbox is anonymous (no HF_TOKEN in molab sandboxes);
  the bucket is public-read. Writes come only from the local repo via
  `mise upload-docs` (needs your `HF_TOKEN` with write scope).

## The notebook ↔ repo contract

- Source of truth for the notebook is THIS REPO
  (`notebooks/molab-mirror/notebook.py`, pushed to GitHub). molab loads
  it from GitHub:
  `https://molab.marimo.io/github/cprivitere/pg-rag-builder/blob/main/notebooks/molab-mirror/notebook.py`
  (private repo → requires the user's GitHub auth on molab).
- Sandbox-local edits via `cm.edit_cell` are LIVE-ONLY. Persist a cell
  edit by exporting the notebook (base64 via scratchpad) and committing
  to the repo. Never assume an edited cell survives a sandbox recreate.
- molab adds `marimo[mcp]>=0.24.0` + its own pinned deps to the PEP 723
  deps block on save; don't hand-craft the header, let the sandbox write it.

## Repo artifacts touched by molab work

- `data/golden/*.json` — golden eval cases; unrelated to molab, but the
  SYSTEM_PROMPT in the notebook mirrors the local pipeline's prompt
  conventions (context-grounded, no fabrication).
- `data/tmp_eval/` (deleted 2026-09-20) — was local scratch from the
  retired synthetic-QA generation route (teacher cell code, judge
  payloads, clean/unsloth JSONLs). The route was removed in `4f95adc`;
  the remaining run artifacts live on the HF bucket under
  `Nubula/paddock/training/` if ever needed again.

## Accessing the notebook from an agent

The `.agents/skills/molab-notebook/` skill covers the protocol (cached
URL+token, `molab-connect.ps1` delegation, cm API usage). This doc
covers the platform around it. Both together are the full picture.
