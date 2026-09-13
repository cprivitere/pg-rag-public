"""Per-server VRAM profile for the production inference stack.

Measures the true VRAM cost of the current configuration for each production
server in isolation on its real port: embed :8081, llm :8080, reranker :8082.

For each server: record the current adapter total, start it via its existing
.mise/tasks/*-start.ps1 (or measure in place if it is already up), wait for
/health 200, warm one request, sleep, then sample per-PID + adapter totals.
vram_mb = adapter delta (resident, trustworthy) when we started the server;
when it was already up we fall back to per-PID dedicated bytes (committed,
flagged started_by_us=false).

Wraps scripts/vram.ps1 — the per-PID/adapter measurement primitive — for the
sampling; never reimplements the GPU counters.

Run: uv run python scripts/vram_profile.py [--only embed|llm|reranker]
Writes: data/vram_profile.json
"""

import argparse
import contextlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SCRIPTS = ROOT / "scripts"
TASKS = ROOT / ".mise" / "tasks"

VRAM_PS1 = SCRIPTS / "vram.ps1"

# (server, port, start-timeout-seconds)
SERVERS = [
    ("embed", 8081, 180),
    ("llm", 8080, 300),
    ("reranker", 8082, 180),
]

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _run_vram_ps1():
    """Run vram.ps1 -Raw -Samples 3 -Filter llama. Return (per_pid dict, adapter bytes)."""
    out = subprocess.run(
        [
            "pwsh",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(VRAM_PS1),
            "-Raw",
            "-Samples",
            "3",
            "-Filter",
            "llama",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    text = _ANSI.sub("", out.stdout)
    by_pid = {}
    for m in re.finditer(r"PID\s+(\d+)\s+(\S+)\s+([\d,]+) B", text):
        by_pid[int(m.group(1))] = int(m.group(3).replace(",", ""))
    adapter = 0.0
    for m in re.finditer(r"LUID\s+\S+\s+([\d,]+) B", text):
        adapter += int(m.group(1).replace(",", ""))
    return by_pid, adapter


def _port_pid(port):
    """Owning PID of the listener on `port`, or None."""
    out = subprocess.run(
        [
            "pwsh",
            "-NoProfile",
            "-Command",
            (
                f"Get-NetTCPConnection -LocalPort {port} -State Listen "
                "-ErrorAction SilentlyContinue | "
                "Select-Object -ExpandProperty OwningProcess"
            ),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    pids = [int(p) for p in out.stdout.split() if p.strip().isdigit()]
    return pids[0] if pids else None


def _health(base_url):
    try:
        return requests.get(f"{base_url}/health", timeout=2).status_code == 200
    except Exception:
        return False


def _wait_health(base_url, timeout):
    for _ in range(timeout):
        time.sleep(1)
        if _health(base_url):
            return True
    return False


def _warm(server, base_url):
    """One request so the server holds its full KV/compute buffers before sampling."""
    try:
        if server == "embed":
            requests.post(
                f"{base_url}/embedding",
                json={"content": "Project Gorgon gardening skill recipe"},
                timeout=60,
            )
        elif server == "llm":
            requests.post(
                f"{base_url}/completion",
                json={"prompt": "Hi", "n_predict": 1, "stream": False},
                timeout=120,
            )
        else:
            requests.post(
                f"{base_url}/rerank",
                json={
                    "query": "how does death work",
                    "documents": ["Death is a status effect in Project Gorgon."],
                },
                timeout=60,
            )
    except Exception as e:  # warm is best-effort; sampling still proceeds
        sys.stderr.write(f"  warm {server} request failed: {e}\n")


def _start_via_task(server):
    script = TASKS / f"{server}-start.ps1"
    if not script.exists():
        sys.stderr.write(f"  [error] start script missing: {script}\n")
        return False
    r = subprocess.run(
        ["pwsh", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if r.returncode != 0:
        sys.stderr.write(f"  [warn] {server}-start.ps1 rc={r.returncode}: {r.stderr.strip()}\n")
    return True


def _profile_one(server, port, timeout):
    base_url = f"http://127.0.0.1:{port}"

    if _health(base_url):
        # Already up: no clean adapter baseline. Attribute via per-PID bytes.
        by_pid, _adapter = _run_vram_ps1()
        pid = _port_pid(port)
        per_pid = by_pid.get(pid) if pid is not None else None
        vram_mb = (per_pid / 1e6) if per_pid else None
        note = (
            "no llama process found"
            if per_pid is None
            else "already up; per-PID (committed) attribution"
        )
        sys.stderr.write(f"  [{server}] already up on {port}; measured in place\n")
        return {
            "server": server,
            "port": port,
            "pid": pid,
            "vram_mb": vram_mb,
            "per_pid_mb": (per_pid / 1e6) if per_pid else None,
            "started_by_us": False,
            "ok": per_pid is not None,
            "note": note,
        }

    base_adapter = _run_vram_ps1()[1]
    sys.stderr.write(
        f"  [{server}] starting on {port} (baseline adapter {base_adapter / 1e6:.0f} MB)...\n"
    )
    if not _start_via_task(server):
        return {
            "server": server,
            "port": port,
            "pid": None,
            "vram_mb": None,
            "per_pid_mb": None,
            "started_by_us": False,
            "ok": False,
            "note": f"start script missing for {server}",
        }

    if not _wait_health(base_url, timeout):
        return {
            "server": server,
            "port": port,
            "pid": _port_pid(port),
            "vram_mb": None,
            "per_pid_mb": None,
            "started_by_us": True,
            "ok": False,
            "note": "start timeout",
        }

    pid = _port_pid(port)
    _warm(server, base_url)
    time.sleep(3)
    by_pid, adapter = _run_vram_ps1()
    per_pid = by_pid.get(pid) if pid is not None else None
    vram_mb = (adapter - base_adapter) / 1e6
    note = None
    if per_pid is None:
        note = "no llama process found"
    sys.stderr.write(
        f"  [{server}] adapter delta {vram_mb:.0f} MB, "
        f"per-PID {(per_pid / 1e6) if per_pid else 0:.0f} MB\n"
    )
    return {
        "server": server,
        "port": port,
        "pid": pid,
        "vram_mb": vram_mb,
        "per_pid_mb": (per_pid / 1e6) if per_pid else None,
        "started_by_us": True,
        "ok": vram_mb is not None,
        "note": note,
    }


def _stop(pid):
    if pid is None:
        return
    with contextlib.suppress(Exception):
        subprocess.run(
            ["pwsh", "-NoProfile", "-Command", f"Stop-Process -Id {pid} -Force"],
            capture_output=True,
            timeout=30,
        )


def main():
    ap = argparse.ArgumentParser(description="Per-server VRAM profile of the production stack")
    ap.add_argument(
        "--only", choices=[s[0] for s in SERVERS], default=None, help="profile only this server"
    )
    args = ap.parse_args()

    servers = [s for s in SERVERS if args.only is None or s[0] == args.only]
    if any(_health(f"http://127.0.0.1:{port}") for _, port, _ in servers):
        sys.stderr.write(
            "[warn] at least one target server already up: adapter-delta "
            "attribution will be per-PID/less accurate for it\n"
        )

    results = []
    started_pids = []
    try:
        for server, port, timeout in servers:
            rec = _profile_one(server, port, timeout)
            results.append(rec)
            if rec["started_by_us"] and rec["ok"] and rec["pid"]:
                started_pids.append(rec["pid"])
    finally:
        for pid in started_pids:
            sys.stderr.write(f"  stopping server we started (PID {pid})\n")
            _stop(pid)

    DATA.mkdir(exist_ok=True)
    out_path = DATA / "vram_profile.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    for r in results:
        vram = r["vram_mb"]
        print(
            f"{r['server']:9} :{r['port']:<5} "
            f"VRAM {(f'{vram:.0f} MB' if vram is not None else 'n/a'):>10}  "
            f"pid={r['pid']}  started_by_us={r['started_by_us']}  {r['note'] or ''}"
        )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
