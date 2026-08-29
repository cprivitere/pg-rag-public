"""VRAM launch-flag sweep for the production models, measured in isolation.

For each server/model, spin each flag variant on a throwaway port
(embed :8084, llm :8085, reranker :8086), wait for /health 200, warm one
request, and sample the adapter VRAM delta vs baseline. One variant at a time,
so the adapter total cleanly attributes to it. Production ports are never
started/stopped here.

Reuses from existing code:
- embed_vram_probe._url_args / _total_vram_mb (adapter total is correct for
  isolation sweeps where only one llama-server variant runs on top of a
  constant production background).
- embed_eval.spawn/stop/health patterns (adapted per server kind).

Run: uv run python scripts/vram_sweep.py [--server embed|llm|reranker] [--only <variant>]
Writes: data/vram_sweep.json
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
import tomllib

import requests

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

try:
    from scripts.embed_vram_probe import _url_args, _total_vram_mb, get_per_pid_vram  # noqa: E402
except ModuleNotFoundError:
    sys.path.insert(0, str(ROOT))
    from scripts.embed_vram_probe import _url_args, _total_vram_mb, get_per_pid_vram

# Production model refs — single source is mise.toml [env]. Read the live
# values so a model swap propagates into the sweep with no literal drift.
# Falls back to the current values if mise.toml is unreadable.
_LLM_FALLBACK = "ornith-ai/Ornith-1.5-9B-GGUF:Q4_K_M"
_RERANK_FALLBACK = "gpustack/bge-reranker-v2-m3-GGUF:Q4_K_M"
_EMBED_FALLBACK = "unsloth/bge-small-en-v1.5-GGUF:f16"
try:
    _model_env = tomllib.loads((ROOT / "mise.toml").read_text(encoding="utf-8")).get("env", {})
    LLM_MODEL = _model_env.get("LLM_MODEL") or _LLM_FALLBACK
    RERANK_MODEL = _model_env.get("RERANK_MODEL") or _RERANK_FALLBACK
    EMBED_MODEL = _model_env.get("EMBED_MODEL") or _EMBED_FALLBACK
except (OSError, tomllib.TOMLDecodeError):
    LLM_MODEL, RERANK_MODEL, EMBED_MODEL = _LLM_FALLBACK, _RERANK_FALLBACK, _EMBED_FALLBACK

PORTS = {"embed": 8084, "llm": 8085, "reranker": 8086}

WARM_EMBED = "Project Gorgon gardening skill recipe for pig poop fertilizer"
WARM_LLM = "Hi"
WARM_RERANK_Q = "how does death work"
WARM_RERANK_D = ["Death is a status effect in Project Gorgon."]


def llm_variants():
    # The swept model is the production Ornith-1.5-9B (LLM_MODEL, read
    # from mise.toml [env]). The variants MUST measure ITS launch config — native
    # thinking capped at @4096(no MTP draft;the production LLM_FLAGS carries no
    # --spec-type). The swept dimensions are context window + flash-attention only.
    base = ["-ngl", "999", "-np", "1", "--reasoning-budget", "4096"]
    def v(name, fa="auto", c=16384):
        flags = list(base)
        if fa:
            flags += ["-fa", fa]
        flags += ["-c", str(c)]
        return (name, flags)
    return [
        v("baseline", fa="on", c=16384),
        v("ctx-8192", fa="on", c=8192),
        v("ctx-12288", fa="on", c=12288),
        v("fa-off", fa="off", c=16384),
    ]


def rerank_variants():
    base = ["--reranking", "--pooling", "rank", "--alias",
            "bge-reranker-v2-m3", "-ngl", "99"]
    def v(name, c, batch):
        return (name, base + ["-c", str(c), "-b", str(batch), "-ub", str(batch)])
    return [
        v("baseline", 32768, 8192),
        v("ctx-8192", 8192, 8192),
        v("ctx-16384", 16384, 8192),
        v("batch-4096", 32768, 4096),
    ]


def embed_variants():
    base = ["--embedding", "--pooling", "mean", "-ngl", "99", "-np", "1"]
    def v(name, c, batch, ubatch):
        return (name, base + ["-c", str(c), "-b", str(batch),
                              "--ubatch-size", str(ubatch)])
    return [
        v("baseline", 4096, 4096, 4096),
        v("ctx-2048", 2048, 4096, 4096),
        v("batch-2048", 4096, 2048, 2048),
        v("batch-8192", 4096, 8192, 8192),
    ]


VARIANT_FN = {"embed": embed_variants, "llm": llm_variants,
              "reranker": rerank_variants}


def _spawn(kind, name, flags, port):
    log_path = DATA / f"vram_sweep_{kind}_{name}.log"
    log_fh = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(
        ["llama-server", *_url_args(_model_of(kind)),
         *flags, "--host", "127.0.0.1", "--port", str(port)],
        stdout=log_fh, stderr=subprocess.STDOUT)
    return proc, log_fh


def _model_of(kind):
    return {"embed": EMBED_MODEL, "llm": LLM_MODEL, "reranker": RERANK_MODEL}[kind]


def _warm(kind, base_url):
    try:
        if kind == "embed":
            requests.post(f"{base_url}/embedding",
                          json={"content": WARM_EMBED}, timeout=60)
        elif kind == "llm":
            requests.post(f"{base_url}/completion",
                          json={"prompt": WARM_LLM, "n_predict": 1,
                                "stream": False}, timeout=120)
        else:
            requests.post(f"{base_url}/rerank",
                          json={"query": WARM_RERANK_Q,
                                "documents": [WARM_RERANK_D]}, timeout=60)
    except Exception as e:
        sys.stderr.write(f"    warm request failed: {e}\n")


def _stop(proc, log_fh):
    if proc is not None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
        time.sleep(2)
    if log_fh is not None:
        log_fh.close()


def run_variant(kind, name, flags, port):
    base = _total_vram_mb()
    proc = log_fh = None
    try:
        proc, log_fh = _spawn(kind, name, flags, port)
        base_url = f"http://127.0.0.1:{port}"
        up = False
        for _ in range(300):
            time.sleep(1)
            try:
                if requests.get(f"{base_url}/health", timeout=2).status_code == 200:
                    up = True
                    break
            except Exception:
                continue
        if not up:
            return {"server": kind, "variant": name, "flags": flags,
                    "vram_mb": None, "per_pid_mb": None,
                    "ok": False, "note": "start timeout"}
        _warm(kind, base_url)
        time.sleep(3)
        # Decision metric: per-PID committed for this spawn (process-scoped;
        # immune to driver page retention across same-shape variants). The
        # adapter-total delta is kept as a secondary resident-pressure signal,
        # but successive variants reuse weight pages, so it under-reports.
        per_pid_mb = get_per_pid_vram(proc.pid)
        samples = []
        for _ in range(5):
            v = _total_vram_mb()
            if v is not None:
                samples.append(v)
            time.sleep(0.5)
        post = max(samples) if samples else None
        vram = (post - base) if (post is not None and base is not None) else None
        return {"server": kind, "variant": name, "flags": flags,
                "vram_mb": vram, "per_pid_mb": per_pid_mb,
                "ok": per_pid_mb is not None, "note": None}
    finally:
        _stop(proc, log_fh)


def main():
    ap = argparse.ArgumentParser(description="VRAM launch-flag sweep, in isolation")
    ap.add_argument("--server", choices=sorted(VARIANT_FN), default=None,
                    help="sweep only this server")
    ap.add_argument("--only", default=None, help="sweep only this variant name")
    args = ap.parse_args()

    servers = [s for s in VARIANT_FN if args.server is None or s == args.server]
    results = []
    for kind in servers:
        variants = [(n, f) for n, f in VARIANT_FN[kind]()
                     if args.only is None or n == args.only]
        sys.stderr.write(f"-- {kind} ({len(variants)} variants on :{PORTS[kind]})\n")
        for name, flags in variants:
            rec = run_variant(kind, name, flags, PORTS[kind])
            v = rec["per_pid_mb"]
            a = rec["vram_mb"]
            sys.stderr.write(f"   {name:12} "
                             f"per-pid {('%.0f MB' % v) if v is not None else 'FAIL':>10}  "
                             f"adapter {('%.0f MB' % a) if a is not None else '—':>10}  "
                             f"{rec['note'] or ''}\n")
            results.append(rec)

    DATA.mkdir(exist_ok=True)
    out_path = DATA / "vram_sweep.json"
    prev = []
    if out_path.exists():
        try:
            prev = json.loads(out_path.read_text(encoding="utf-8"))
        except Exception:
            prev = []
    seen = {(r["server"], r["variant"]) for r in results}
    merged = prev + results
    if prev:
        # replace stale entries for the servers we just swept, keep others
        merged = [r for r in prev if (r["server"], r["variant"]) not in seen] + results
    merged.sort(key=lambda r: (r["server"], r["variant"]))
    out_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print(f"wrote {out_path} ({len(merged)} entries)")


if __name__ == "__main__":
    main()
