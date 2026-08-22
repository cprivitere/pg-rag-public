"""Embedder bake-off eval.

Deterministic fat-doc corpus (data/bakeoff_corpus.json) scored in each
model's own vector space: MRR@10 / hit@3 / hit@5 / recall@10 / NDCG@10 per
candidate plus measured per-PID VRAM, ranked report
(data/bakeoff_report.json).

Run:
  uv run python scripts/embed_eval.py --bakeoff [--only <substring>]

Candidates spawn like the VRAM probe: llama-server --embedding on :8085,
pooling per candidate (uniform naive unless the candidate declares a `pooling`
field, e.g. bge-m3/jina-v5 -> last). Docs truncated to 512 chars for
uniformity across short-context models. Metrics computed in each model's own
vector space (dims differ — no cross-model vectors).

Writes: data/bakeoff_report.json
"""
import argparse
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path

try:
    from scripts.embed_vram_probe import get_per_pid_vram
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from scripts.embed_vram_probe import get_per_pid_vram

DATA = Path(__file__).resolve().parent.parent / "data"

TRUNC = 512
BATCH = 32
PORT = 8085

BAKEOFF_CANDIDATES = [
    {
        "name": "embeddinggemma-300m-Q8",
        "hf": "unsloth/embeddinggemma-300M-GGUF:Q8_0",
        "pooling": "mean",
        "dims": 768,
        "ctx": 2048,
        "query_prefix": "task: search result | query: ",
        "doc_prefix": "title: none | text: ",
        "vram_mb": 329,
    },
    {
        "name": "jina-v5-nano-retrieval-Q8",
        "hf": "jinaai/jina-embeddings-v5-text-nano-retrieval-GGUF:Q8_0",
        "pooling": "last",
        "dims": 768,
        "ctx": 4096,
        "query_prefix": "Query: ",
        "doc_prefix": "Document: ",
        "vram_mb": 233,
    },
    {
        "name": "MiniLM-L6-Q4",
        "hf": "second-state/All-MiniLM-L6-v2-Embedding-GGUF:Q4_K_M",
        "pooling": "mean",
        "dims": 384,
        "ctx": 512,
        "query_prefix": None,
        "doc_prefix": None,
        "vram_mb": 21,
    },
    {
        "name": "bge-small-Q8",
        "hf": "ggml-org/bge-small-en-v1.5-Q8_0-GGUF:Q8_0",
        "pooling": "cls",
        "dims": 384,
        "ctx": 512,
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "doc_prefix": None,
        "vram_mb": 37,
    },
    {
        "name": "bge-small-f16",
        "hf": "unsloth/bge-small-en-v1.5-GGUF:f16",
        "pooling": "cls",
        "dims": 384,
        "ctx": 512,
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "doc_prefix": None,
        "vram_mb": 68,
    },
    {
        "name": "mxbai-xsmall-Q8",
        "hf": "twine-network/mxbai-embed-xsmall-v1-Q8_0-GGUF:Q8_0",
        "pooling": "mean",
        "dims": 384,
        "ctx": 4096,
        "query_prefix": None,
        "doc_prefix": None,
        "vram_mb": 31,
    },
    {
        "name": "bge-m3-Q8",
        "hf": "ggml-org/bge-m3-Q8_0-GGUF:Q8_0",
        "pooling": "cls",
        "dims": 1024,
        "ctx": 4096,
        "query_prefix": None,
        "doc_prefix": None,
        "vram_mb": 635,
    },
    {
        "name": "bge-m3-Q4",
        "hf": "gpustack/bge-m3-GGUF:Q4_K_M",
        "pooling": "cls",
        "dims": 1024,
        "ctx": 4096,
        "query_prefix": None,
        "doc_prefix": None,
        "vram_mb": 438,
    },
    {
        "name": "jina-v5-small-retrieval-Q8",
        "hf": "jinaai/jina-embeddings-v5-text-small-retrieval-GGUF:Q8_0",
        "pooling": "last",
        "dims": 1024,
        "ctx": 4096,
        "query_prefix": "Query: ",
        "doc_prefix": "Document: ",
        "vram_mb": 639,
    },
    {
        "name": "nomic-embed-v1.5-Q8",
        "hf": "nomic-ai/nomic-embed-text-v1.5-GGUF:Q8_0",
        "pooling": "mean",
        "dims": 768,
        "ctx": 2048,
        "query_prefix": "search_query: ",
        "doc_prefix": "search_document: ",
        "vram_mb": 146,
    },
    {
        "name": "embeddinggemma-300m-qat-Q8",
        "hf": "ggml-org/embeddinggemma-300m-qat-q8_0-GGUF:Q8_0",
        "pooling": "mean",
        "dims": 768,
        "ctx": 2048,
        "query_prefix": "task: search result | query: ",
        "doc_prefix": "title: none | text: ",
        "vram_mb": 329,
    },
]


def truncate(texts):
    return texts if TRUNC is None else [t[:TRUNC] for t in texts]


def _prefix(texts, prefix):
    if not prefix:
        return texts
    return [(prefix + t)[:TRUNC] for t in texts] if TRUNC is not None \
        else [prefix + t for t in texts]


def _post_embed(texts, url):
    import requests

    url = url if url.endswith("/embedding") else url.rstrip("/") + "/embedding"
    r = requests.post(url, json={"content": list(texts)}, timeout=300)
    r.raise_for_status()
    data = r.json()
    vecs = []
    for item in data:
        v = item["embedding"][0]
        if not v:
            raise RuntimeError("empty embedding from server")
        vecs.append(v)
    return vecs


def embed_all(texts, url, batch=BATCH, prefix=None):
    vecs = []
    truncated = truncate(_prefix(texts, prefix))
    for i in range(0, len(truncated), batch):
        for attempt in range(3):
            try:
                vecs.extend(_post_embed(truncated[i:i + batch], url))
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(5)
    assert len(vecs) == len(texts), (len(vecs), len(texts))
    return vecs


def _cosine_scores(query, docs):
    import numpy as np

    q = np.asarray(query, dtype=float)
    q = q / np.linalg.norm(q)
    M = np.asarray(docs, dtype=float)
    norms = np.linalg.norm(M, axis=1)
    M = M / norms[:, None]
    return list(M @ q)


def _ndcg(ranked, rel_set, k):
    """NDCG@k over binary relevance (rel_set), rank-discounted."""
    dcg = sum(1.0 / math.log2(r + 1) for r, idx in enumerate(ranked[:k], start=1)
              if idx in rel_set)
    nrel = min(len(rel_set), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, nrel + 1))
    return dcg / idcg if idcg else 0.0


def metrics(query_vecs, doc_vecs, labels, k=10, kinds=None):
    """Overall IR metrics; when kinds is a per-query tier list (e.g. 'hard'/
    'easy'), also compute each tier separately so the report shows whether
    the hard "needle" tier still discriminates."""
    def _accums():
        return {"mrr": 0.0, "ndcg": 0.0, "recall": 0.0, "h3": 0, "h5": 0, "h10": 0, "n": 0}

    all_acc = _accums()
    by_kind = {}
    for qv, rel, kd in zip(query_vecs, labels, kinds if kinds else [None] * len(labels)):
        rel_set = set(rel)
        scores = _cosine_scores(qv, doc_vecs)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        mrr = 0.0
        for rank, idx in enumerate(ranked, start=1):
            if idx in rel_set:
                mrr = 1.0 / rank
                break
        rec = sum(1 for i in ranked if i in rel_set) / len(rel_set)
        ndcg = _ndcg(ranked, rel_set, k)
        h3 = sum(1 for i in ranked[:3] if i in rel_set) > 0
        h5 = sum(1 for i in ranked[:5] if i in rel_set) > 0
        h10 = sum(1 for i in ranked[:10] if i in rel_set) > 0
        vals = [mrr, ndcg, rec, h3, h5, h10]
        for a in (all_acc, by_kind.setdefault(kd, _accums()) if kd else None):
            if a is None:
                continue
            a["mrr"] += vals[0]; a["ndcg"] += vals[1]; a["recall"] += vals[2]
            a["h3"] += vals[3]; a["h5"] += vals[4]; a["h10"] += vals[5]
            a["n"] += 1
    n = all_acc["n"]
    out = {"mrr10": round(all_acc["mrr"] / n, 4),
           "ndcg10": round(all_acc["ndcg"] / n, 4),
           "recall10": round(all_acc["recall"] / n, 4),
           "hit3": round(all_acc["h3"] / n, 4),
           "hit5": round(all_acc["h5"] / n, 4),
           "hit10": round(all_acc["h10"] / n, 4)}
    for kd, b in by_kind.items():
        bn = b["n"]
        out[f"{kd}_mrr10"] = round(b["mrr"] / bn, 4)
        out[f"{kd}_ndcg10"] = round(b["ndcg"] / bn, 4)
        out[f"{kd}_recall10"] = round(b["recall"] / bn, 4)
        out[f"{kd}_hit3"] = round(b["h3"] / bn, 4)
        out[f"{kd}_hit5"] = round(b["h5"] / bn, 4)
        out[f"{kd}_hit10"] = round(b["h10"] / bn, 4)
    return out


def _kill_port(port):
    """Kill any process listening on the given port."""
    try:
        conns = subprocess.run(
            ["pwsh", "-NoProfile", "-Command",
             f"(Get-NetTCPConnection -LocalPort {port} -ErrorAction SilentlyContinue).OwningProcess"],
            capture_output=True, text=True, timeout=10,
        )
        pids = set(conns.stdout.strip().split())
        for pid in pids:
            if pid.isdigit() and int(pid) > 0:
                subprocess.run(["taskkill", "/F", "/PID", pid],
                               capture_output=True, timeout=5)
    except Exception:
        pass


def requests_health(base_url):
    import requests

    r = requests.get(f"{base_url}/health", timeout=2)
    return r.status_code == 200


def run_model(subset, url, doc_prefix=None, query_prefix=None):
    doc_texts = [d["text"] for d in subset["docs"]]
    q_texts = [q["q"] for q in subset["queries"]]
    sys.stderr.write(f"  embedding {len(doc_texts)} docs at {url}...\n")
    doc_vecs = embed_all(doc_texts, url=url, prefix=doc_prefix)
    query_vecs = embed_all(q_texts, url=url, prefix=query_prefix)
    labels = label_indices(subset)
    kinds = ["hard" if str(q.get("id", "")).startswith("q_hard_") else "easy"
             for q in subset["queries"]]
    return metrics(query_vecs, doc_vecs, labels, kinds=kinds)


def label_indices(subset):
    idx = {d["id"]: i for i, d in enumerate(subset["docs"])}
    return [[idx[i] for i in q["relevant"]] for q in subset["queries"]]


def run_bakeoff(args):
    """Run bake-off: fat-doc corpus, measured per-PID VRAM, ranked report."""
    import requests

    corpus_path = DATA / "bakeoff_corpus.json"
    if not corpus_path.exists():
        sys.stderr.write("bakeoff corpus not found, run: uv run python scripts/bakeoff_corpus.py\n")
        return
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    subset = {"docs": corpus["docs"], "queries": [{"q": q["text"], "relevant": q["expected_doc_ids"],
                                                    "id": q.get("id")} for q in corpus["queries"]]}
    sys.stderr.write(f"bakeoff corpus: {len(subset['docs'])} docs, {len(subset['queries'])} queries\n")

    results = {"run_date": time.strftime("%Y-%m-%d"),
               "corpus_size": len(subset["docs"]),
               "ctx_chars": TRUNC or 512,
               "candidates": []}
    # Surface the corpus fingerprint so a changed/regenerated corpus is
    # detectable and runs are comparable (see bakeoff_corpus.py).
    fp = corpus.get("fingerprint", {})
    for k in ("n_docs", "n_queries", "seed", "golds_per_query_mean",
              "golds_per_query_min", "golds_per_query_max",
              "multi_gold_queries", "hard_queries", "content_hash"):
        if k in fp:
            results[k] = fp[k]
    sys.stderr.write(f"  corpus fingerprint: mean {fp.get('golds_per_query_mean')} "
                     f"golds/q ({fp.get('multi_gold_queries')}/{fp.get('n_queries')} "
                     f"multi-gold), hash {fp.get('content_hash')}\n")

    for cand in BAKEOFF_CANDIDATES:
        if args.only and args.only not in cand["name"]:
            continue
        sys.stderr.write(f"\n--- {cand['name']} ---\n")
        info = spawn_candidate_bakeoff(cand)
        if not info.get("ok", True):
            sys.stderr.write(f"  FAILED: {info.get('note')}\n")
            results["candidates"].append({"name": cand["name"], "ok": False,
                                           "note": info.get("note")})
            continue

        proc = info["proc"]
        base_url = info["base_url"]
        try:
            # Embed and score
            m = run_model(subset, base_url,
                          doc_prefix=cand.get("doc_prefix"),
                          query_prefix=cand.get("query_prefix"))
            # Parse server log for memory info
            log_path = DATA / f"embed_eval_{cand['name'].split()[0]}.log"
            server_log = parse_server_log(log_path)

            # Measured resident VRAM for this candidate's server, per-PID and
            # settled over samples: single WDDM reads are noisy and back-to-back
            # same-shape candidates recycle driver pages (see vram_sweep).
            samples = []
            for _ in range(4):
                v = get_per_pid_vram(proc.pid)
                if v is not None:
                    samples.append(v)
                time.sleep(0.3)
            vram_mb = max(samples) if samples else None

            entry = {"name": cand["name"], "dims": cand["dims"],
                     "quant": cand["hf"].split(":")[-1],
                     "vram_mb": vram_mb, **m, "server_log": server_log}
            results["candidates"].append(entry)
            vram_txt = f"{vram_mb:.0f}MB" if vram_mb is not None else "n/a"
            sys.stderr.write(f"  {cand['name']}: mrr={m['mrr10']}, vram={vram_txt}\n")
        finally:
            try:
                proc.kill()
                proc.wait(timeout=5)
            except Exception:
                pass
            # Kill any lingering llama-server on the port
            _kill_port(PORT)
            time.sleep(2)

    # Rank by MRR@10, tie-break by VRAM
    ranked = sorted([c for c in results["candidates"] if c.get("ok", True)],
                    key=lambda c: (-c.get("mrr10", 0), c.get("vram_mb") or 9999))
    if ranked:
        results["winner"] = ranked[0]["name"]
        results["runner_up"] = ranked[1]["name"] if len(ranked) > 1 else None

    # Write report
    report_path = DATA / "bakeoff_report.json"
    report_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    sys.stderr.write(f"\nwrote {report_path}\n")

    # Print ranked table
    print(f"\n{'Name':<30} {'Dims':>5} {'Quant':<8} {'VRAM MB':>8} {'MRR@10':>8} {'Recall@10':>9} {'NDCG@10':>8} {'HardMRR':>8} {'Hit@3':>7} {'Hit@5':>7} {'Hit@10':>8}")
    print("-" * 118)
    for c in ranked:
        print(f"{c['name']:<30} {c['dims']:>5} {c['quant']:<8} {c['vram_mb']:>8} "
              f"{c.get('mrr10', 0):>8.4f} {c.get('recall10', 0):>9.4f} {c.get('ndcg10', 0):>8.4f} "
              f"{c.get('hard_mrr10', 0):>8.4f} "
              f"{c.get('hit3', 0):>7.4f} {c.get('hit5', 0):>7.4f} {c.get('hit10', 0):>8.4f}")
    if results.get("winner"):
        print(f"\nWinner: {results['winner']}")


def spawn_candidate_bakeoff(cand):
    """Spawn llama-server for a bake-off candidate."""
    log_file = DATA / f"embed_eval_{cand['name'].split()[0]}.log"
    cmd = ["llama-server", "-hf", cand["hf"],
           "--host", "0.0.0.0", "--port", str(PORT),
           "--embedding", "--pooling", cand["pooling"], "-ngl", "99",
           "-c", str(cand["ctx"]), "-v",
           "--log-file", str(log_file)]
    sys.stderr.write(f"  spawning {cand['name']}...\n")
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base_url = f"http://localhost:{PORT}"
    try:
        for i in range(600):
            time.sleep(1)
            try:
                if requests_health(base_url):
                    sys.stderr.write(f"  server up after {i+1}s\n")
                    return {"base_url": base_url, "proc": proc}
            except Exception as e:
                if i % 30 == 0:
                    sys.stderr.write(f"  health check error at {i+1}s: {e}\n")
                continue
    except Exception:
        pass
    return {"ok": False, "note": "start timeout"}


def parse_server_log(log_path):
    """Extract memory breakdown from llama-server verbose startup log."""
    if not log_path.exists():
        return "no log file"
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        mem_lines = [l for l in lines if any(k in l.lower() for k in
                     ("kv", "buffer", "memory", "backend", "offload", "model size"))]
        return " | ".join(mem_lines[-10:]) if mem_lines else "no memory info in log"
    except Exception as e:
        return f"log parse error: {e}"


def main():
    parser = argparse.ArgumentParser(description="Embedder bake-off eval")
    parser.add_argument("--bakeoff", action="store_true",
                        help="run bake-off: fat-doc corpus, measured per-PID VRAM, ranked report")
    parser.add_argument("--only", default=None,
                        help="run only this candidate (substring match)")
    args = parser.parse_args()
    run_bakeoff(args)


if __name__ == "__main__":
    main()