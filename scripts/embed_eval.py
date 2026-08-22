"""Embedder subset eval (SPEC T86).

Deterministic 1.2k-doc subset + 13 sources-derived queries
(data/eval_subset.json), MRR@10 / hit@3 / hit@5 / recall@10 per embedder
vs jina baseline (:8081).

Run:
  uv run python scripts/embed_eval.py --gen-subset   # (re)build subset
  uv run python scripts/embed_eval.py                # jina baseline only
  uv run python scripts/embed_eval.py --sweep        # + spawn candidates on :8085

Candidate servers spawn like the VRAM probe: llama-server --embedding
--pooling mean (uniform naive pooling for every candidate) unless the
candidate declares a `pooling` field (e.g. bge-m3/jina-v5 -> last).
Baseline runs its production pooling (jina-v5 requires --pooling last).
Docs truncated to 512 chars for uniformity across short-context models;
pass --no-trunc to run at corpus chunk ceiling instead. Metrics computed
in each model's own vector space (dims differ — no cross-model vectors).

Writes: data/eval_subset.json, data/embed_eval.json
"""
import argparse
import hashlib
import json
import math
import random
import re
import subprocess
import sys
import time
from pathlib import Path

try:
    from scripts.embed_vram_probe import CANDIDATES, _url_args, get_base_url, get_per_pid_vram
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from scripts.embed_vram_probe import CANDIDATES, _url_args, get_base_url, get_per_pid_vram

DATA = Path(__file__).resolve().parent.parent / "data"
DOCS_PATH = DATA / "documents.json"
SUBSET_PATH = DATA / "eval_subset.json"
OUT_PATH = DATA / "embed_eval.json"

SEED = 42
TARGET = 1200
N_QUERIES = 13
TRUNC = 512
BATCH = 32
PORT = 8085
HOST = "Scamper"
BASELINE_URL = "http://localhost:8081/embedding"

CHUNK_SUFFIX = re.compile(r"_chunk_\d+$")

SLOTS = [
    ("recipe", "How much XP does {name} give?"),
    ("recipe", "What do I need to craft {name}?"),
    ("item", "What does the item {name} do?"),
    ("item", None),
    ("ability", "What does the ability {name} do?"),
    ("skillprofile", "Tell me everything about the {name} skill"),
    ("quest", "How do I start the quest {name}?"),
    ("wiki", "What does the wiki say about {title}?"),
    ("effect", "What effect does {name} have?"),
    ("tsys", "What does {name} grant?"),
    ("attribute", "What does the stat {name} do?"),
    ("itemuse", "What can {name} be used on?"),
    ("source", "What do I learn from {name}?"),
]

JINA_CURRENT = "jina-v5-text-small (current)"

BAKEOFF_CANDIDATES = [
    {
        "name": "embeddinggemma-300m-Q8",
        "hf": "unsloth/embeddinggemma-300m-GGUF:Q8_0",
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
        "ctx": 8192,
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
        "pooling": "mean",
        "dims": 384,
        "ctx": 512,
        "query_prefix": None,
        "doc_prefix": None,
        "vram_mb": 37,
    },
]


def load_docs():
    with open(DOCS_PATH, encoding="utf-8") as f:
        return json.load(f)


def base_key(doc_id):
    return CHUNK_SUFFIX.sub("", doc_id)


def doc_name(doc):
    meta = doc.get("metadata", {})
    return meta.get("name") or base_key(doc["id"])


def _type_budgets(counts, target):
    weights = {t: (c ** 0.5) for t, c in counts.items()}
    total_w = sum(weights.values())
    budgets = {}
    for t, w in weights.items():
        b = max(3, int(target * w / total_w))
        budgets[t] = min(counts[t], b)
    remaining = target - sum(budgets.values())
    assert remaining >= 0, (counts, budgets)
    rng = random.Random(SEED)
    order = [t for t, _ in sorted(counts.items(), key=lambda kv: kv[1])]
    while remaining:
        progressed = False
        for t in order:
            if remaining and budgets[t] < counts[t]:
                budgets[t] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break
    return budgets


def _pick_doc(rng, ids_by_type, used, type_):
    pool = [i for i in ids_by_type[type_] if i not in used]
    if not pool:
        return None
    return rng.choice(pool)


def _family_labels(subset_ids, doc_id):
    key = base_key(doc_id)
    return sorted(i for i in subset_ids if base_key(i) == key)


def _produces_recipes(subset_docs_by_id, item_id, item_name):
    if not item_name:
        return []
    found = []
    for did, doc in subset_docs_by_id.items():
        if doc["type"] != "recipe" or did == item_id:
            continue
        text = doc["text"]
        tail = text.partition("Produces:")[2]
        for line in tail.splitlines():
            if line.startswith("-") and item_name.lower() in line.lower():
                found.append(did)
                break
        if len(found) >= 5:
            break
    return sorted(found)


def gen_subset(docs=None, target=TARGET):
    if docs is None:
        docs = load_docs()
    docs = [d for d in docs if d.get("text", "").strip()]
    target = min(target, len(docs))
    counts = {}
    ids_by_type = {}
    for doc in docs:
        t = doc["type"]
        counts[t] = counts.get(t, 0) + 1
        ids_by_type.setdefault(t, []).append(doc["id"])

    budgets = _type_budgets(counts, target)
    rng = random.Random(SEED)
    order = {}
    for t, ids in ids_by_type.items():
        order[t] = rng.sample(ids, budgets[t]) if len(ids) > budgets[t] else list(ids)

    subset_ids = [i for t in order for i in order[t]]
    assert len(subset_ids) == sum(budgets.values())
    assert len(set(subset_ids)) == len(subset_ids)
    by_id = {d["id"]: d for d in docs}

    docs_out = []
    for did in subset_ids:
        d = by_id[did]
        docs_out.append({"id": did, "type": d["type"], "name": doc_name(d), "text": d["text"]})

    queries = []
    used = set()
    attempts = 0
    while len(queries) < N_QUERIES and attempts < 200:
        attempts += 1
        slot = SLOTS[len(queries)]
        type_, template = slot
        pick = _pick_doc(rng, order, used, type_)
        if pick is None:
            if len(queries) < N_QUERIES:
                break
        used.add(pick)
        doc = by_id[pick]
        name = doc_name(doc)
        if type_ == "wiki":
            title = base_key(doc["id"])
            if title.startswith("wiki_"):
                title = title[len("wiki_"):]
            q = template.format(title=title.replace("_", " ").strip() or name)
        elif template is None:
            item_name = name
            recipes = _produces_recipes({d["id"]: d for d in docs_out}, pick, item_name)
            if recipes:
                q = f"What recipes can produce {item_name}?"
                labels = recipes + [pick]
            else:
                q = f"What does the item {name} do?"
                labels = _family_labels(subset_ids, pick)
        else:
            q = template.format(name=name)
            labels = _family_labels(subset_ids, pick)
        if not labels:
            continue
        queries.append({"q": q, "relevant": sorted(labels)})

    if len(queries) != N_QUERIES:
        raise RuntimeError(f"gen failed: {len(queries)}/{N_QUERIES} queries")
    for q in queries:
        assert all(i in by_id for i in q["relevant"])
    return {"docs": docs_out, "queries": queries}


def save_subset(subset):
    SUBSET_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = SUBSET_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(subset, indent=1, ensure_ascii=False), encoding="utf-8")
    tmp.replace(SUBSET_PATH)


def load_subset():
    return json.loads(SUBSET_PATH.read_text(encoding="utf-8"))


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


def embed_all(texts, url=BASELINE_URL, batch=BATCH, prefix=None):
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


def metrics(query_vecs, doc_vecs, labels, k=10):
    total_mrr = 0.0
    hit3 = hit5 = hit10 = 0
    recall_sum = 0.0
    ndcg_sum = 0.0
    for qv, rel in zip(query_vecs, labels):
        rel_set = set(rel)
        scores = _cosine_scores(qv, doc_vecs)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        for rank, idx in enumerate(ranked, start=1):
            if idx in rel_set:
                total_mrr += 1.0 / rank
                break
        top3 = ranked[:3]
        top5 = ranked[:5]
        if any(i in rel_set for i in top3):
            hit3 += 1
        if any(i in rel_set for i in top5):
            hit5 += 1
        if any(i in rel_set for i in ranked[:10]):
            hit10 += 1
        recall_sum += len([i for i in ranked if i in rel_set]) / len(rel_set)
        ndcg_sum += _ndcg(ranked, rel_set, k)
    n = len(query_vecs)
    return {
        "mrr10": round(total_mrr / n, 4),
        "ndcg10": round(ndcg_sum / n, 4),
        "recall10": round(recall_sum / n, 4),
        "hit3": round(hit3 / n, 4),
        "hit5": round(hit5 / n, 4),
        "hit10": round(hit10 / n, 4),
    }


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


def resolve_base_url(host=HOST, port=PORT):
    import socket

    try:
        addrs = socket.getaddrinfo(host, port, socket.AF_INET6, socket.SOCK_STREAM)
        for res in addrs:
            _af, _st, _p, _cn, sa = res
            ip, _port, _flow, scope = sa
            if scope:
                return f"http://[{ip}%{scope}]:{port}"
            return f"http://[{ip}]:{port}"
    except Exception:
        pass
    return f"http://{host}:{port}"


def spawn_candidate(cand, pooling="mean", ctx=4096):
    log_file = DATA / f"embed_eval_{cand['name'].split()[0]}.log"
    cmd = ["llama-server", *_url_args(cand["url"], cand["local"]),
           "--host", HOST, "--port", str(PORT),
           "--embedding", "--pooling", pooling, "-ngl", "99",
           "-c", str(ctx), "-np", "1", "--ubatch-size", cand.get("ubatch", "512"),
           "--log-file", str(log_file)]
    sys.stderr.write(f"  spawning {cand['name']}...\n")
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base_url = resolve_base_url()
    try:
        for _ in range(300):
            time.sleep(1)
            try:
                if requests_health(base_url):
                    return {"base_url": base_url, "proc": proc}
            except Exception:
                continue
    except Exception:
        pass
    return stop_candidate(cand)

def stop_candidate(cand, proc=None):
    if proc is not None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
        time.sleep(2)
    return {"name": cand["name"], "ok": False, "note": "start timeout"}


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
    return metrics(query_vecs, doc_vecs, labels)


def label_indices(subset):
    idx = {d["id"]: i for i, d in enumerate(subset["docs"])}
    return [[idx[i] for i in q["relevant"]] for q in subset["queries"]]


def with_deltas(cand_metrics, base):
    out = dict(cand_metrics)
    for k, v in base.items():
        out[f"delta_{k}"] = round(cand_metrics[k] - v, 4)
    return out


def evaluate_candidate(cand, subset, pooling="mean", ctx=4096, doc_prefix=None, query_prefix=None):
    info = spawn_candidate(cand, pooling=pooling, ctx=ctx)
    if not info.get("ok", True):
        return {"ok": False, "note": info.get("note")}
    proc = info["proc"]
    try:
        m = run_model(subset, info["base_url"], doc_prefix=doc_prefix, query_prefix=query_prefix)
        return m
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
        time.sleep(2)


def run_bakeoff(args):
    """Run bake-off: fat-doc corpus, measured per-PID VRAM, ranked report."""
    import requests

    corpus_path = DATA / "bakeoff_corpus.json"
    if not corpus_path.exists():
        sys.stderr.write("bakeoff corpus not found, run: uv run python scripts/bakeoff_corpus.py\n")
        return
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    subset = {"docs": corpus["docs"], "queries": [{"q": q["text"], "relevant": q["expected_doc_ids"]}
                                                    for q in corpus["queries"]]}
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
    print(f"\n{'Name':<30} {'Dims':>5} {'Quant':<8} {'VRAM MB':>8} {'MRR@10':>8} {'Recall@10':>9} {'NDCG@10':>8} {'Hit@3':>7} {'Hit@5':>7} {'Hit@10':>8}")
    print("-" * 110)
    for c in ranked:
        print(f"{c['name']:<30} {c['dims']:>5} {c['quant']:<8} {c['vram_mb']:>8} "
              f"{c.get('mrr10', 0):>8.4f} {c.get('recall10', 0):>9.4f} {c.get('ndcg10', 0):>8.4f} "
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


def get_gguf_size_mb(cand):
    """Get GGUF file size in MB. Check local HF cache first, fall back to estimate."""
    import os
    # Try local HF cache
    cache_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    # Search for the file in cache
    hf_repo = cand.get("gghf_repo", "")
    hf_file = cand.get("gghf_file", "")
    if hf_repo and hf_file:
        # Try common HF cache paths
        for snapshots_dir in [Path(cache_home) / "hub" / f"models--{hf_repo.replace('/', '--')}" / "snapshots"]:
            if snapshots_dir.exists():
                for snap in snapshots_dir.iterdir():
                    fpath = snap / hf_file
                    if fpath.exists():
                        return round(fpath.stat().st_size / (1024 * 1024), 1)
    # Fallback: return None (user can check HF card)
    return None


def main():
    parser = argparse.ArgumentParser(description="Embedder subset eval (SPEC T86)")
    parser.add_argument("--gen-subset", action="store_true", help="regenerate eval_subset.json")
    parser.add_argument("--sweep", action="store_true", help="spawn + evaluate candidate embedders")
    parser.add_argument("--only", default=None, help="run only this candidate (substring match)")
    parser.add_argument("--mxbai-prefix", action="store_true",
                        help="instruct prefixes (search_document:/search_query:) for mxbai candidates")
    parser.add_argument("--no-trunc", action="store_true",
                        help="do not truncate docs to 512 chars (corpus chunk ceiling, SPEC B29)")
    parser.add_argument("--ctx", type=int, default=4096,
                        help="server -c context size for ALL models (uniform token ceiling)")
    parser.add_argument("--trunc-chars", type=int, default=None,
                        help="truncate docs to N chars client-side (safe under ctx; server 400s on over-ctx)")
    parser.add_argument("--bakeoff", action="store_true",
                        help="run bake-off: fat-doc corpus, GGUF size as VRAM, ranked report")
    args = parser.parse_args()

    if args.no_trunc or args.trunc_chars is not None:
        global TRUNC
        TRUNC = None if args.no_trunc else args.trunc_chars

    if args.bakeoff:
        run_bakeoff(args)
        return

    if args.gen_subset or not SUBSET_PATH.exists():
        sys.stderr.write("building subset...\n")
        save_subset(gen_subset())
    subset = load_subset()
    sys.stderr.write(f"subset: {len(subset['docs'])} docs, {len(subset['queries'])} queries\n")

    if not args.sweep:
        base_url = BASELINE_URL
        sys.stderr.write(f"baseline evaluate at {base_url}...\n")
        baseline = run_model(subset, base_url)
        print(json.dumps({"subset": {"docs": len(subset["docs"]), "queries": len(subset["queries"])},
                          "baseline": baseline}, indent=2))
        OUT_PATH.write_text(json.dumps(
            {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
             "subset": {"docs": len(subset["docs"]), "queries": len(subset["queries"])},
             "baseline": baseline}, indent=2), encoding="utf-8")
        return

    jina = next(c for c in CANDIDATES if c["name"] == JINA_CURRENT)
    sys.stderr.write(f"baseline (spawned jina, pooling last) on :{PORT}...\n")
    baseline = evaluate_candidate(jina, subset, pooling="last", ctx=args.ctx)
    if not isinstance(baseline, dict) or "ok" in baseline:
        sys.stderr.write(f"baseline spawn failed: {baseline}\n")
        return
    results = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "subset": {"docs": len(subset["docs"]), "queries": len(subset["queries"])},
               "baseline": baseline,
               "candidates": {}}
    only_list = [o.strip().lower() for o in (args.only or "").split(",") if o.strip()]
    for cand in CANDIDATES:
        if cand["name"] == JINA_CURRENT:
            continue
        if only_list and not any(o in cand["name"].lower() for o in only_list):
            continue
        qp = cand.get("query_prefix")
        dp = cand.get("doc_prefix")
        if args.mxbai_prefix and "mxbai" in cand["name"]:
            qp = qp or "search_query: "
            dp = dp or "search_document: "
        m = evaluate_candidate(cand, subset, pooling=cand.get("pooling", "mean"),
                               ctx=args.ctx, doc_prefix=dp, query_prefix=qp)
        if not isinstance(m, dict) or "ok" in m:
            sys.stderr.write(f"  {cand['name']}: FAILED {m}\n")
            results["candidates"][cand["name"]] = m
            continue
        label = cand["name"] + (" (prefix)" if args.mxbai_prefix else "")
        results["candidates"][label] = with_deltas(m, baseline)
        sys.stderr.write(f"  {label}: {m}\n")
    print(json.dumps(results, indent=2))
    OUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()