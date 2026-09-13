import contextlib
import json
import logging
import os
import tempfile
import time

import requests

from pgrag.config import DATA_DIR

logger = logging.getLogger(__name__)

RERANK_URL = "http://127.0.0.1:8082/v1/rerank"
RERANK_TIMEOUT = 60
STATS_FILE = DATA_DIR / "rerank_stats.json"

# Defense-in-depth: keep every rerank payload well under the server's batch
# ceiling so llama.cpp never 500s with "input is too large to process".
# bge-reranker-v2-m3 also saturates around this length anyway.
MAX_RERANK_DOC_CHARS = 12000
MAX_RERANK_DOCS = 100
MAX_RERANK_QUERY_CHARS = 2000


class RerankError(ConnectionError):
    pass


def rerank_documents(query, documents, top_n):
    """Cross-encoder rerank via llama.cpp. Returns original indices, best first.

    Raises RerankError on server/parse failure — caller falls back to lexical
    order. Never returns fewer than the requested top_n unless docs input
    is shorter.
    """
    if not documents:
        return []
    documents = [d[:MAX_RERANK_DOC_CHARS] for d in documents][:MAX_RERANK_DOCS]
    query = query[:MAX_RERANK_QUERY_CHARS]
    try:
        response = requests.post(
            RERANK_URL,
            json={
                "query": query,
                "documents": documents,
                "top_n": min(top_n, len(documents)),
            },
            timeout=RERANK_TIMEOUT,
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        raise RerankError(f"reranker unavailable at {RERANK_URL}: {exc}") from exc

    try:
        results = response.json().get("results", [])
        ordered = sorted(results, key=lambda r: r["relevance_score"], reverse=True)
        indices = [int(r["index"]) for r in ordered]
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise RerankError(f"reranker bad response shape: {exc!r}") from exc

    if not indices or max(indices) >= len(documents):
        raise RerankError(f"reranker out-of-range index: {indices!r}")

    return indices


def load_stats():
    try:
        return json.loads(STATS_FILE.read_text(encoding="utf-8"))
    except OSError, ValueError:
        return {"failures": 0, "last_failure": None, "last_success": None}


def record_failure():
    stats = load_stats()
    stats["failures"] = stats.get("failures", 0) + 1
    stats["last_failure"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    _write_stats(stats)


def record_success():
    stats = load_stats()
    stats["last_success"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    _write_stats(stats)


def _write_stats(stats):
    """Best-effort atomic write (V64) — failure must not break retrieval."""
    try:
        STATS_FILE.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=STATS_FILE.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(stats, fh)
            os.replace(tmp, STATS_FILE)
        finally:
            if os.path.exists(tmp):
                with contextlib.suppress(OSError):
                    os.remove(tmp)
    except OSError as exc:
        logger.warning("rerank stats write failed: %s", exc)
