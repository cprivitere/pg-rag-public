import json
import math
import os
import pickle
import re
from collections import Counter

CACHE_VERSION = 1
DEFAULT_INDEX_PATH = "data/documents.json"
DEFAULT_PICKLE_PATH = "data/bm25_index.pkl"


class BM25:
    def __init__(self, k1=1.2, b=0.75):
        self.k1 = k1
        self.b = b
        self.doc_count = 0
        self.avgdl = 0.0
        self.doc_lengths = []
        self.documents = []
        self.postings = {}
        self.df = Counter()

    def index(self, documents):
        self.documents = list(documents)
        self.doc_count = len(self.documents)
        total_length = 0
        self.doc_lengths = []
        self.postings = {}
        self.df = Counter()

        for doc_idx, doc in enumerate(self.documents):
            tokens = self._tokenize(doc)
            seen = set()
            for t in tokens:
                if t not in seen:
                    seen.add(t)
                    self.postings.setdefault(t, {})[doc_idx] = 1
                    self.df[t] += 1
                else:
                    self.postings[t][doc_idx] += 1
            self.doc_lengths.append(len(tokens))
            total_length += len(tokens)

        self.avgdl = total_length / self.doc_count if self.doc_count else 0.0

    def _tokenize(self, text):
        return re.findall(r"\w+", text.lower())

    def _idf(self, term):
        n = len(self.postings.get(term, {}))
        return math.log((self.doc_count - n + 0.5) / (n + 0.5) + 1.0)

    def search(self, query, k=10):
        query_terms = self._tokenize(query)
        if not query_terms or not self.doc_count:
            return [], []

        tf = Counter(query_terms)
        scores = {}
        # Sparse pass over precomputed postings: only docs that contain a
        # query term are scored, so no per-query 98 MB re-tokenization.
        for term in query_terms:
            idf = self._idf(term)
            qf = tf[term]
            for doc_idx, term_freq in self.postings.get(term, {}).items():
                denominator = term_freq + self.k1 * (
                    1 - self.b + self.b * self.doc_lengths[doc_idx] / self.avgdl
                )
                contribution = idf * qf * term_freq * (self.k1 + 1) / denominator
                scores[doc_idx] = scores.get(doc_idx, 0.0) + contribution

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top = ranked[:k]
        return [idx for idx, _ in top], [score for _, score in top]


def _documents_mtime(source_path):
    try:
        return os.path.getmtime(source_path)
    except OSError:
        return None


def save_bm25_index(model, documents, path=DEFAULT_PICKLE_PATH, source_path=DEFAULT_INDEX_PATH):
    """Persist an indexed BM25 + the doc store to `path`, keyed by the
    documents.json mtime so `load_bm25_index` can detect staleness."""
    payload = {
        "__version": CACHE_VERSION,
        "documents_mtime": _documents_mtime(source_path),
        "k1": model.k1,
        "b": model.b,
        "doc_count": model.doc_count,
        "avgdl": model.avgdl,
        "doc_lengths": model.doc_lengths,
        "postings": model.postings,
        "documents": documents,
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def load_bm25_index(path=DEFAULT_INDEX_PATH, pkl_path=DEFAULT_PICKLE_PATH):
    """Return (BM25, docs).

    Serves the cached index from `pkl_path` while `documents.json` is
    unchanged (mtime-keyed); otherwise rebuilds and rewrites the cache.
    """
    mtime = _documents_mtime(path)
    if os.path.exists(pkl_path):
        try:
            with open(pkl_path, "rb") as f:
                state = pickle.load(f)
            if (
                state.get("__version") == CACHE_VERSION
                and state.get("documents_mtime") == mtime
                and state.get("doc_count", 0) > 0
                and isinstance(state.get("documents"), list)
            ):
                model = BM25(k1=state["k1"], b=state["b"])
                model.doc_count = state["doc_count"]
                model.avgdl = state["avgdl"]
                model.doc_lengths = state["doc_lengths"]
                model.postings = state["postings"]
                model.df = Counter({term: len(p) for term, p in state["postings"].items()})
                return model, state["documents"]
        except (
            OSError,
            EOFError,
            pickle.UnpicklingError,
            AttributeError,
            KeyError,
            TypeError,
            ValueError,
        ):
            pass  # stale/corrupt cache -> rebuild below

    with open(path, encoding="utf-8") as f:
        docs = json.load(f)
    model = BM25()
    model.index([doc["text"] for doc in docs])
    try:
        dirname = os.path.dirname(pkl_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        save_bm25_index(model, docs, pkl_path, path)
    except OSError:
        pass  # cache write is best-effort
    return model, docs


if __name__ == "__main__":
    # Pre-warm/build the BM25 index. load_bm25_index() rebuilds in-memory
    # AND persists when documents.json's mtime changed — otherwise the first
    # hybrid retrieve() pays a multi-minute sync rebuild inline. Run after
    # build-documents to avoid that first-query spike.
    import sys

    sys.stdout.reconfigure(line_buffering=True)
    load_bm25_index()
    print("BM25 index ready (rebuilt if documents.json changed)")
