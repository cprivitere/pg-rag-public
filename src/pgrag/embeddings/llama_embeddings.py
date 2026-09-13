import requests

EMBEDDING_URL = "http://localhost:8081/embedding"

# The production encoder (bge-small-en-v1.5) has a hard 512-token position
# limit; llama-server REJECTS any input above it with HTTP 400 ("input (N
# tokens) is larger than the max context size (512 tokens)"). Live-validated
# by sweeping -c 512..4096 / -b / -ub: no server flag raises the ceiling
# (unlike mxbai's 4096 ctx). This corpus tokenizes at ~4.9 chars/token, so a
# 2000-char cap stays well under 512 tokens; a per-text shrink fallback covers
# the rare denser-tokenized input without over-truncating siblings.
MAX_EMBED_CHARS = 2000
# The encoder's hard position cap (bge-small-en-v1.5). A chunking regression
# or non-chunked doc should fail fast at the build guard rather than silently
# taking the shrink fallback, so this is exposed for callers to assert on.
MAX_EMBED_TOKENS = 512
_TRUNC_STEPS = (MAX_EMBED_CHARS, 1000, 512, 256)

# bge-small-en-v1.5 is instruction-tuned and asymmetric: the BGE v1.5 card
# recommends this query prompt for short-query retrieval, and the bakeoff
# measured it for bge-small (bare 0.9028 -> 0.9875 with it). Production
# documents embed bare (build_index calls embed_batch directly, matching the
# bakeoff's doc_prefix None); only queries get the prompt, via embed_text.
# Keep in sync with BAKEOFF_CANDIDATES["bge-small-*"].query_prefix.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def _clip(texts, budget):
    return [t[:budget] for t in texts]


class _InputTooLong(Exception):
    pass


class EmbeddingServerError(ConnectionError):
    pass


def _post(texts, budget):
    try:
        response = requests.post(EMBEDDING_URL, json={"content": _clip(texts, budget)}, timeout=300)
    except requests.exceptions.ConnectionError as e:
        raise EmbeddingServerError(
            f"Cannot connect to embedding server at {EMBEDDING_URL}. "
            "Ensure llama.cpp is running on port 8081."
        ) from e
    except requests.exceptions.Timeout as e:
        raise EmbeddingServerError(
            f"Embedding server at {EMBEDDING_URL} timed out after 300s."
        ) from e
    if response.status_code == 400 and "larger than the max context size" in response.text:
        raise _InputTooLong
    response.raise_for_status()
    return response.json()


def _embed_one(text):
    # Shrink only this text until it fits, so one dense straggler can't force
    # the rest of a batch below their valid budget.
    for budget in _TRUNC_STEPS:
        try:
            return _post([text], budget)[0]["embedding"][0]
        except _InputTooLong:
            continue
    raise EmbeddingServerError(
        f"Embedding server at {EMBEDDING_URL} rejected even the 256-char "
        "truncation of a request (unexpected: the 512-token ceiling should "
        "always be satisfiable)."
    )


def embed_text(text):
    # Query embed: prepend bge-small's retrieval prompt (see QUERY_PREFIX).
    # Documents never get it — build_index calls embed_batch directly.
    return embed_batch([QUERY_PREFIX + text])[0]


def embed_batch(texts):
    vectors = _embed_batch_recursive(texts, MAX_EMBED_CHARS)
    return validate_embeddings(vectors)


def _embed_batch_recursive(texts, budget):
    """Embed a batch, isolating the texts that exceed the window.

    Bisection: on _InputTooLong, split the batch and retry each half. A
    single-text batch that still 400s falls through to _embed_one (per-text
    shrink). This bounds the API request count to O(k log n) where k is the
    number of genuinely over-window texts — not O(n) serial round-trips for
    the whole batch — and keeps in-window siblings at their full budget.
    """
    if len(texts) == 1:
        return [_embed_one(texts[0])]
    try:
        data = _post(texts, budget)
        return [item["embedding"][0] for item in data]
    except _InputTooLong:
        mid = len(texts) // 2
        return _embed_batch_recursive(texts[:mid], budget) + _embed_batch_recursive(
            texts[mid:], budget
        )


class EmbeddingValidationError(ValueError):
    pass


def validate_embeddings(vectors, expected_dim=None):
    if not vectors:
        raise EmbeddingValidationError("Empty embedding response")

    for i, vec in enumerate(vectors):
        if not isinstance(vec, list):
            raise EmbeddingValidationError(f"Vector {i}: expected list, got {type(vec).__name__}")
        if not vec:
            raise EmbeddingValidationError(f"Vector {i} is empty")
        for j, v in enumerate(vec):
            if not isinstance(v, (int, float)):
                raise EmbeddingValidationError(
                    f"Vector {i}[{j}]: expected float, got {type(v).__name__}"
                )

    if expected_dim is not None:
        for i, vec in enumerate(vectors):
            if len(vec) != expected_dim:
                raise EmbeddingValidationError(
                    f"Vector {i} length {len(vec)} != expected {expected_dim}"
                )

    return vectors
