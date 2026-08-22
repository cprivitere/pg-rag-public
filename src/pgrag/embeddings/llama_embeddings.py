import requests


EMBEDDING_URL = "http://localhost:8081/embedding"

# The production encoder (bge-small-en-v1.5) has a hard 512-token position
# limit; the llama-server embedding endpoint REJECTS any input above it with
# HTTP 400 ("input (N tokens) is larger than the max context size (512
# tokens)"). Live-validated by sweeping -c 512..4096 / -b / -ub: no server
# flag raises the ceiling (unlike mxbai's 4096 ctx). This corpus tokenizes at
# ~4.9 chars/token, so a 2000-char cap stays well under 512 tokens; the
# oversize-tail fallback re-clips smaller and retries in case some future
# input tokenizes much denser.
MAX_EMBED_CHARS = 2000
_TRUNC_STEPS = (MAX_EMBED_CHARS, 1000, 512, 256)


def _clip(texts, budget):
    return [t[:budget] for t in texts]


def embed_text(text):
    return embed_batch([text])[0]


class EmbeddingServerError(ConnectionError):
    pass


def embed_batch(texts):
    data = None
    for budget in _TRUNC_STEPS:
        try:
            response = requests.post(
                EMBEDDING_URL,
                json={"content": _clip(texts, budget)},
                timeout=300
            )
            if response.status_code == 400 and "larger than the max context size" \
                    in response.text:
                continue  # still over the 512-token ceiling after clipping
            response.raise_for_status()
            data = response.json()
            break
        except requests.exceptions.ConnectionError as e:
            raise EmbeddingServerError(
                f"Cannot connect to embedding server at {EMBEDDING_URL}. "
                "Ensure llama.cpp is running on port 8081."
            ) from e
        except requests.exceptions.Timeout as e:
            raise EmbeddingServerError(
                f"Embedding server at {EMBEDDING_URL} timed out after 300s."
            ) from e

    if data is None:
        raise EmbeddingServerError(
            f"Embedding server at {EMBEDDING_URL} rejected even 256-char inputs "
            "(unexpected: 512-token ceiling should always be satisfiable)."
        )

    vectors = [item["embedding"][0] for item in data]

    return validate_embeddings(vectors)


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