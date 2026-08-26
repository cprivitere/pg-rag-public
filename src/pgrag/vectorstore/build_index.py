import json
import sys

import chromadb

from pgrag.config import DOCUMENTS_VERSION, DOCUMENTS_VERSION_FILE, EMBEDDING_DIM
from pgrag.documents.tokenizer import token_count
from pgrag.embeddings.llama_embeddings import (
    MAX_EMBED_CHARS,
    MAX_EMBED_TOKENS,
    embed_batch,
    validate_embeddings,
)
from pgrag.vectorstore.hashes import embedding_hash, metadata_hash

try:
    sys.stdout.reconfigure(line_buffering=True)
except (AttributeError, ValueError):
    pass  # not a real stream (e.g. captured by pytest)

BATCH_SIZE = 5000
EMBED_BATCH_SIZE = 10000


def load_documents():
    """Load the persisted documents.json, refusing a stale generation.

    build-documents stamps DOCUMENTS_VERSION_FILE with the generator version.
    A mismatch (or missing marker) means documents.json predates the current
    document shape and a build-index would silently embed outdated docs.
    Fail loudly instead: run build-documents (or `mise sync-<source>`) first.
    """
    try:
        meta = json.loads(DOCUMENTS_VERSION_FILE.read_text(encoding="utf-8"))
        stored = meta.get("version")
    except (OSError, ValueError):
        stored = None
    if stored != DOCUMENTS_VERSION:
        raise ValueError(
            f"data/documents.json is stale (generator v{stored!r}, expected "
            f"v{DOCUMENTS_VERSION}). Run `uv run pgrag build-documents` "
            "(or `mise generate-docs`, or a `mise sync-*` task) first — "
            "build-index only embeds what documents.json already holds "
            "and cannot regenerate it."
        )
    with open("data/documents.json", "r", encoding="utf-8") as f:
        return json.load(f)


def _get_existing_dim(collection):
    # lightweight — fetches full vector only to check its length
    existing = collection.get(limit=1, include=["embeddings"])
    if len(existing["ids"]) > 0 and len(existing["embeddings"]) > 0:
        emb = existing["embeddings"][0]
        if emb is not None and len(emb) > 0:
            return len(emb)
    return None


def build_index(documents=None, chroma_path="data/chroma", source=None):
    """Incrementally reconcile the Chroma "project_gorgon" collection with
    ``documents`` (default: load_documents(), which refuses a stale
    DOCUMENTS_VERSION). Validates the collection dim == EMBEDDING_DIM,
    deletes removed ids (source-restricted for a partial rebuild when
    ``source`` is given), and embeds only changed docs: metadata-only changes
    go through collection.update (no re-embed), unchanged docs are skipped.
    Batches embedding at EMBED_BATCH_SIZE and upserts/metadata updates at
    BATCH_SIZE."""
    if documents is None:
        documents = load_documents()

    if source is not None:
        scoped = [
            doc for doc in documents
            if doc.get("metadata", {}).get("source") == source
        ]
        if not scoped:
            raise ValueError(
                f"No documents with source='{source}' in documents.json"
            )
        print(
            f"Partial rebuild: source='{source}' ({len(scoped)} of {len(documents)} documents)"
        )
        documents = scoped

    client = chromadb.PersistentClient(
        path=chroma_path
    )

    collection = client.get_or_create_collection(
        name="project_gorgon"
    )

    expected_dim = _get_existing_dim(collection)
    if expected_dim is not None:
        if expected_dim != EMBEDDING_DIM:
            raise ValueError(
                f"Existing collection dimension {expected_dim} != EMBEDDING_DIM {EMBEDDING_DIM}. "
                "Aborting — model config changed."
            )
        print(f"Existing collection dimension: {expected_dim}")
    else:
        print("No existing collection — skipping dimension check")

    existing_hashes = {}
    existing_ids = set()
    existing_sources = {}

    for start in range(0, collection.count(), BATCH_SIZE):
        batch = collection.get(
            limit=BATCH_SIZE,
            offset=start,
            include=["metadatas"]
        )
        for doc_id, metadata in zip(
            batch["ids"],
            batch["metadatas"]
        ):
            existing_ids.add(doc_id)
            if metadata:
                existing_sources[doc_id] = metadata.get("source")
                if "embedding_hash" in metadata:
                    existing_hashes[doc_id] = {
                        "embedding_hash": metadata["embedding_hash"],
                        "metadata_hash": metadata["metadata_hash"],
                    }

    current_ids = {doc["id"] for doc in documents}

    if source is None:
        deleted_ids = existing_ids - current_ids
    else:
        deleted_ids = {
            doc_id for doc_id in existing_ids
            if existing_sources.get(doc_id) == source
            and doc_id not in current_ids
        }

    if deleted_ids:
        print(f"Deleting {len(deleted_ids)} removed documents")

        collection.delete(
            ids=list(deleted_ids)
        )
    else:
        print("No deleted documents found")

    documents_to_embed = []
    metadata_only_updates = []

    for doc in documents:
        if "id" not in doc or "text" not in doc:
            raise ValueError(
                f"Document missing required keys 'id'/'text': {doc.get('id', 'unknown')}"
            )
        doc_id = doc["id"]
        doc_embed_hash = embedding_hash(doc)
        doc_meta_hash = metadata_hash(doc)

        existing = existing_hashes.get(doc_id)

        if existing is not None:
            existing_embed_hash = existing.get("embedding_hash")
            existing_meta_hash = existing.get("metadata_hash")

            if existing_embed_hash == doc_embed_hash and existing_meta_hash == doc_meta_hash:
                continue

            if existing_embed_hash == doc_embed_hash and existing_meta_hash != doc_meta_hash:
                metadata_only_updates.append(doc)
                continue

        documents_to_embed.append(doc)

    print(
        f"Need to embed {len(documents_to_embed)} of {len(documents)} documents"
    )

    if metadata_only_updates:
        print(f"Metadata-only updates: {len(metadata_only_updates)} documents")

        meta_ids = []
        meta_metadatas = []

        for doc in metadata_only_updates:
            metadata = dict(doc["metadata"])
            metadata["type"] = doc["type"]
            metadata["embedding_hash"] = embedding_hash(doc)
            metadata["metadata_hash"] = metadata_hash(doc)

            meta_ids.append(doc["id"])
            meta_metadatas.append(metadata)

        for i in range(0, len(meta_ids), BATCH_SIZE):
            print(
                f"Updating metadata {i} - {min(i + BATCH_SIZE, len(meta_ids))}"
            )

            collection.update(
                ids=meta_ids[i:i + BATCH_SIZE],
                metadatas=meta_metadatas[i:i + BATCH_SIZE]
            )

    if not documents_to_embed:
        print("No documents need embedding.")
    else:
        print(
            f"Embedding {len(documents_to_embed)} documents..."
        )

        # Pre-embed window guard: fail fast on a chunking regression instead
        # of letting an over-window doc fall silently into the (now-bounded)
        # embed_batch fallback. token_count is None when the tokenizer is
        # unavailable (offline) -> char-only check.
        over = [
            doc["id"] for doc in documents_to_embed
            if len(doc["text"]) > MAX_EMBED_CHARS
            or (token_count(doc["text"]) or 0) > MAX_EMBED_TOKENS
        ]
        if over:
            raise ValueError(
                f"{len(over)} document(s) exceed the embed window "
                f"(>{MAX_EMBED_CHARS}c or >{MAX_EMBED_TOKENS} tokens); "
                f"chunking regression? First: {over[:3]}"
            )

        for start in range(0, len(documents_to_embed), EMBED_BATCH_SIZE):

            batch = documents_to_embed[start:start + EMBED_BATCH_SIZE]

            print(
                f"Embedding {start}/{len(documents_to_embed)}"
            )

            batch_embeddings = embed_batch(
                [doc["text"] for doc in batch]
            )

            validate_embeddings(batch_embeddings, expected_dim=EMBEDDING_DIM)

            ids = []
            embeddings = []
            texts = []
            metadatas = []

            for doc, embedding in zip(batch, batch_embeddings):

                ids.append(doc["id"])
                embeddings.append(embedding)
                texts.append(doc["text"])

                metadata = dict(
                    doc["metadata"]
                )

                metadata["type"] = doc["type"]
                metadata["embedding_hash"] = embedding_hash(doc)
                metadata["metadata_hash"] = metadata_hash(doc)

                metadatas.append(metadata)

            for i in range(0, len(ids), BATCH_SIZE):
                print(
                    f"Adding vectors {start + i} - "
                    f"{start + min(i + BATCH_SIZE, len(ids))}"
                )

                collection.upsert(
                    ids=ids[i:i + BATCH_SIZE],
                    embeddings=embeddings[i:i + BATCH_SIZE],
                    documents=texts[i:i + BATCH_SIZE],
                    metadatas=metadatas[i:i + BATCH_SIZE]
                )

    print("Done.")


if __name__ == "__main__":
    build_index()
