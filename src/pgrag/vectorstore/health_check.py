import json
import sys

import chromadb

from pgrag.config import EMBEDDING_DIM
from pgrag.vectorstore.hashes import embedding_hash, metadata_hash

CHROMA_PATH = "data/chroma"
COLLECTION_NAME = "project_gorgon"
DOCUMENTS_PATH = "data/documents.json"
BATCH_SIZE = 5000


def _iter_collection(collection, include=None, batch_size=BATCH_SIZE):
    """Paginated get — unbounded get() hits SQLite variable limit on large collections."""
    count = collection.count()
    for start in range(0, count, batch_size):
        yield collection.get(
            limit=batch_size,
            offset=start,
            include=include or [],
        )


def _check_hash_integrity(collection, issues, source_docs=None):
    ids = []
    metadatas = []
    for batch in _iter_collection(collection, include=["metadatas"]):
        ids.extend(batch["ids"])
        metadatas.extend(batch["metadatas"])
    if not ids:
        return

    source_by_id = {d["id"]: d for d in source_docs} if source_docs else {}

    bad_embed = 0
    bad_meta = 0
    sample_embed = []
    sample_meta = []

    for doc_id, meta in zip(ids, metadatas, strict=False):
        stored_embed = meta.get("embedding_hash")
        stored_metahash = meta.get("metadata_hash")

        source = source_by_id.get(doc_id)
        if source:
            expected_embed = embedding_hash(source)
            expected_metahash = metadata_hash(source)
        else:
            expected_embed = None
            expected_metahash = None

        if expected_embed is not None and expected_embed != stored_embed:
            if bad_embed < 5:
                sample_embed.append(doc_id)
            bad_embed += 1

        if expected_metahash is not None and expected_metahash != stored_metahash:
            if bad_meta < 5:
                sample_meta.append(doc_id)
            bad_meta += 1

    if bad_embed:
        issues.append(f"Embedding hash mismatch: {bad_embed} doc(s) — sample: {sample_embed}")
    if bad_meta:
        issues.append(f"Metadata hash mismatch: {bad_meta} doc(s) — sample: {sample_meta}")
    if not bad_embed and not bad_meta:
        print(f"Hash integrity: OK ({len(ids)} docs)")


def health_check(
    chroma_path=CHROMA_PATH, collection_name=COLLECTION_NAME, documents_path=DOCUMENTS_PATH
):
    issues = []

    try:
        client = chromadb.PersistentClient(path=chroma_path)
        collection = client.get_collection(collection_name)
    except Exception as e:
        print(f"Cannot connect to ChromaDB at {chroma_path}: {e}")
        return 1

    chroma_count = collection.count()
    print(f"ChromaDB document count: {chroma_count}")

    result = collection.get(limit=1, include=["embeddings"])
    if result["ids"] and len(result["embeddings"]) > 0:
        emb = result["embeddings"][0]
        if emb is not None and len(emb) > 0:
            dim = len(emb)
            print(f"Embedding dimension: {dim}")
            if dim != EMBEDDING_DIM:
                issues.append(
                    f"Embedding dimension mismatch: collection={dim}, EMBEDDING_DIM={EMBEDDING_DIM}"
                )
        else:
            issues.append("No embedding data in collection")
    else:
        print("Collection is empty — skipping dimension check")

    docs = None
    try:
        with open(documents_path, encoding="utf-8") as f:
            docs = json.load(f)
        expected_count = len(docs)
        print(f"Expected document count (from {documents_path}): {expected_count}")
        if chroma_count != expected_count:
            issues.append(
                f"Document count mismatch: ChromaDB={chroma_count}, expected={expected_count}"
            )

        chroma_ids = set()
        for batch in _iter_collection(collection):
            chroma_ids.update(batch["ids"])
        doc_ids = set(d["id"] for d in docs)
        orphaned = chroma_ids - doc_ids
        if orphaned:
            sample = sorted(orphaned)[:10]
            issues.append(
                f"Orphaned documents in ChromaDB (not in documents.json): "
                f"{len(orphaned)} — {sample}"
            )
        missing = doc_ids - chroma_ids
        if missing:
            sample = sorted(missing)[:10]
            issues.append(
                f"Missing documents in ChromaDB (in documents.json but not indexed): "
                f"{len(missing)} — {sample}"
            )
    except FileNotFoundError:
        issues.append(f"{documents_path} not found — cannot verify expected count")
    except json.JSONDecodeError as e:
        issues.append(f"{documents_path} is not valid JSON: {e}")

    _check_hash_integrity(collection, issues, source_docs=docs)

    if issues:
        print("\nIssues found:")
        for issue in issues:
            print(f"  - {issue}")
        print(f"\nFAIL: {len(issues)} issue(s) found")
        return 1

    print("\nOK: All checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(health_check())
