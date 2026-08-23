from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"

CDN_DIR = DATA_DIR / "cdn"

WIKI_DIR = DATA_DIR / "wiki"

# Derived caches live outside the source dirs (data/cdn, data/wiki) so the
# raw download stays a read-only cache owned only by the download/verify
# scripts.
DERIVED_DIR = DATA_DIR / "derived"
WIKI_PARSED_CACHE = DERIVED_DIR / "wiki_parsed.json"
CURATED_DIR = WIKI_DIR / "curated"

# Document-shape version. generate_documents() stamps
# DERIVED/documents_version.json with this; build-index refuses to embed a
# documents.json whose stored version differs (otherwise it would silently
# serve stale docs — the classic build-index-vs-build-documents trap).
# Bump whenever document generation changes shape (new metadata keys, table
# records, chunking) so a stale persist is surfaced, not re-embedded.
DOCUMENTS_VERSION = 6

DOCUMENTS_VERSION_FILE = DERIVED_DIR / "documents_version.json"

EMBEDDING_DIM = 384

CONTEXT_BUDGET = 34000