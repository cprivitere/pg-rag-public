"""Tests pgrag.validation.validate_all: a healthy repo exits 0, while
document shape violations and index-layer failures are hard (1). Upstream
transient states — missing/empty sources, stale DOCUMENTS_VERSION, and
wiki-meta drift — are warnings (0) with a remediation pointer, never a gate.
Layer checks are isolated by patching health_check; one integration test
builds a real index."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from pgrag.config import DOCUMENTS_VERSION, EMBEDDING_DIM
from pgrag.validation import validate_all
from pgrag.vectorstore.build_index import build_index

META = {"source": "cdn", "table": "items", "type": "item"}


def _write_documents(doc_path, docs):
    with open(doc_path, "w", encoding="utf-8") as f:
        json.dump(docs, f)


def _write_version(version_file, version=DOCUMENTS_VERSION):
    with open(version_file, "w", encoding="utf-8") as f:
        json.dump({"version": version, "updated": 0}, f)


def _fixture(tmp):
    """Healthy offline fixture: sources, documents, version file. Index layer
    is the caller's responsibility (patched or built with _build_test_index)."""
    cdn_dir = Path(tmp) / "cdn"
    wiki_dir = Path(tmp) / "wiki"
    cdn_dir.mkdir()
    wiki_dir.mkdir()
    (cdn_dir / "items.json").write_text(json.dumps([{"Id": 1, "Name": "alpha"}]), encoding="utf-8")
    (wiki_dir / "AlphaPage_abcdef12.txt").write_text("# Alpha\n", encoding="utf-8")
    (wiki_dir / ".meta.json").write_text(
        json.dumps({"pages": {"AlphaPage": {"filename": "AlphaPage_abcdef12.txt"}}}),
        encoding="utf-8",
    )
    doc_path = Path(tmp) / "documents.json"
    version_path = Path(tmp) / "documents_version.json"
    _write_documents(
        doc_path,
        [
            {"id": "a", "type": "item", "text": "alpha", "metadata": dict(META)},
            {"id": "b", "type": "item", "text": "beta", "metadata": dict(META)},
        ],
    )
    _write_version(version_path)
    return cdn_dir, wiki_dir, doc_path, version_path


def fake_embed_batch(texts):
    return [[0.1] * EMBEDDING_DIM for _ in texts]


def _build_test_index(docs, chroma_path):
    with patch("pgrag.vectorstore.build_index.embed_batch", side_effect=fake_embed_batch):
        build_index(documents=docs, chroma_path=chroma_path)


def _healthy_args(tmp):
    cdn_dir, wiki_dir, doc_path, version_path = _fixture(tmp)
    return dict(
        cdn_dir=str(cdn_dir),
        wiki_dir=str(wiki_dir),
        documents_path=str(doc_path),
        version_file=str(version_path),
    )


def test_healthy_repo_exits_0():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        args = _healthy_args(tmp)
        docs = json.loads(Path(args["documents_path"]).read_text(encoding="utf-8"))
        _build_test_index(list(docs), str(Path(tmp) / "chroma"))
        assert validate_all(chroma_path=str(Path(tmp) / "chroma"), **args) == 0


def test_missing_sources_warns_not_fails():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        cdn_dir, wiki_dir, doc_path, version_path = _fixture(tmp)
        # Drain the source dirs where _check_sources expects JSON / txt.
        for p in Path(cdn_dir).glob("*.json"):
            p.unlink()
        for p in Path(wiki_dir).glob("*.txt"):
            p.unlink()
        (Path(cdn_dir) / "items.json").write_text("[]", encoding="utf-8")
        with patch("pgrag.validation.health_check", return_value=0):
            assert (
                validate_all(
                    cdn_dir=str(cdn_dir),
                    wiki_dir=str(wiki_dir),
                    documents_path=str(doc_path),
                    version_file=str(version_path),
                )
                == 0
            )


def test_stale_documents_version_warns_not_fails(capsys):
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        cdn_dir, wiki_dir, doc_path, version_path = _fixture(tmp)
        _write_version(version_path, version=DOCUMENTS_VERSION + 1)
        with patch("pgrag.validation.health_check", return_value=0):
            rc = validate_all(
                cdn_dir=str(cdn_dir),
                wiki_dir=str(wiki_dir),
                documents_path=str(doc_path),
                version_file=str(version_path),
            )
            assert rc == 0
            assert "sync" in capsys.readouterr().out


def test_document_schema_violations_reported():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        cdn_dir, wiki_dir, doc_path, version_path = _fixture(tmp)
        _write_documents(
            doc_path,
            [
                {"id": "dup", "type": "item", "text": "x", "metadata": dict(META)},
                {"id": "dup", "type": "item", "text": "y", "metadata": dict(META)},
                {"id": "bad", "type": "item", "text": "", "metadata": dict(META)},
                {
                    "id": "src",
                    "type": "item",
                    "text": "z",
                    "metadata": {"source": "nope", "table": "items"},
                },
                {
                    "id": "tbl",
                    "type": "item",
                    "text": "w",
                    "metadata": {"source": "cdn", "table": ""},
                },
            ],
        )
        with patch("pgrag.validation.health_check", return_value=0):
            assert (
                validate_all(
                    cdn_dir=str(cdn_dir),
                    wiki_dir=str(wiki_dir),
                    documents_path=str(doc_path),
                    version_file=str(version_path),
                )
                == 1
            )


def test_wiki_meta_drift_warns_not_fails():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        cdn_dir, wiki_dir, doc_path, version_path = _fixture(tmp)
        # Orphan file not tracked by meta, and a tracked file missing on disk.
        (Path(wiki_dir) / "Orphan_11111111.txt").write_text("x", encoding="utf-8")
        (Path(wiki_dir) / "AlphaPage_abcdef12.txt").unlink()
        with patch("pgrag.validation.health_check", return_value=0):
            assert (
                validate_all(
                    cdn_dir=str(cdn_dir),
                    wiki_dir=str(wiki_dir),
                    documents_path=str(doc_path),
                    version_file=str(version_path),
                )
                == 0
            )


def test_index_layer_failure_propagates():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        args = _healthy_args(tmp)
        # health_check against an empty/nonexistent chroma dir fails.
        assert validate_all(chroma_path=str(Path(tmp) / "empty_chroma"), **args) == 1
