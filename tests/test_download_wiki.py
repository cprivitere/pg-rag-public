"""Tests pgrag.loaders.download_wiki sync behavior.

Covers V43 batch splitting (≤50 titles per API call), skip/base delay use,
redirect and negative pageid treated as missing, fetch/category failure
aborts, V45/V48 timestamp completeness with tombstones, V51 stable
deterministic filenames, atomic metadata saves, stale/orphan purge, and
skip-vs-download page logic.
"""

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

from pgrag.loaders import download_wiki
from pgrag.loaders.download_wiki import (
    api_call_with_retry,
    enumerate_category_pages,
    fetch_page_content_batch,
    fetch_timestamps,
    get_stable_filename,
    load_metadata,
    save_metadata,
    write_page_content,
    remove_stale_files,
    remove_orphan_files,
    BASE_DELAY,
    MAX_RETRIES,
)


# --- V43: batching ---


def test_v43_batch_splits_titles_into_groups_of_50():
    """fetch_page_content_batch must not send >50 titles per API call."""
    session = MagicMock()
    titles = [f"Page {i}" for i in range(120)]

    call_titles = []

    def fake_call(s, params):
        batch_str = params.get("titles", "")
        count = len(batch_str.split("|")) if batch_str else 0
        call_titles.append(count)
        pages = {}
        for t in params.get("titles", "").split("|"):
            if t:
                pages[str(hash(t))] = {
                    "pageid": abs(hash(t)) % 100000,
                    "title": t,
                    "revisions": [{"slots": {"main": {"*": f"content of {t}"}}}],
                }
        return {"query": {"pages": pages}}

    session.get.side_effect = lambda url, **kw: MagicMock(
        json=lambda: fake_call(session, kw.get("params", {})),
        raise_for_status=lambda: None,
        headers={},
        status_code=200,
    )

    with patch("pgrag.loaders.download_wiki.api_call_with_retry", side_effect=lambda s, p: fake_call(s, p)):
        result = fetch_page_content_batch(session, titles)

    assert len(result) == 120
    assert max(call_titles) <= 50, f"batch size {max(call_titles)} exceeds 50"


def test_v43_batch_of_one_sends_one():
    """Single title → single API call."""
    session = MagicMock()

    def fake_call(s, params):
        titles_str = params.get("titles", "")
        pages = {}
        for t in titles_str.split("|"):
            if t:
                pages[str(hash(t))] = {
                    "pageid": 1,
                    "title": t,
                    "revisions": [{"slots": {"main": {"*": "content"}}}],
                }
        return {"query": {"pages": pages}}

    with patch("pgrag.loaders.download_wiki.api_call_with_retry", side_effect=lambda s, p: fake_call(s, p)):
        result = fetch_page_content_batch(session, ["Single Page"])

    assert "Single Page" in result
    assert result["Single Page"] == ("content", False)


# --- V44: skip delay ---


def test_v44_skip_delay_uses_base_delay(monkeypatch, tmp_path):
    """Skipped pages must use BASE_DELAY, not shorter."""
    delays = []
    original_sleep = time.sleep
    monkeypatch.setattr(time, "sleep", lambda d: delays.append(d))

    session = MagicMock()
    session.get.return_value = MagicMock(
        json=lambda: {"query": {"pages": {}}},
        raise_for_status=lambda: None,
        headers={},
    )

    meta = {
        "pages": {
            "Existing Page": {
                "touched": "2026-01-01T00:00:00Z",
                "filename": "Existing_Page_abc12345.txt",
            }
        }
    }

    test_file = tmp_path / "Existing_Page_abc12345.txt"
    test_file.write_text("old content", encoding="utf-8")

    try:
        with patch("pgrag.loaders.download_wiki.api_call_with_retry") as mock_api:
            mock_api.return_value = {
                "query": {
                    "pages": {
                        "1": {
                            "pageid": 1,
                            "title": "Existing Page",
                            "touched": "2026-01-01T00:00:00Z",
                        }
                    }
                }
            }
            with patch("pgrag.loaders.download_wiki.load_metadata", return_value=meta):
                with patch.object(download_wiki, "WIKI_DIR", tmp_path):
                    with patch.object(download_wiki, "META_FILE", tmp_path / ".meta.json"):
                        download_wiki.main()

        skip_delays = [d for d in delays if d >= 0.1]
        assert all(d >= BASE_DELAY for d in skip_delays if d < 1.0), (
            f"skip delay too short: {[d for d in skip_delays if d < BASE_DELAY]}"
        )
    finally:
        if test_file.exists():
            test_file.unlink()


# --- V49: redirect handling ---


def test_v49_redirect_pageid_negative_treated_as_missing():
    """pageid < 0 (redirect) → treated as missing, no content written."""
    session = MagicMock()

    def fake_call(s, params):
        pages = {}
        for t in params.get("titles", "").split("|"):
            if t == "Redirect Page":
                pages["-1"] = {"pageid": -1, "title": "Redirect Page"}
            elif t:
                pages[str(hash(t))] = {
                    "pageid": abs(hash(t)) % 100000,
                    "title": t,
                    "revisions": [{"slots": {"main": {"*": "content"}}}],
                }
        return {"query": {"pages": pages}}

    with patch("pgrag.loaders.download_wiki.api_call_with_retry", side_effect=lambda s, p: fake_call(s, p)):
        result = fetch_page_content_batch(session, ["Redirect Page", "Normal Page"])

    assert result["Redirect Page"] == ("", True), "redirect should be treated as missing"
    assert result["Normal Page"] == ("content", False)


# --- V50: content fetch failure ---


def test_v50_category_failure_aborts(monkeypatch, tmp_path):
    """Category enumeration exception → sync aborts with exit code 1."""
    session = MagicMock()
    meta = {"pages": {}}

    with patch("pgrag.loaders.download_wiki.api_call_with_retry") as mock_api:
        mock_api.side_effect = RuntimeError("Connection lost")
        with patch("pgrag.loaders.download_wiki.load_metadata", return_value=meta):
            with patch.object(download_wiki, "META_FILE", tmp_path / ".meta.json"):
                with patch.object(download_wiki, "WIKI_DIR", tmp_path):
                    result = download_wiki.main()

    assert result == 1, "should abort on category enumeration failure"


def test_v50_content_fetch_failure_aborts(monkeypatch, tmp_path):
    """Content fetch exception → sync aborts with exit code 1."""
    session = MagicMock()
    meta = {"pages": {}}

    with (
        patch("pgrag.loaders.download_wiki.api_call_with_retry", side_effect=RuntimeError("Connection lost")),
        patch("pgrag.loaders.download_wiki.load_metadata", return_value=meta),
        patch("pgrag.loaders.download_wiki.enumerate_category_pages", return_value=["New Page"]),
        patch("pgrag.loaders.download_wiki.enumerate_category_pages_recursive", return_value=[]),
        patch("pgrag.loaders.download_wiki.fetch_timestamps", return_value={"New Page": "2026-01-01T00:00:00Z"}),
        patch.object(download_wiki, "META_FILE", tmp_path / ".meta.json"),
        patch.object(download_wiki, "WIKI_DIR", tmp_path),
    ):
        result = download_wiki.main()

    assert result == 1, "should abort on content fetch failure"


# --- V45: timestamp completeness ---


def test_v45_absent_title_aborts(tmp_path):
    """Title absent from timestamp response (truncation) → abort before content fetch."""
    session = MagicMock()

    with patch("pgrag.loaders.download_wiki.api_call_with_retry") as mock_api:
        mock_api.return_value = {
            "query": {
                "pages": {
                    "1": {"pageid": 1, "title": "Page A", "touched": "2026-01-01T00:00:00Z"},
                }
            }
        }
        with patch("pgrag.loaders.download_wiki.enumerate_category_pages", return_value=["Page A", "Page B"]):
            with patch("pgrag.loaders.download_wiki.fetch_timestamps", return_value={"Page A": "2026-01-01T00:00:00Z"}):
                with patch.object(download_wiki, "META_FILE", tmp_path / ".meta.json"):
                    with patch.object(download_wiki, "WIKI_DIR", tmp_path):
                        result = download_wiki.main()

    assert result == 1, "should abort when timestamp response is incomplete"


def test_v45_missing_title_deletes_tombstones(tmp_path):
    """Explicit `missing` timestamp → delete local .txt, tombstone meta, no abort."""
    stale_file = tmp_path / "Gone_Page_00000000.txt"
    stale_file.write_text("old", encoding="utf-8")
    meta = {"pages": {"Gone Page": {"touched": "2026-01-01T00:00:00Z", "filename": "Gone_Page_00000000.txt"}}}

    try:
        with (
            patch("pgrag.loaders.download_wiki.enumerate_category_pages", return_value=["Gone Page", "Alive Page"]),
            patch("pgrag.loaders.download_wiki.enumerate_category_pages_recursive", return_value=[]),
            patch("pgrag.loaders.download_wiki.fetch_timestamps", return_value={"Gone Page": None, "Alive Page": "2026-01-01T00:00:00Z"}),
            patch("pgrag.loaders.download_wiki.fetch_page_content_batch", return_value={
                "Gone Page": ("", True),
                "Alive Page": ("content", False),
            }),
            patch.object(download_wiki, "WIKI_DIR", tmp_path),
            patch.object(download_wiki, "META_FILE", tmp_path / ".meta.json"),
        ):
            with patch("pgrag.loaders.download_wiki.load_metadata", return_value=meta):
                result = download_wiki.main()

        assert result == 0, "should not abort on explicit missing title"
        assert not stale_file.exists(), "stale .txt for missing page should be deleted"
    finally:
        if stale_file.exists():
            stale_file.unlink()


# --- V48: inventory failure ---


def test_v48_category_failure_aborts(tmp_path):
    """Category enumeration failure → abort before timestamps."""
    session = MagicMock()

    with patch("pgrag.loaders.download_wiki.enumerate_category_pages", side_effect=RuntimeError("API down")), \
         patch.object(download_wiki, "WIKI_DIR", tmp_path), \
         patch.object(download_wiki, "META_FILE", tmp_path / ".meta.json"):
        result = download_wiki.main()

    assert result == 1, "should abort on category failure"


# --- V51: stable filename ---


def test_v51_stable_filename_deterministic():
    """Same title → same filename across calls."""
    f1 = get_stable_filename("Test Page")
    f2 = get_stable_filename("Test Page")
    assert f1 == f2


def test_v51_stable_filename_no_index_dependency():
    """Filename derived from title only, not enumeration order."""
    titles = ["Zebra", "Alpha", "Middle"]
    filenames = [get_stable_filename(t) for t in titles]
    filenames_shuffled = [get_stable_filename(t) for t in reversed(titles)]
    assert set(filenames) == set(filenames_shuffled)


# --- V53: collision detection ---


def test_v53_collision_different_titles_different_filenames():
    """Different titles → different filenames (via SHA256 digest)."""
    f1 = get_stable_filename("Foo (Bar)")
    f2 = get_stable_filename("Foo [Bar]")
    assert f1 != f2


# --- V54: atomic metadata ---


def test_v54_save_metadata_atomic(tmp_path):
    """Metadata saved via temp + os.replace."""
    meta_file = tmp_path / ".meta.json"
    meta = {"pages": {"Test": {"touched": "2026-01-01", "filename": "test.txt"}}}

    with patch.object(download_wiki, "META_FILE", meta_file):
        save_metadata(meta)

    assert meta_file.exists()
    loaded = json.loads(meta_file.read_text(encoding="utf-8"))
    assert loaded == meta


def test_v54_load_metadata_missing_file():
    """Missing .meta.json → empty dict."""
    with patch.object(download_wiki, "META_FILE", Path("/nonexistent/.meta.json")):
        result = load_metadata()
    assert result == {}


def test_v54_load_metadata_corrupt_file(tmp_path):
    """Corrupt .meta.json → empty dict (graceful fallback)."""
    meta_file = tmp_path / ".meta.json"
    meta_file.write_text("not json {{{", encoding="utf-8")

    with patch.object(download_wiki, "META_FILE", meta_file):
        result = load_metadata()
    assert result == {}


# --- V46: stale purge ---


def test_v46_stale_purge_removes_old_files(tmp_path):
    """Files for removed titles get deleted."""
    old_file = tmp_path / "old_page_abc12345.txt"
    old_file.write_text("stale", encoding="utf-8")
    meta = {"pages": {"Old Page": {"touched": "2026-01-01", "filename": "old_page_abc12345.txt"}}}

    with patch.object(download_wiki, "WIKI_DIR", tmp_path):
        remove_stale_files(meta, set())

    assert not old_file.exists()


def test_v46_stale_purge_keeps_current_files(tmp_path):
    """Files for current titles are NOT deleted."""
    current_file = tmp_path / "keep_page_abc12345.txt"
    current_file.write_text("keep", encoding="utf-8")
    meta = {"pages": {"Keep Page": {"touched": "2026-01-01", "filename": "keep_page_abc12345.txt"}}}

    with patch.object(download_wiki, "WIKI_DIR", tmp_path):
        remove_stale_files(meta, {"Keep Page"})

    assert current_file.exists()


def test_v55_orphan_files_removed(tmp_path):
    """.txt files not tracked by meta are purged; tracked ones survive."""
    meta = {"pages": {"Keep Page": {"touched": "2026-01-01", "filename": "keep_page_abc12345.txt"}}}
    (tmp_path / "keep_page_abc12345.txt").write_text("keep", encoding="utf-8")
    (tmp_path / "orphan_page_ffffffff.txt").write_text("orphan", encoding="utf-8")

    with patch.object(download_wiki, "WIKI_DIR", tmp_path):
        removed = remove_orphan_files(meta)

    assert removed == 1
    assert (tmp_path / "keep_page_abc12345.txt").exists()
    assert not (tmp_path / "orphan_page_ffffffff.txt").exists()


def test_v55_orphan_files_nothing_to_do(tmp_path):
    """All .txt files tracked in meta -> nothing removed."""
    meta = {"pages": {"Keep Page": {"touched": "2026-01-01", "filename": "keep_page_abc12345.txt"}}}
    (tmp_path / "keep_page_abc12345.txt").write_text("keep", encoding="utf-8")

    with patch.object(download_wiki, "WIKI_DIR", tmp_path):
        removed = remove_orphan_files(meta)

    assert removed == 0
    assert (tmp_path / "keep_page_abc12345.txt").exists()


# --- write_page_content ---


def test_write_page_content_atomic(tmp_path):
    """Content written via temp + os.replace."""
    with patch.object(download_wiki, "WIKI_DIR", tmp_path):
        result = write_page_content("Test", "content here", "test_file.txt")

    assert result is True
    assert (tmp_path / "test_file.txt").read_text(encoding="utf-8") == "content here"


def test_write_page_content_failure(tmp_path):
    """Write failure → returns False."""
    with patch.object(download_wiki, "WIKI_DIR", Path("/nonexistent/dir")):
        result = write_page_content("Test", "content", "test.txt")

    assert result is False


# --- skip logic: new categories ---


def test_skip_existing_pages_download_new_pages(tmp_path):
    """Pages in metadata with matching timestamps are skipped; new pages are downloaded."""
    meta = {
        "pages": {
            "Existing Page": {
                "touched": "2026-01-01T00:00:00Z",
                "filename": "Existing_Page_abc12345.txt",
            }
        }
    }

    existing_file = tmp_path / "Existing_Page_abc12345.txt"
    existing_file.write_text("old content", encoding="utf-8")

    def fake_api(s, params):
        action = params.get("action", "")
        if params.get("list") == "categorymembers":
            return {
                "query": {
                    "categorymembers": [
                        {"pageid": 1, "title": "Existing Page"},
                        {"pageid": 2, "title": "New Page"},
                    ]
                }
            }
        if "prop" in params and params["prop"] == "info":
            return {
                "query": {
                    "pages": {
                        "1": {
                            "pageid": 1,
                            "title": "Existing Page",
                            "touched": "2026-01-01T00:00:00Z",
                        },
                        "2": {
                            "pageid": 2,
                            "title": "New Page",
                            "touched": "2026-01-01T00:00:00Z",
                        },
                    }
                }
            }
        if "prop" in params and params["prop"] == "revisions":
            pages = {}
            for t in params.get("titles", "").split("|"):
                if t:
                    pages[str(hash(t))] = {
                        "pageid": abs(hash(t)) % 100000,
                        "title": t,
                        "revisions": [{"slots": {"main": {"*": f"content of {t}"}}}],
                    }
            return {"query": {"pages": pages}}
        return {"query": {"pages": {}}}

    with patch.object(download_wiki, "WIKI_DIR", tmp_path):
        with patch("pgrag.loaders.download_wiki.api_call_with_retry", side_effect=fake_api):
            with patch("pgrag.loaders.download_wiki.load_metadata", return_value=meta):
                with patch.object(download_wiki, "RECURSIVE_CATEGORIES", {}):
                    with patch.object(download_wiki, "META_FILE", tmp_path / ".meta.json"):
                        result = download_wiki.main()

    assert result == 0
    assert not existing_file.exists() or existing_file.read_text(encoding="utf-8") == "old content"
    new_files = list(tmp_path.glob("New_Page_*.txt"))
    assert len(new_files) == 1, f"new page file should exist, found: {new_files}"


def test_stale_metadata_existing_file_still_skipped(tmp_path):
    """File exists on disk but not in metadata with matching timestamp → skip."""
    meta = {"pages": {}}

    expected_filename = get_stable_filename("Orphan Page")
    existing_file = tmp_path / expected_filename
    existing_file.write_text("existing content", encoding="utf-8")

    def fake_api(s, params):
        if params.get("list") == "categorymembers":
            return {
                "query": {
                    "categorymembers": [{"pageid": 1, "title": "Orphan Page"}]
                }
            }
        if "prop" in params and params["prop"] == "info":
            return {
                "query": {
                    "pages": {
                        "1": {
                            "pageid": 1,
                            "title": "Orphan Page",
                            "touched": "2026-01-01T00:00:00Z",
                        }
                    }
                }
            }
        if "prop" in params and params["prop"] == "revisions":
            return {
                "query": {
                    "pages": {
                        "1": {
                            "pageid": 1,
                            "title": "Orphan Page",
                            "revisions": [{"slots": {"main": {"*": "new content"}}}],
                        }
                    }
                }
            }
        return {"query": {"pages": {}}}

    with patch.object(download_wiki, "WIKI_DIR", tmp_path):
        with patch("pgrag.loaders.download_wiki.api_call_with_retry", side_effect=fake_api):
            with patch("pgrag.loaders.download_wiki.load_metadata", return_value=meta):
                with patch.object(download_wiki, "RECURSIVE_CATEGORIES", {}):
                    with patch.object(download_wiki, "META_FILE", tmp_path / ".meta.json"):
                        result = download_wiki.main()

    assert result == 0
    assert existing_file.read_text(encoding="utf-8") == "existing content", (
        "existing file content should not be overwritten when metadata is stale"
    )
