"""Source tree immutability guard unit tests (Phase 6)."""

from tests.conftest import _assert_source_unchanged, _snapshot_source


def test_snapshot_and_assert_unchanged(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.txt").write_text("one", encoding="utf-8")
    (src / "sub").mkdir()
    (src / "sub" / "b.json").write_text("two", encoding="utf-8")

    before = _snapshot_source([src])
    assert before["a.txt"][0] == 3
    assert before["sub/b.json"][0] == 3
    assert _assert_source_unchanged(before, [src]) == []


def test_detects_added_file(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.txt").write_text("one", encoding="utf-8")

    before = _snapshot_source([src])
    (src / "b.txt").write_text("two", encoding="utf-8")

    changed = _assert_source_unchanged(before, [src])
    assert "b.txt" in changed


def test_detects_changed_size(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    p = src / "a.txt"
    p.write_text("one", encoding="utf-8")

    before = _snapshot_source([src])
    p.write_text("one-longer", encoding="utf-8")

    changed = _assert_source_unchanged(before, [src])
    assert "a.txt" in changed


def test_detects_removed_file(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    p = src / "a.txt"
    p.write_text("one", encoding="utf-8")

    before = _snapshot_source([src])
    p.unlink()

    changed = _assert_source_unchanged(before, [src])
    assert "a.txt" in changed


def test_excludes_nonsource_subtree(monkeypatch, tmp_path):
    """The wiki branch (keyed on conftest.WIKI_DIR) snapshots only *.txt +
    .meta.json, so derived artifacts in the source dir are not part of the
    source snapshot."""
    import tests.conftest as conftest

    src = tmp_path / "src"
    src.mkdir()
    (src / "page.txt").write_text("x", encoding="utf-8")
    (src / "derived.json").write_text("y", encoding="utf-8")
    (src / ".meta.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(conftest, "WIKI_DIR", src)
    monkeypatch.setattr(conftest, "WIKI_META", src / ".meta.json")

    before = _snapshot_source([src])
    assert "page.txt" in before
    assert ".meta.json" in before
    assert "derived.json" not in before
    assert _assert_source_unchanged(before, [src]) == []

    # a derived artifact mutation is invisible to the source snapshot
    (src / "derived.json").write_text("z", encoding="utf-8")
    assert _assert_source_unchanged(before, [src]) == []
