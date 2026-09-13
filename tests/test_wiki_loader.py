"""Tests pgrag.loaders.wiki_loader.load_wiki and _HASHED_FILE_RE: real titles
from .meta.json, stripped-stem fallback, skipping legacy unhashed files,
collision suffixes, hex-like titles, unreadable-meta fallback, and the
hashed-filename regex."""

from pgrag.loaders.wiki_loader import _HASHED_FILE_RE, load_wiki


def _load(tmp_path, monkeypatch, files, meta=None):
    if meta is not None:
        (tmp_path / ".meta.json").write_text(meta, encoding="utf-8")
    for name, content in files.items():
        (tmp_path / name).write_text(content, encoding="utf-8")
    monkeypatch.setattr("pgrag.loaders.wiki_loader.WIKI_DIR", tmp_path)
    db = type("DB", (), {})()
    load_wiki(db)
    return db


def test_load_wiki_uses_real_title_from_meta(tmp_path, monkeypatch):
    meta = (
        '{"pages": {'
        '"Mycology": {"filename": "Mycology_91d06ad5.txt", "touched": "t"},'
        '"Wool": {"filename": "Wool_5f91efa4.txt", "touched": "t"},'
        '"Wool!": {"filename": "Wool_9e28a3e0.txt", "touched": "t"}}}'
    )
    db = _load(
        tmp_path,
        monkeypatch,
        {
            "Mycology_91d06ad5.txt": "x",
            "Wool_5f91efa4.txt": "y",
            "Wool_9e28a3e0.txt": "z",
        },
        meta=meta,
    )
    assert set(db.wiki) == {"Mycology", "Wool", "Wool!"}
    assert set(db.wiki_mtimes) == set(db.wiki)


def test_load_wiki_falls_back_to_stripped_stem(tmp_path, monkeypatch):
    db = _load(
        tmp_path,
        monkeypatch,
        {
            "Mycology_91d06ad5.txt": "x",
            "Wool_5f91efa4.txt": "y",
        },
    )
    assert set(db.wiki) == {"Mycology", "Wool"}


def test_load_wiki_skips_legacy_unhashed(tmp_path, monkeypatch):
    db = _load(
        tmp_path,
        monkeypatch,
        {
            "Mycology_91d06ad5.txt": "x",
            "Mycology.txt": "old duplicate",
            "Plain.txt": "old duplicate 2",
        },
    )
    assert set(db.wiki) == {"Mycology"}


def test_load_wiki_defensive_collision_suffix(tmp_path, monkeypatch):
    db = _load(
        tmp_path,
        monkeypatch,
        {
            "Wool_5f91efa4.txt": "item",
            "Wool_9e28a3e0.txt": "quest",
        },
    )
    assert set(db.wiki) == {"Wool", "Wool_2"}


def test_load_wiki_preserves_hex_like_real_title(tmp_path, monkeypatch):
    db = _load(
        tmp_path,
        monkeypatch,
        {
            "Foo_12345678_abcdef12.txt": "x",
        },
    )
    assert set(db.wiki) == {"Foo_12345678"}


def test_load_wiki_unreadable_meta_falls_back(tmp_path, monkeypatch):
    (tmp_path / ".meta.json").write_text("{broken", encoding="utf-8")
    db = _load(
        tmp_path,
        monkeypatch,
        {
            "Mycology_91d06ad5.txt": "x",
        },
    )
    assert set(db.wiki) == {"Mycology"}


def test_load_wiki_empty_dir(tmp_path, monkeypatch):
    db = _load(tmp_path, monkeypatch, {})
    assert db.wiki == {}
    assert db.wiki_mtimes == {}


def test_hashed_file_re_matches_only_hashed_names():
    assert _HASHED_FILE_RE.match("Mycology_91d06ad5.txt")
    assert _HASHED_FILE_RE.match("Foo_12345678_abcdef12.txt")
    assert not _HASHED_FILE_RE.match("Mycology.txt")
    assert not _HASHED_FILE_RE.match("Mycology_91d06ad5X.txt")
