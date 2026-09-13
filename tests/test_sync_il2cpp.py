"""Contract tests for scripts/sync_il2cpp.py orchestration.

True unit tests: every external touchpoint (game client dir, IL2CPP_DIR,
dumper subprocess) is monkeypatched — the real data/il2cpp tree and Steam
client are never touched. Contracts: staging is idempotent (same size +
mtime → no copy), dumper failure and bad dump output leave the previous
out_lean/Dump0 intact (restore-on-failure), and a successful swap replaces
out_lean/Dump0 with the fresh dump in one shot.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import sync_il2cpp

MARKER = "Z" * 2_000_000  # > sync_il2cpp.PAYLOAD_MIN_DUMP_BYTES (1 MB)


def _make_dumper(il2cpp):
    dumper = il2cpp / "tools" / "il2cpp-dumper-rs" / "target" / "release" / "il2cpp_dumper.exe"
    dumper.parent.mkdir(parents=True, exist_ok=True)
    dumper.write_bytes(b"MZ\x00\x00")  # file exists; content irrelevant
    return dumper


def _seed_client(game: Path):
    (game / "GameAssembly.dll").write_bytes(b"GA")
    meta = game / "WindowsPlayer_Data" / "il2cpp_data" / "Metadata"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "global-metadata.dat").write_bytes(b"META")


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """Point IL2CPP_DIR / game client at tmp dirs; stub the dumper.

    fake_run parses `-o OUT` and writes marker payload files under
    OUT/Dump0, standing in for real dumper output.
    """
    il2cpp = tmp_path / "data" / "il2cpp"
    il2cpp.mkdir(parents=True)
    game = tmp_path / "game"
    game.mkdir()
    monkeypatch.setenv(sync_il2cpp.IL2CPP_ROOT_ENV, str(tmp_path))
    monkeypatch.setenv("GAME_INSTALL_DIR", str(game))
    calls: list[tuple[str, ...]] = []

    def fake_run(args, **kwargs):
        calls.append(tuple(args))
        out = Path(args[args.index("-o") + 1])
        dump = out / "Dump0"
        dump.mkdir(parents=True, exist_ok=True)
        (dump / "dump.cs").write_text(MARKER, encoding="utf-8")
        (dump / "stringliteral.json").write_text("[]", encoding="utf-8")

    monkeypatch.setattr(sync_il2cpp.subprocess, "run", fake_run)
    _make_dumper(il2cpp)
    return SimpleNamespace(il2cpp=il2cpp, game=game, calls=calls, set_run=monkeypatch.setattr)


def test_missing_client_binary_fails(isolated):
    # game dir empty: no GameAssembly.dll staged
    assert sync_il2cpp.main() == 1
    assert not (isolated.il2cpp / "out_lean").exists()
    assert isolated.calls == []  # dumper never invoked


def test_stage_copies_when_stale_skips_when_current(isolated):
    _seed_client(isolated.game)
    assert sync_il2cpp.main() == 0
    ga = isolated.il2cpp / "GameAssembly.dll"
    meta = isolated.il2cpp / "global-metadata.dat"
    assert ga.read_bytes() == b"GA"
    assert meta.read_bytes() == b"META"

    ga_stat, meta_stat = ga.stat(), meta.stat()
    fingerprint = (ga_stat.st_mtime_ns, ga_stat.st_size, meta_stat.st_mtime_ns, meta_stat.st_size)
    assert sync_il2cpp.main() == 0
    ga_stat2, meta_stat2 = ga.stat(), meta.stat()
    assert (
        ga_stat2.st_mtime_ns,
        ga_stat2.st_size,
        meta_stat2.st_mtime_ns,
        meta_stat2.st_size,
    ) == fingerprint  # identical dst → copy2 skipped, nothing truncated
    assert len(isolated.calls) == 2  # dumper re-ran on both passes (always fresh)


def test_missing_dumper_returns_1(isolated):
    _seed_client(isolated.game)
    dumper = (
        isolated.il2cpp / "tools" / "il2cpp-dumper-rs" / "target" / "release" / "il2cpp_dumper.exe"
    )
    dumper.unlink()
    assert sync_il2cpp.main() == 1
    # staging succeeded even though the dumper is missing
    assert (isolated.il2cpp / "GameAssembly.dll").read_bytes() == b"GA"
    assert not (isolated.il2cpp / "out_lean").exists()


def test_swap_replaces_out_lean(isolated):
    _seed_client(isolated.game)
    old = isolated.il2cpp / "out_lean" / "Dump0"
    old.mkdir(parents=True)
    (old / "dump.cs").write_text("OLD", encoding="utf-8")

    assert sync_il2cpp.main() == 0
    assert (isolated.il2cpp / "out_lean" / "Dump0" / "dump.cs").read_text(
        encoding="utf-8"
    ) == MARKER
    assert (isolated.il2cpp / "out_lean" / "Dump0" / "stringliteral.json").read_text(
        encoding="utf-8"
    ) == "[]"
    assert not (isolated.il2cpp / "out_new").exists()  # shell dir removed
    assert not (isolated.il2cpp / "out_lean" / "Dump0_prev").exists()


def test_bad_dump_keeps_previous_out_lean(isolated):
    _seed_client(isolated.game)
    old = isolated.il2cpp / "out_lean" / "Dump0"
    old.mkdir(parents=True)
    (old / "dump.cs").write_text("OLD", encoding="utf-8")
    (old / "stringliteral.json").write_text("[]", encoding="utf-8")

    def bad_run(args, **kwargs):
        out = Path(args[args.index("-o") + 1])
        dump = out / "Dump0"
        dump.mkdir(parents=True, exist_ok=True)
        (dump / "dump.cs").write_text("", encoding="utf-8")  # too small → sanity fails

    isolated.set_run(sync_il2cpp.subprocess, "run", bad_run)
    assert sync_il2cpp.main() == 1
    assert (isolated.il2cpp / "out_lean" / "Dump0" / "dump.cs").read_text(
        encoding="utf-8"
    ) == "OLD"
    assert (isolated.il2cpp / "out_lean" / "Dump0" / "stringliteral.json").read_text(
        encoding="utf-8"
    ) == "[]"
    assert not (isolated.il2cpp / "out_new").exists()  # no half-swapped leftovers
    assert not (isolated.il2cpp / "out_lean" / "Dump0_prev").exists()
