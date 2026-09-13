"""One command to converge the IL2CPP decomp source to the local Steam client.

Stages GameAssembly.dll + global-metadata.dat from the Steam install,
re-runs the IL2CPP dumper, and swaps the result into the payload dir the
document builder reads (`data/il2cpp/out_lean`). Run via `mise sync-il2cpp`.
"""

import os
import shutil
import subprocess
from pathlib import Path

from pgrag.config import PROJECT_ROOT

IL2CPP_ROOT_ENV = "PG_IL2CPP_ROOT"


def _il2cpp_root() -> Path:
    """Data-side root override: run the script from any worktree but act on
    the canonical data tree (defaults to PROJECT_ROOT / data / il2cpp)."""
    return Path(os.environ.get(IL2CPP_ROOT_ENV) or PROJECT_ROOT) / "data" / "il2cpp"


GAME_INSTALL_DIR_ENV = "GAME_INSTALL_DIR"
GAME_INSTALL_DIR_DEFAULT = r"C:\Program Files (x86)\Steam\steamapps\common\Project Gorgon"


def _dumper() -> Path:
    return _il2cpp_root() / "tools" / "il2cpp-dumper-rs" / "target" / "release" / "il2cpp_dumper.exe"


def _config() -> Path:
    return _il2cpp_root() / "config.json"


def _out_lean_dir() -> Path:
    return _il2cpp_root() / "out_lean"


def _out_new_dir() -> Path:
    return _il2cpp_root() / "out_new"


# Payload files the decomp document builder requires (decomp_builder.py reads
# exactly out_lean/Dump0/dump.cs + stringliteral.json).
PAYLOAD_MIN_DUMP_BYTES = 1_000_000


def _stage_file(src: Path, dst: Path) -> None:
    if dst.is_file() and (
        dst.stat().st_mtime_ns == src.stat().st_mtime_ns and dst.stat().st_size == src.stat().st_size
    ):
        print(f"unchanged {dst.name}")
        return
    shutil.copy2(src, dst)
    print(f"staged {dst.name} from {src}")


def _game_client_pair() -> tuple[Path, Path]:
    install = os.environ.get(GAME_INSTALL_DIR_ENV) or GAME_INSTALL_DIR_DEFAULT
    return (
        Path(install) / "GameAssembly.dll",
        Path(install) / "WindowsPlayer_Data/il2cpp_data/Metadata/global-metadata.dat",
    )

def _staged_binaries() -> tuple[Path, Path]:
    return _il2cpp_root() / "GameAssembly.dll", _il2cpp_root() / "global-metadata.dat"


def _dumper_ready() -> bool:
    return _dumper().is_file()


def _dump(args: list[str]) -> None:
    subprocess.run(args, check=False)


def _rm(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def _swap(out_lean: Path, out_new: Path) -> None:
    """Replace out_lean/Dump0 with out_new/Dump0 in one shot.

    Fresh checkout: no Dump0 yet — just rename the new one in. On rename
    failure the previous dump is restored (Dump0_prev back to Dump0), so
    out_lean stays intact at the previous state.
    """
    out_lean.mkdir(parents=True, exist_ok=True)
    payload = out_lean / "Dump0"
    payload_new = out_new / "Dump0"
    prev: Path | None = None
    if payload.exists():
        prev = out_lean / "Dump0_prev"
        _rm(prev)
        payload.rename(prev)
    payload_new.rename(payload)
    if prev is not None:
        _rm(prev)
    _rm(out_new)



def main() -> int:
    ga_src, meta_src = _game_client_pair()
    ga_dst, meta_dst = _staged_binaries()
    for src, dst in ((ga_src, ga_dst), (meta_src, meta_dst)):
        if not src.is_file():
            print(
                f"error: missing client binary {src} — set {GAME_INSTALL_DIR_ENV} to the live Steam client root (default {GAME_INSTALL_DIR_DEFAULT})",
            )
            return 1
        _stage_file(src, dst)

    if not _dumper_ready():
        print(
            f"error: IL2CPP dumper not found at {_dumper()}\n"
            "  build it: cd data/il2cpp/tools/il2cpp-dumper-rs && cargo build --release\n"
            "  or fetch a prebuilt release from https://github.com/rodroidmods/il2cpp-dumper-rs/releases"
        )
        return 1

    ga, meta = _staged_binaries()
    out_lean, out_new = _out_lean_dir(), _out_new_dir()
    _rm(out_new)
    _dump([str(_dumper()), "-c", str(_config()), "-b", str(ga), "-m", str(meta), "-o", str(out_new)])
    dump_cs = out_new / "Dump0" / "dump.cs"
    size = dump_cs.stat().st_size if dump_cs.is_file() else 0
    if size <= PAYLOAD_MIN_DUMP_BYTES or not (out_new / "Dump0" / "stringliteral.json").is_file():
        print(f"error: dump output missing or suspicious in {out_new / 'Dump0'}")
        _rm(out_new)
        return 1

    _swap(out_lean, out_new)
    print(f"out_lean refreshed from {ga_src.parent.parent.parent}")  # steamapps/common/Project Gorgon
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
