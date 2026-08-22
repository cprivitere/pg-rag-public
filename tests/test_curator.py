"""Tests scripts.curator.run_curator: it creates a curated area-levels
document from wiki sources and regenerates it when the source files
change."""

from pathlib import Path
from unittest.mock import patch

from scripts.curator import run_curator


def _write_wiki(tmp, files):
    wiki = Path(tmp) / "wiki"
    wiki.mkdir(exist_ok=True)
    for name, content in files.items():
        (wiki / name).write_text(content, encoding="utf-8")
    return wiki


def _run(tmp):
    curated = Path(tmp) / "curated"
    with patch("scripts.curator.WIKI_DIR", Path(tmp) / "wiki"), \
         patch("scripts.curator.CURATED_DIR", curated):
        return run_curator()


def test_v20_curator_creates_curated_doc(tmp_path):
    wiki = _write_wiki(tmp_path, {
        "area1.txt": "Eltibule is a level 20 area with good zones.",
        "area2.txt": "Serbule Hills is a level 20 area too.",
        "area3.txt": "Gazluk is a level 40 area for endgame.",
    })
    _run(tmp_path)
    created = list(wiki.parent.joinpath("curated").glob("*_curated.txt"))
    assert len(created) == 1
    assert "area_levels" in created[0].name


def test_v20_curator_regenerates_on_source_change(tmp_path):
    _write_wiki(tmp_path, {
        "area1.txt": "Eltibule is a level 20 area with good zones.",
        "area2.txt": "Serbule Hills is a level 20 area too.",
        "area3.txt": "Gazluk is a level 40 area for endgame.",
    })
    _run(tmp_path)
    curated_file = list(Path(tmp_path).joinpath("curated").glob("*_curated.txt"))[0]
    first = curated_file.read_text(encoding="utf-8")

    _write_wiki(tmp_path, {
        "area1.txt": "Eltibule is a level 25 area now — updated.",
        "area2.txt": "Serbule Hills is a level 25 area now — updated.",
        "area3.txt": "Gazluk is a level 45 area for endgame.",
    })
    _run(tmp_path)
    second = curated_file.read_text(encoding="utf-8")

    assert first != second, "V20: curator must regenerate stale docs on source change"
