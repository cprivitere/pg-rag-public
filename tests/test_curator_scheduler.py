"""Tests scripts.curator_scheduler.run_curator_with_scheduler ordering: the
build-documents subprocess must run before build-index (V25)."""

import sys
from unittest.mock import MagicMock, patch

from scripts.curator_scheduler import run_curator_with_scheduler


def test_v25_scheduler_runs_main_before_build_index():
    calls = []

    def fake_subprocess(args, **kwargs):
        calls.append(args)
        return MagicMock(returncode=0, stdout="ok", stderr="")

    with patch("scripts.curator_scheduler.load_state", return_value={"files_hash": {}}), \
         patch("scripts.curator_scheduler.get_wiki_hash", return_value={"page.txt": "h"}), \
         patch("scripts.curator_scheduler.detect_changes", return_value=True), \
         patch("scripts.curator_scheduler.save_state") as save_state, \
         patch("scripts.curator_scheduler.subprocess.run", side_effect=fake_subprocess):
        run_curator_with_scheduler()

    main_idx = calls.index([sys.executable, "-m", "pgrag.cli", "build-documents"])
    build_idx = calls.index([sys.executable, "-m", "pgrag.cli", "build-index"])
    assert main_idx < build_idx, "V25: build-documents must run before build-index"
