"""Tests scripts.golden_check.check_golden against the golden JSON files in
GOLDEN_DIR; requires the LLM (:8080) and embedding (:8081) servers and retries
sampled LLM answers to damp nondeterminism."""

import json
from pathlib import Path

import pytest

from scripts.golden_check import check_golden, GOLDEN_DIR

_GOLDEN_FILES = sorted(GOLDEN_DIR.glob("*.json"))


def _servers_up():
    import requests

    for port in (8080, 8081):
        try:
            requests.get(f"http://127.0.0.1:{port}/health", timeout=2)
        except Exception:
            return False
    return True


@pytest.fixture()
def require_servers():
    """Runtime skip: golden checks need the LLM (:8080) + embedding (:8081)
    servers. Checked per-test so a server coming up/down mid-run doesn't make
    the suite depend on collection-time state."""
    if not _servers_up():
        pytest.skip("V38: golden check needs LLM (:8080) + embedding (:8081) servers")


@pytest.mark.parametrize("path", _GOLDEN_FILES, ids=lambda p: p.stem)
def test_golden_facts(path, require_servers):
    golden = json.loads(path.read_text(encoding="utf-8"))
    # LLM answers are sampled: retry once to damp nondeterminism. A real
    # regression still fails both attempts.
    for _attempt in range(2):
        _, misses = check_golden(golden)
        if not misses:
            return
    assert misses == [], f"missing facts: {misses}"
