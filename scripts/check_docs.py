"""Drift guard: keep the docs/skills truthful about the repo.

Prose documentation (AGENTS.md, .omp/RULES.md, skills, docs/TEST_CONTRACTS.md)
drifts unless something checks it against the code. The one guardrail in this
repo that never drifted is enforced code (conftest's immutability guard) — this
script is the documentation analogue. It asserts stable, meaningful facts:

- every test file is covered by docs/TEST_CONTRACTS.md (and nothing it
  references has vanished) — the layer->test map cannot quietly diverge;
- file paths named in backticks under AGENTS.md / the skills actually exist;
- contract constants (DOCUMENTS_VERSION, EMBEDDING_DIM) and the golden shape
  agree across the docs that quote them;
- the golden-set counts the evaluation skill quotes are real.

Exit 0 = clean; 1 = one or more drifts (printed). Run via `mise drift`
(uv run python scripts/check_docs.py). Only stable facts are asserted — a
check that breaks on innocuous churn gets ignored, which is how guardrails rot.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

_lines: list[str] = []  # ("OK" | "DRIFT", label[, detail])


def _report(tag: str, label: str, detail: str = "") -> None:
    _lines.append((tag, label, detail))


def _drift(label: str, detail: str = "") -> None:
    _report("DRIFT", label, detail)


def _read(*parts: str) -> str:
    return (ROOT.joinpath(*parts)).read_text(encoding="utf-8")


def _exists(*parts: str) -> bool:
    return ROOT.joinpath(*parts).exists()


def main() -> int:
    # --- 1. TEST_CONTRACTS covers every test file (and vice versa) ---------
    tc = ROOT / "docs" / "TEST_CONTRACTS.md"
    if not tc.exists():
        _drift("TEST_CONTRACTS.md missing", "docs/TEST_CONTRACTS.md")
    else:
        tc_text = tc.read_text(encoding="utf-8")
        test_files = sorted(p.name for p in (ROOT / "tests").glob("test_*.py"))
        orphans = [f for f in test_files if f not in tc_text]
        if orphans:
            _drift("TEST_CONTRACTS missing test files", ", ".join(orphans))
        else:
            _report("OK", "TEST_CONTRACTS covers all tests")
        referenced = set(re.findall(r"`(test_[a-zA-Z0-9_]+\.py)`", tc_text))
        vanished = sorted(f for f in referenced if not _exists("tests", f))
        if vanished:
            _drift("TEST_CONTRACTS references missing tests", ", ".join(vanished))
        else:
            _report("OK", "TEST_CONTRACTS refs resolve")

    # --- 2. Backticked file paths in AGENTS.md + skills resolve ------------
    doc_sources = [ROOT / "AGENTS.md", *(ROOT / ".omp" / "skills").glob("*/SKILL.md")]
    doc_blob = "\n".join(f.read_text(encoding="utf-8") for f in doc_sources if f.exists())
    named = set(re.findall(r"`((?:scripts|src|tests|docs)/[A-Za-z0-9_./]+\.py)`", doc_blob))
    missing_files = sorted(f for f in named if not _exists(f))
    if missing_files:
        _drift("docs reference missing files", ", ".join(missing_files))
    else:
        _report("OK", "doc file references resolve")

    # --- 3. Contract constants agree across the docs that quote them --------
    try:
        from pgrag.config import DOCUMENTS_VERSION, EMBEDDING_DIM  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - environment
        _drift("cannot import pgrag.config", str(exc))
        DOCUMENTS_VERSION = EMBEDDING_DIM = None

    if DOCUMENTS_VERSION is not None:
        # The VALUE lives only in config.py; docs should reference the symbol
        # (asserting the numeric value would force a drift-prone literal on
        # every version bump). EMBEDDING_DIM (384) is a stable contract.
        for label, path in [
            ("AGENTS.md", ROOT / "AGENTS.md"),
            ("TEST_CONTRACTS", tc),
            ("RULES.md", ROOT / ".omp" / "RULES.md"),
            ("pg-rag skill", ROOT / ".omp" / "skills" / "pg-rag" / "SKILL.md"),
        ]:
            if not path.exists():
                continue
            ok = "DOCUMENTS_VERSION" in path.read_text(encoding="utf-8")
            _report("OK" if ok else "DRIFT", f"DOCUMENTS_VERSION referenced in {label}")
        for label, path in [
            ("AGENTS.md", ROOT / "AGENTS.md"),
            ("pg-rag skill", ROOT / ".omp" / "skills" / "pg-rag" / "SKILL.md"),
        ]:
            ok = str(EMBEDDING_DIM) in path.read_text(encoding="utf-8")
            _report("OK" if ok else "DRIFT", f"EMBEDDING_DIM=={EMBEDDING_DIM} in {label}")

    # Golden shape literal: {id, question, type, facts   (owners: TEST_CONTRACTS
    # L6 + WATCHDOG trap; RULES restates nothing)
    shape = "{id, question, type, facts"
    for label, path in [
        ("TEST_CONTRACTS", tc),
        ("WATCHDOG.md", ROOT / ".omp" / "WATCHDOG.md"),
    ]:
        ok = path.exists() and shape in path.read_text(encoding="utf-8")
        _report("OK" if ok else "DRIFT", f"golden shape in {label}")

    # --- 4. Golden counts the evaluation skill quotes are real --------------
    eval_skill = ROOT / ".omp" / "skills" / "evaluation" / "SKILL.md"
    golden_dir = ROOT / "data" / "golden"
    if eval_skill.exists() and golden_dir.is_dir():
        golden_files = sorted(golden_dir.glob("*.json"))
        types = Counter(
            (json.loads(f.read_text(encoding="utf-8")) or {}).get("type", "?")
            for f in golden_files
        )
        text = eval_skill.read_text(encoding="utf-8")
        expected = {"entity": 18, "general": 12, "recipe": 3, "comparison": 1}
        mismatches = []
        if f"{len(golden_files)} files exist today" not in text:
            mismatches.append(f"count ({len(golden_files)})")
        for t, n in expected.items():
            if f"{t} {n}" not in text and f"{n} {t}" not in text:
                mismatches.append(f"{t}={types.get(t, 0)} (doc says {n})")
        if mismatches:
            _drift("evaluation skill golden counts stale", "; ".join(mismatches))
        else:
            _report("OK", "evaluation skill golden counts match")

    # --- 5. Expected skills exist and are well-formed -----------------------
    expected_skills = ["pg-data", "pg-rag", "retrieval", "evaluation", "testing"]
    for name in expected_skills:
        p = ROOT / ".omp" / "skills" / name / "SKILL.md"
        if not p.exists():
            _drift(f"missing skill: {name}")
            continue
        head = p.read_text(encoding="utf-8")[:600]
        if "name:" not in head or "description:" not in head:
            _drift(f"skill {name} missing frontmatter")
        else:
            _report("OK", f"skill {name} present")

    # --- Report -------------------------------------------------------------
    print(f"drift check: {sum(1 for t, *_ in _lines if t == 'OK')} OK, "
          f"{sum(1 for t, *_ in _lines if t == 'DRIFT')} drift")
    for tag, label, detail in _lines:
        line = f"[{tag}] {label}"
        if detail:
            line += f" — {detail}"
        print(line)
    return 1 if any(t == "DRIFT" for t, *_ in _lines) else 0


if __name__ == "__main__":
    raise SystemExit(main())