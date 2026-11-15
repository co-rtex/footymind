"""The claims ledger is the only source of quotable numbers (CLAUDE.md). These checks keep the
documents and the ledger consistent: every claim id cited in the README or docs exists, and
every committed results file the ledger names exists and records the code revision it came
from. They read only committed files (no data)."""

from __future__ import annotations

import json
import re
from pathlib import Path

from footymind.paths import REPO_ROOT

LEDGER = REPO_ROOT / "docs" / "CLAIMS_LEDGER.md"
CLAIM_ROW = re.compile(r"^\| (C\d+) \|", re.M)
CLAIM_REF = re.compile(r"\bC(\d{1,3})\b")
RESULTS_PATH = re.compile(r"`(experiments/results/[A-Za-z0-9_./-]+?/results\.json)`")


def _ledger_ids() -> set[str]:
    return set(CLAIM_ROW.findall(LEDGER.read_text()))


def _documents() -> list[Path]:
    return [REPO_ROOT / "README.md", *sorted((REPO_ROOT / "docs").rglob("*.md"))]


def test_ledger_ids_are_unique() -> None:
    rows = CLAIM_ROW.findall(LEDGER.read_text())
    assert rows and len(rows) == len(set(rows))


def test_every_cited_claim_exists_in_the_ledger() -> None:
    ids = _ledger_ids()
    missing = sorted(
        {
            (f"C{n}", str(doc.relative_to(REPO_ROOT)))
            for doc in _documents()
            for n in CLAIM_REF.findall(doc.read_text())
            if f"C{n}" not in ids
        }
    )
    assert not missing, f"claim ids cited but not in the ledger: {missing}"


def test_every_named_results_file_exists_and_records_its_revision() -> None:
    paths = sorted(set(RESULTS_PATH.findall(LEDGER.read_text())))
    assert paths
    for rel in paths:
        f = REPO_ROOT / rel
        assert f.is_file(), f"{rel} is named in the ledger but not committed"
        rev = json.loads(f.read_text()).get("code_revision")
        assert isinstance(rev, str) and re.fullmatch(r"[0-9a-f]{7,40}", rev), rel
