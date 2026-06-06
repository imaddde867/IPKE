from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
FULL_REPORT = REPO / "datasets/paper/reports/issue_53_iaa_report.json"
SMOKE_REPORT = REPO / "datasets/paper/reports/issue_53_iaa_report.smoke.json"


def _run_smoke() -> None:
    if SMOKE_REPORT.exists():
        SMOKE_REPORT.unlink()
    result = subprocess.run(
        [sys.executable, "scripts/compute_iaa.py", "--smoke-only"],
        capture_output=True,
        text=True,
        cwd=REPO,
        check=False,
    )
    assert result.returncode == 0, f"smoke run failed: {result.stderr}"
    assert SMOKE_REPORT.exists(), "smoke report was not written"


def test_smoke_run_writes_tiny_report() -> None:
    _run_smoke()
    report = json.loads(SMOKE_REPORT.read_text(encoding="utf-8"))
    assert report["mode"] == "smoke"
    assert report["n_files"] == 1
    assert report["documents_checked"] == 1
    forbidden = {"pairwise_f1", "step_set_jaccard", "constraint_set_jaccard", "per_document"}
    for field in forbidden:
        assert field not in report, f"smoke report must not contain {field!r}"


def test_smoke_run_does_not_mutate_full_report() -> None:
    snapshot = FULL_REPORT.read_text(encoding="utf-8") if FULL_REPORT.exists() else None
    _run_smoke()
    if snapshot is None:
        assert not FULL_REPORT.exists()
    else:
        assert FULL_REPORT.read_text(encoding="utf-8") == snapshot
