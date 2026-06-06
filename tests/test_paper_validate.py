from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent


def test_paper_validate_runs_clean() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/paper_validate.py"],
        capture_output=True,
        text=True,
        cwd=REPO,
        check=False,
    )
    assert result.returncode == 0, f"paper_validate failed: {result.stderr or result.stdout}"
    assert "paper-validate" in result.stdout or "OK" in result.stdout, (
        f"missing one-line summary: {result.stdout!r}"
    )
    assert "validated" in result.stdout.lower(), (
        f"expected validation count in stdout: {result.stdout!r}"
    )
