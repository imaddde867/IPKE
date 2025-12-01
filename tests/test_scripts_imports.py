"""
Tests for scripts to ensure sys.path modification allows importing project modules.
"""
import os
import sys
import subprocess
from pathlib import Path

import pytest


@pytest.mark.integration
def test_run_experiments_sys_path_allows_project_imports(tmp_path):
    """
    Ensure that importing scripts/run_experiments.py modifies sys.path to include
    the repository root so that imports like `src.*` succeed even when the repo
    root was deliberately removed from sys.path.
    """
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_experiments.py"
    assert script_path.exists(), "scripts/run_experiments.py should exist"

    # Python program to run in a clean subprocess
    program = f"""
import sys, importlib.util
from pathlib import Path
repo_root = Path({str(repo_root)!r})
# Remove repo_root from sys.path if present
sys.path = [p for p in sys.path if Path(p).resolve() != repo_root.resolve()]

# Import the script as a module (not __main__), so it won't execute main()
spec = importlib.util.spec_from_file_location("run_experiments", str(Path({str(script_path)!r})))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)  # This should add repo_root back to sys.path

# Verify we can now import a module from the project root
import src.processors.streamlined_processor as sp
import src.pipelines.baseline as baseline
print("IMPORT_OK", bool(sp.StreamlinedDocumentProcessor))
"""

    env = os.environ.copy()
    # Ensure no accidental PYTHONPATH interferes
    env.pop("PYTHONPATH", None)

    proc = subprocess.run(
        [sys.executable, "-c", program],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    if proc.returncode != 0:
        raise AssertionError(
            f"Subprocess failed (exit {proc.returncode})\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )

    assert "IMPORT_OK True" in proc.stdout, proc.stdout