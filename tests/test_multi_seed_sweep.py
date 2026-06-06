"""Smoke tests for scripts/experiments/multi_seed_sweep.py.

All tests run without a live model or GPU.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_multi_seed_sweep_dry_run_exits_zero():
    """--dry-run must print the plan and exit 0 without touching any model."""
    result = subprocess.run(
        [sys.executable, "scripts/experiments/multi_seed_sweep.py", "--dry-run"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, (
        f"--dry-run exited {result.returncode}\nstdout: {result.stdout[:500]}\nstderr: {result.stderr[:500]}"
    )
    assert len(result.stdout) > 0, "--dry-run must print the run plan to stdout"


def test_multi_seed_sweep_rejects_unknown_config():
    """Unknown --configs value must exit non-zero with a clear error."""
    result = subprocess.run(
        [sys.executable, "scripts/experiments/multi_seed_sweep.py",
         "--configs", "no_such_config", "--dry-run"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode != 0, "Unknown config must not exit 0"
    combined = result.stdout + result.stderr
    assert "no_such_config" in combined or "invalid choice" in combined, (
        f"Expected argparse rejection message in output, got:\n{combined[:500]}"
    )


def test_config_picks_up_llm_model_path(monkeypatch):
    """reload_config() must propagate LLM_MODEL_PATH into config.llm_model_path."""
    import sys
    sys.path.insert(0, str(REPO_ROOT))
    monkeypatch.setenv("LLM_MODEL_PATH", "/tmp/sentinel_model.gguf")
    monkeypatch.setenv("EXPLAINIUM_ENV", "development")

    from src.core.unified_config import reload_config
    cfg = reload_config()
    assert cfg.llm_model_path == "/tmp/sentinel_model.gguf", (
        "LLM_MODEL_PATH must be reflected in config.llm_model_path after reload_config()"
    )
