"""Smoke tests for scripts/experiments/multi_seed_sweep.py.

All tests run without a live model or GPU.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

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


def test_aggregate_results_marks_partial_rows_when_seeds_missing():
    """aggregate_results must set partial=True on rows where fewer seeds completed than requested."""
    from scripts.experiments.multi_seed_sweep import aggregate_results, DEFAULT_CONFIGS

    cfg = DEFAULT_CONFIGS[0]
    seeds = [1, 2, 3]

    # Only seed 1 succeeded; seeds 2 and 3 failed (None).
    all_metrics = {
        cfg.config_id: {
            1: {"macro_avg": {"Phi": 0.7, "StepF1": 0.8, "ConstraintCoverage": 0.6}},
            2: None,
            3: None,
        }
    }

    summary_rows, _ = aggregate_results(
        all_metrics=all_metrics,
        seeds=seeds,
        configs=[cfg],
        n_bootstrap=10,
    )

    assert len(summary_rows) == 1
    assert summary_rows[0]["partial"] is True, (
        "Row must be marked partial=True when fewer seeds completed than requested"
    )


def test_aggregate_results_marks_complete_rows_when_all_seeds_succeed():
    """aggregate_results must set partial=False when all seeds produced results."""
    from scripts.experiments.multi_seed_sweep import aggregate_results, DEFAULT_CONFIGS

    cfg = DEFAULT_CONFIGS[0]
    seeds = [1, 2]
    all_metrics = {
        cfg.config_id: {
            1: {"macro_avg": {"Phi": 0.7, "StepF1": 0.8, "ConstraintCoverage": 0.6}},
            2: {"macro_avg": {"Phi": 0.75, "StepF1": 0.82, "ConstraintCoverage": 0.65}},
        }
    }

    summary_rows, _ = aggregate_results(
        all_metrics=all_metrics,
        seeds=seeds,
        configs=[cfg],
        n_bootstrap=10,
    )

    assert summary_rows[0]["partial"] is False


def test_main_exits_nonzero_when_all_evaluations_fail(tmp_path, monkeypatch):
    """main() must exit non-zero when every _evaluate_one() call returns None.

    This guards against the silent-failure mode where evaluation exceptions are
    swallowed and an incomplete summary.csv is written with exit code 0.
    """
    import scripts.experiments.multi_seed_sweep as sweep

    # Stub Phase 1 (extraction) — no model needed.
    async def _fake_extract(*args, **kwargs):
        pass

    monkeypatch.setattr(sweep, "_extract_one", _fake_extract)

    # Stub _evaluate_one to simulate a total failure (returns None every time).
    monkeypatch.setattr(sweep, "_evaluate_one", lambda **kwargs: None)

    # TIER_A_TEST_DOCS values are joined with REPO_ROOT; an absolute path
    # passes through unchanged (Python Path division semantics).
    doc_file = tmp_path / "doc1.txt"
    doc_file.write_text("content")
    monkeypatch.setattr(sweep, "TIER_A_TEST_DOCS", {"doc1": str(doc_file)})

    # parse_args() reads sys.argv directly — patch it to avoid pytest args leaking in.
    monkeypatch.setattr(sys, "argv", [
        "multi_seed_sweep.py",
        "--configs", "mistral7b_dsc_p3",
        "--seeds", "42",
        "--output-root", str(tmp_path),
    ])

    with pytest.raises(SystemExit) as exc_info:
        sweep.main()

    assert exc_info.value.code != 0, (
        "main() must exit non-zero when all _evaluate_one() calls fail"
    )


def test_resolve_model_path_with_model_dir_uses_filename_only():
    """--model-dir must not double-prefix the models/llm/ directory segment."""
    from scripts.experiments.multi_seed_sweep import _resolve_model_path

    result = _resolve_model_path(
        "models/llm/Mistral-7B-Instruct-v0.2-Q4_K_M.gguf",
        model_dir=Path("/tmp/mymodels"),
    )
    assert result == "/tmp/mymodels/Mistral-7B-Instruct-v0.2-Q4_K_M.gguf", (
        f"Expected filename-only join under model_dir, got: {result}"
    )


def test_resolve_model_path_absolute_unchanged():
    """Absolute paths must pass through unchanged regardless of model_dir."""
    from scripts.experiments.multi_seed_sweep import _resolve_model_path

    result = _resolve_model_path(
        "/abs/path/to/model.gguf",
        model_dir=Path("/tmp/mymodels"),
    )
    assert result == "/abs/path/to/model.gguf"


def test_resolve_model_path_no_model_dir_anchors_at_repo_root():
    """Without model_dir, relative paths anchor at REPO_ROOT."""
    from scripts.experiments.multi_seed_sweep import _resolve_model_path, REPO_ROOT

    result = _resolve_model_path("models/llm/some.gguf", model_dir=None)
    assert result == str(REPO_ROOT / "models/llm/some.gguf")
