from scripts.experiments import experiment_utils
from scripts.experiments import run_all_chunking_experiments


def test_default_experiment_root_uses_runs():
    parts = experiment_utils.DEFAULT_RESULTS_ROOT.relative_to(experiment_utils.REPO_ROOT).parts
    assert parts[:2] == ("runs", "experiments")


def test_master_run_roots_use_runs():
    roots = run_all_chunking_experiments.build_run_roots("20260604_120000")
    assert roots["run_root"].relative_to(run_all_chunking_experiments.REPO_ROOT).parts[:2] == (
        "runs",
        "chunking_sweeps",
    )
    assert roots["log_dir"].relative_to(run_all_chunking_experiments.REPO_ROOT).parts[:2] == (
        "runs",
        "master_logs",
    )
    assert roots["latest_summary"].relative_to(run_all_chunking_experiments.REPO_ROOT).parts[:2] == (
        "runs",
        "latest",
    )
