import json
import sys

from scripts.experiments import build_summary_csv


def test_parse_args_defaults_output_to_latest_paper_summary():
    args = build_summary_csv.parse_args([])

    assert args.output == build_summary_csv.REPO_ROOT / "runs" / "latest" / "paper_summary.csv"


def test_discover_summary_run_dirs_finds_nested_summary_rows(tmp_path):
    root = tmp_path / "runs"
    first_run = root / "experiments" / "run-a"
    second_run = root / "chunking_sweeps" / "sweep" / "run-b"
    first_run.mkdir(parents=True)
    second_run.mkdir(parents=True)
    (first_run / "summary_row.json").write_text("{}", encoding="utf-8")
    (second_run / "summary_row.json").write_text("{}", encoding="utf-8")

    discovered = build_summary_csv.discover_summary_run_dirs("summary_row.json", roots=(root,))

    assert set(discovered) == {first_run, second_run}


def test_main_discovers_default_roots_and_writes_csv(tmp_path, monkeypatch):
    summary_root = tmp_path / "summaries"
    run_dir = summary_root / "experiment-1"
    output = tmp_path / "latest" / "paper_summary.csv"
    run_dir.mkdir(parents=True)
    (run_dir / "summary_row.json").write_text(json.dumps({"model": "mistral", "Phi": 0.699}), encoding="utf-8")
    monkeypatch.setattr(build_summary_csv, "DEFAULT_SUMMARY_ROOTS", (summary_root,))
    monkeypatch.setattr(build_summary_csv, "DEFAULT_OUTPUT", output)
    monkeypatch.setattr(sys, "argv", ["build_summary_csv.py"])

    exit_code = build_summary_csv.main()

    assert exit_code == 0
    assert output.read_text(encoding="utf-8") == "Phi,model\n0.699,mistral\n"
