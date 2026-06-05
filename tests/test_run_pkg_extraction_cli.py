import os
from pathlib import Path

from scripts.run_pkg_extraction import _apply_env_overrides, parse_args


def test_parse_args_accepts_dsc_alias():
    args = parse_args(["--chunking-method", "dsc"])
    assert args.chunking_method == "dsc"


def test_parse_args_defaults_to_runs_output_dir():
    args = parse_args([])
    assert Path(args.output_dir).parts[:2] == ("runs", "pkg_extraction")


def test_apply_env_overrides_preserves_dsc_alias(monkeypatch):
    monkeypatch.delenv("CHUNKING_METHOD", raising=False)
    _apply_env_overrides(
        chunking_method="dsc",
        prompting_strategy="P3",
        gpu_backend="cpu",
        llm_backend=None,
        hf_model_id=None,
        hf_quantization=None,
    )
    assert os.environ["CHUNKING_METHOD"] == "dsc"
