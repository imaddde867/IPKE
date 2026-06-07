"""Regression tests for scripts/eval_multiseed.py gold-validation logic.

These tests exercise the path where malformed gold JSON files must be detected
before extraction begins — even during --dry-run, so a fresh-clone check can
never succeed silently on a broken dataset.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_pair(gold_dir: Path, text_dir: Path, stem: str, gold_content: str) -> None:
    """Write a (gold JSON, text) pair into their respective directories."""
    (gold_dir / f"{stem}.json").write_text(gold_content, encoding="utf-8")
    (text_dir / f"{stem}.txt").write_text("dummy content", encoding="utf-8")


def _run_main(argv: list[str]) -> int:
    """Import and invoke eval_multiseed.main() with the given argv."""
    import importlib, sys
    # Reload so module-level state from previous calls doesn't bleed in.
    import scripts.eval_multiseed as mod
    importlib.reload(mod)
    return mod.main(argv)


# ---------------------------------------------------------------------------
# Malformed gold fails under --dry-run
# ---------------------------------------------------------------------------

def test_dry_run_fails_on_malformed_gold(tmp_path):
    """--dry-run must return non-zero when a gold file contains invalid JSON.

    Without this guard, `make eval` can pass on a freshly-cloned repo even
    when the gold dataset is corrupted or partially written.
    """
    gold_dir = tmp_path / "gold"
    text_dir = tmp_path / "text"
    gold_dir.mkdir()
    text_dir.mkdir()

    _write_pair(gold_dir, text_dir, "doc1", "{invalid json !!!")

    result = _run_main([
        "--gold-dir", str(gold_dir),
        "--text-dir", str(text_dir),
        "--seeds", "1",
        "--dry-run",
    ])
    assert result != 0, (
        "--dry-run must return non-zero when a gold file is malformed"
    )


def test_dry_run_succeeds_on_valid_gold(tmp_path):
    """--dry-run must return zero when all gold files are well-formed JSON."""
    gold_dir = tmp_path / "gold"
    text_dir = tmp_path / "text"
    gold_dir.mkdir()
    text_dir.mkdir()

    _write_pair(gold_dir, text_dir, "doc1", json.dumps({"steps": []}))

    result = _run_main([
        "--gold-dir", str(gold_dir),
        "--text-dir", str(text_dir),
        "--seeds", "1",
        "--dry-run",
    ])
    assert result == 0, "--dry-run must return zero when gold is valid"


# ---------------------------------------------------------------------------
# Malformed gold fails in a full run (non-dry-run path)
# ---------------------------------------------------------------------------

def test_full_run_fails_on_malformed_gold(tmp_path):
    """Gold validation must also fire on a non-dry-run invocation.

    The hoist must not be conditional on --dry-run; a full run on broken gold
    should fail immediately rather than proceeding to the extraction phase.
    """
    gold_dir = tmp_path / "gold"
    text_dir = tmp_path / "text"
    gold_dir.mkdir()
    text_dir.mkdir()

    _write_pair(gold_dir, text_dir, "doc1", "not json at all")

    # We don't actually want extraction to run. Monkeypatching _extract_one
    # would require pytest monkeypatch, but this test can just assert the
    # return code is non-zero without reaching extraction (gold check fires first).
    result = _run_main([
        "--gold-dir", str(gold_dir),
        "--text-dir", str(text_dir),
        "--seeds", "1",
    ])
    assert result != 0, (
        "Full run must return non-zero immediately when a gold file is malformed"
    )
