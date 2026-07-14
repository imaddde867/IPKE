"""Regression tests for scripts/eval_multiseed.py gold-validation logic.

These tests exercise the path where malformed gold JSON files must be detected
before extraction begins — even during --dry-run, so a fresh-clone check can
never succeed silently on a broken dataset.
"""
from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_pair(gold_dir: Path, text_dir: Path, stem: str, gold_content: str) -> None:
    """Write a (gold JSON, text) pair into their respective directories."""
    (gold_dir / f"{stem}.json").write_text(gold_content, encoding="utf-8")
    (text_dir / f"{stem}.txt").write_text("dummy content", encoding="utf-8")


def _manifest_entry(
    doc_id: str,
    *,
    include: bool,
    status: str = "candidate",
) -> dict[str, object]:
    return {
        "doc_id": doc_id,
        "source_family": "test",
        "role": "procedure_candidate",
        "status": status,
        "include_for_evaluation": include,
        "reason": "Test fixture.",
    }


def _write_manifest(path: Path, documents: list[dict[str, object]]) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "manifest_status": "provisional",
                "documents": documents,
            }
        ),
        encoding="utf-8",
    )
    return path


def _run_main(argv: list[str]) -> int:
    """Import and invoke eval_multiseed.main() with the given argv."""
    import importlib
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


def test_paper_evidence_run_requires_manifest(tmp_path, capsys):
    gold_dir = tmp_path / "gold"
    text_dir = tmp_path / "text"
    evidence_dir = tmp_path / "evidence"
    gold_dir.mkdir()
    text_dir.mkdir()
    evidence_dir.mkdir()
    _write_pair(gold_dir, text_dir, "doc1", json.dumps({"steps": []}))
    (evidence_dir / "doc1.json").write_text("{}", encoding="utf-8")

    result = _run_main(
        [
            "--gold-dir",
            str(gold_dir),
            "--text-dir",
            str(text_dir),
            "--evidence-dir",
            str(evidence_dir),
            "--seeds",
            "1",
        ]
    )

    assert result == 1
    assert "--manifest is required for paper-evidence runs" in capsys.readouterr().err


def test_paper_evidence_run_rejects_provisional_manifest(tmp_path, capsys):
    gold_dir = tmp_path / "gold"
    text_dir = tmp_path / "text"
    evidence_dir = tmp_path / "evidence"
    gold_dir.mkdir()
    text_dir.mkdir()
    evidence_dir.mkdir()
    _write_pair(gold_dir, text_dir, "doc1", json.dumps({"steps": []}))
    (evidence_dir / "doc1.json").write_text("{}", encoding="utf-8")
    manifest_path = _write_manifest(
        tmp_path / "manifest.json",
        [_manifest_entry("doc1", include=True)],
    )

    result = _run_main(
        [
            "--gold-dir",
            str(gold_dir),
            "--text-dir",
            str(text_dir),
            "--evidence-dir",
            str(evidence_dir),
            "--manifest",
            str(manifest_path),
            "--seeds",
            "1",
        ]
    )

    assert result == 1
    assert "frozen manifest required" in capsys.readouterr().err


def test_manifest_dry_run_ignores_malformed_excluded_gold(tmp_path, capsys):
    gold_dir = tmp_path / "gold"
    text_dir = tmp_path / "text"
    gold_dir.mkdir()
    text_dir.mkdir()
    _write_pair(gold_dir, text_dir, "good", json.dumps({"steps": []}))
    _write_pair(gold_dir, text_dir, "bad", "{malformed")
    manifest_path = _write_manifest(
        tmp_path / "manifest.json",
        [
            _manifest_entry("good", include=True),
            _manifest_entry(
                "bad",
                include=False,
                status="excluded_pending_reannotation",
            ),
        ],
    )

    result = _run_main(
        [
            "--gold-dir",
            str(gold_dir),
            "--text-dir",
            str(text_dir),
            "--manifest",
            str(manifest_path),
            "--seeds",
            "1",
            "--dry-run",
        ]
    )
    captured = capsys.readouterr()

    assert result == 0
    assert "Plan: 1 docs" in captured.out
    assert "development-only" in captured.err


def test_manifest_dry_run_fails_cleanly_on_unclassified_gold(tmp_path, capsys):
    gold_dir = tmp_path / "gold"
    text_dir = tmp_path / "text"
    gold_dir.mkdir()
    text_dir.mkdir()
    _write_pair(gold_dir, text_dir, "included", json.dumps({"steps": []}))
    _write_pair(gold_dir, text_dir, "unclassified", json.dumps({"steps": []}))
    manifest_path = _write_manifest(
        tmp_path / "manifest.json",
        [_manifest_entry("included", include=True)],
    )

    result = _run_main(
        [
            "--gold-dir",
            str(gold_dir),
            "--text-dir",
            str(text_dir),
            "--manifest",
            str(manifest_path),
            "--seeds",
            "1",
            "--dry-run",
        ]
    )
    captured = capsys.readouterr()

    assert result == 1
    assert "ERROR: invalid corpus manifest" in captured.err
    assert "unclassified files: unclassified" in captured.err


def test_manifest_dry_run_fails_when_selected_text_is_missing(tmp_path, capsys):
    gold_dir = tmp_path / "gold"
    text_dir = tmp_path / "text"
    gold_dir.mkdir()
    text_dir.mkdir()
    _write_pair(gold_dir, text_dir, "present", json.dumps({"steps": []}))
    (gold_dir / "missing.json").write_text(
        json.dumps({"steps": []}), encoding="utf-8"
    )
    manifest_path = _write_manifest(
        tmp_path / "manifest.json",
        [
            _manifest_entry("present", include=True),
            _manifest_entry("missing", include=True),
        ],
    )

    result = _run_main(
        [
            "--gold-dir",
            str(gold_dir),
            "--text-dir",
            str(text_dir),
            "--manifest",
            str(manifest_path),
            "--seeds",
            "1",
            "--dry-run",
        ]
    )
    captured = capsys.readouterr()

    assert result == 1
    assert "text files missing for manifest-selected gold: missing" in captured.err


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
