"""Unit tests for tools/annotate_gold.py, tools/iaa_check.py, and scripts/eval_multiseed.py.

All tests run without a live model, GPU, or network access.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.annotate_gold import _extraction_to_draft, _load_schema, _validate
from scripts.compute_iaa import compare_annotations
from scripts.eval_multiseed import _ci95, _summary_rows, paired_bootstrap_pvalue, METRIC_COLUMNS
from src.evaluation.metrics import compute_phi


# ---------------------------------------------------------------------------
# annotate_gold helpers
# ---------------------------------------------------------------------------

def test_extraction_to_draft_basic():
    payload = {
        "steps": [
            {"id": "S1", "text": "Inspect the pump", "order": 1},
            {"id": "S2", "text": "Tighten the bolts", "order": 2},
        ],
        "constraints": [
            {"id": "C1", "text": "Wear gloves", "steps": ["S1"]},
        ],
    }
    draft = _extraction_to_draft("doc1", payload, "Test Procedure", "maintenance")
    assert draft["procedure"]["doc_id"] == "doc1"
    assert draft["procedure"]["title"] == "Test Procedure"
    assert len(draft["steps"]) == 2
    s1 = next(s for s in draft["steps"] if s["id"] == "S1")
    assert s1["label"] == "Inspect the pump"
    # Constraint should be nested under S1
    assert len(s1["constraints"]) == 1
    assert s1["constraints"][0]["id"] == "C1"
    # S2 has no constraints
    s2 = next(s for s in draft["steps"] if s["id"] == "S2")
    assert s2["constraints"] == []


def test_extraction_to_draft_missing_text_falls_back_to_id():
    payload = {"steps": [{"id": "S1"}], "constraints": []}
    draft = _extraction_to_draft("doc1", payload, "", "")
    assert draft["steps"][0]["label"] == "S1"


def test_extraction_to_draft_quality_uses_review_status():
    payload = {"steps": [{"id": "S1", "text": "Do thing", "order": 1}], "constraints": []}
    draft = _extraction_to_draft("doc1", payload, "T", "domain")
    quality = draft["quality"]
    assert quality.get("review_status") == "unreviewed", "must use review_status not status"
    assert "status" not in quality, "old 'status' field must not appear"


def test_extraction_to_draft_validates_against_schema(tmp_path):
    payload = {
        "steps": [{"id": "S1", "text": "Do the thing", "order": 1}],
        "constraints": [],
    }
    schema = _load_schema()
    draft = _extraction_to_draft("test_doc", payload, "A Procedure", "test")
    errors = _validate(draft, schema)
    assert errors == [], f"Unexpected validation errors: {errors}"


def test_validate_reports_missing_procedure():
    schema = _load_schema()
    bad_draft = {"steps": [{"id": "S1", "label": "step"}]}
    errors = _validate(bad_draft, schema)
    assert any("procedure" in e for e in errors)


def test_validate_reports_empty_steps():
    schema = _load_schema()
    bad_draft = {"procedure": {"doc_id": "x", "title": "y"}, "steps": []}
    errors = _validate(bad_draft, schema)
    assert any("steps" in e for e in errors)


# ---------------------------------------------------------------------------
# iaa_check — exercises compare_annotations which iaa_check wraps
# ---------------------------------------------------------------------------

def test_iaa_identical_annotations_perfect_scores():
    ann = {
        "steps": [{"id": "S1", "label": "Inspect the pump"}],
        "constraints": [{"id": "C1", "text": "Wear gloves", "applies_to": ["S1"]}],
    }
    import copy
    metrics = compare_annotations(ann, copy.deepcopy(ann))
    assert metrics["step_exact"]["f1"] == 1.0
    assert metrics["constraint_exact"]["f1"] == 1.0
    assert metrics["token_label_kappa"] == 1.0


def test_iaa_different_annotations_lower_scores():
    ann_a = {
        "steps": [
            {"id": "S1", "label": "Inspect the pump"},
            {"id": "S2", "label": "Tighten the bolts"},
        ],
        "constraints": [{"id": "C1", "text": "Wear gloves", "applies_to": ["S1"]}],
    }
    ann_b = {
        "steps": [
            {"id": "S1", "label": "Inspect the pump"},
            {"id": "S2", "label": "Close the valve"},
        ],
        "constraints": [],
    }
    metrics = compare_annotations(ann_a, ann_b)
    assert metrics["step_exact"]["f1"] < 1.0
    assert metrics["constraint_exact"]["f1"] is not None


# ---------------------------------------------------------------------------
# eval_multiseed helpers
# ---------------------------------------------------------------------------

def test_ci95_correct_interval():
    vals = [0.70, 0.75, 0.72]
    lo, hi = _ci95(vals)
    mean = sum(vals) / 3
    # Interval must be centred on mean and symmetric
    assert abs((lo + hi) / 2 - mean) < 1e-9
    assert hi > mean > lo


def test_ci95_single_value_returns_nan():
    lo, hi = _ci95([0.5])
    import math
    assert math.isnan(lo) and math.isnan(hi)


def _make_detail_rows(n: int = 3) -> list[dict]:
    return [
        {"model_id": "m1", "chunker": "dsc", "prompter": "P3", "doc_id": f"d{i}", "seed": 0,
         "StepF1": 0.7 + i * 0.01, "AdjacencyF1": 0.5, "Kendall": 0.8,
         "ConstraintCoverage": 0.6, "ConstraintAttachmentF1": 0.4, "Phi": 0.65}
        for i in range(n)
    ]


def test_summary_rows_contains_metrics():
    rows = _make_detail_rows(3)
    summary = _summary_rows(rows, [(0.5, 0.3, 0.2)])
    assert len(summary) == 1
    s = summary[0]
    assert s["model_id"] == "m1"
    assert s["chunker"] == "dsc"
    for m in METRIC_COLUMNS:
        assert f"{m}_mean" in s
    # CI exists (3 values)
    assert s["StepF1_ci_lo"] is not None


def test_summary_rows_phi_sensitivity():
    rows = _make_detail_rows(3)
    schemes = [(0.5, 0.3, 0.2), (0.4, 0.4, 0.2), (0.6, 0.2, 0.2)]
    summary = _summary_rows(rows, schemes)
    s = summary[0]
    for w_cov, w_step, w_tau in schemes:
        key = f"Phi_{w_cov}_{w_step}_{w_tau}_mean"
        assert key in s and s[key] is not None


def test_summary_rows_handles_none_values():
    rows = [
        {"model_id": "m1", "chunker": "dsc", "prompter": "P3", "doc_id": "d1", "seed": 0,
         "StepF1": None, "AdjacencyF1": None, "Kendall": None,
         "ConstraintCoverage": None, "ConstraintAttachmentF1": None, "Phi": None}
    ]
    summary = _summary_rows(rows, [(0.5, 0.3, 0.2)])
    assert len(summary) == 1
    s = summary[0]
    assert s["StepF1_mean"] is None
    assert s["StepF1_n"] == 0


def test_phi_function():
    assert abs(compute_phi(0.6, 0.7, 0.8) - (0.5*0.6 + 0.3*0.7 + 0.2*0.8)) < 1e-9
    # None treated as 0
    assert compute_phi(None, 0.7, None) == pytest.approx(0.3 * 0.7)


def test_bootstrap_detects_real_difference():
    # A clearly dominates B — p-value should be small
    a = [0.9, 0.85, 0.92, 0.88, 0.91]
    b = [0.3, 0.28, 0.31, 0.29, 0.30]
    pval = paired_bootstrap_pvalue(a, b, n_resamples=5000, rng_seed=42)
    assert pval < 0.05


def test_bootstrap_null_gives_high_pvalue():
    # Identical scores — p-value should be high (not significant)
    vals = [0.7, 0.65, 0.72, 0.68, 0.71]
    pval = paired_bootstrap_pvalue(vals, vals, n_resamples=5000, rng_seed=42)
    assert pval > 0.3


def test_bootstrap_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        paired_bootstrap_pvalue([0.5, 0.6], [0.5])


def test_eval_multiseed_rejects_unreviewed_gold(tmp_path):
    """Guard blocks full sweep when gold has review_status != 'reviewed'."""
    from scripts.eval_multiseed import main

    gold_dir = tmp_path / "gold"
    text_dir = tmp_path / "text"
    gold_dir.mkdir()
    text_dir.mkdir()

    # Unreviewed gold file
    (gold_dir / "doc1.json").write_text(json.dumps({
        "procedure": {"doc_id": "doc1", "title": "t"},
        "steps": [{"id": "S1", "label": "step"}],
        "quality": {"review_status": "unreviewed"},
    }))
    (text_dir / "doc1.txt").write_text("step content")

    rc = main([
        "--gold-dir", str(gold_dir),
        "--text-dir", str(text_dir),
        "--seeds", "1",
    ])
    assert rc == 1


def test_eval_multiseed_rejects_reviewed_but_unsigned_gold(
    tmp_path, monkeypatch
):
    import scripts.eval_multiseed as em

    async def _fake_extract(text_path, doc_id, seed):
        return {"steps": [], "constraints": []}

    monkeypatch.setattr(em, "_extract_one", _fake_extract)
    import src.evaluation.metrics as metrics_mod
    monkeypatch.setattr(
        metrics_mod, "prepare_evaluator", lambda *args, **kwargs: (None, None)
    )

    gold_dir = tmp_path / "gold"
    text_dir = tmp_path / "text"
    gold_dir.mkdir()
    text_dir.mkdir()
    (gold_dir / "doc1.json").write_text(json.dumps({
        "procedure": {"doc_id": "doc1", "title": "t"},
        "steps": [{"id": "S1", "label": "step"}],
        "quality": {
            "review_status": "reviewed",
            "annotator": "agent-adjudicated (pending human sign-off)",
            "review_date": "2026-07-10",
        },
    }))
    (text_dir / "doc1.txt").write_text("step content")

    rc = em.main([
        "--gold-dir", str(gold_dir),
        "--text-dir", str(text_dir),
        "--seeds", "1",
    ])

    assert rc == 1


def test_eval_multiseed_dry_run_skips_guard(tmp_path):
    """--dry-run bypasses the unreviewed guard."""
    from scripts.eval_multiseed import main

    gold_dir = tmp_path / "gold"
    text_dir = tmp_path / "text"
    gold_dir.mkdir()
    text_dir.mkdir()

    (gold_dir / "doc1.json").write_text(json.dumps({
        "procedure": {"doc_id": "doc1", "title": "t"},
        "steps": [{"id": "S1", "label": "step"}],
        "quality": {"review_status": "unreviewed"},
    }))
    (text_dir / "doc1.txt").write_text("step content")

    rc = main([
        "--gold-dir", str(gold_dir),
        "--text-dir", str(text_dir),
        "--seeds", "1",
        "--dry-run",
    ])
    assert rc == 0


def test_eval_multiseed_allow_unverified_bypasses_guard(tmp_path, monkeypatch, capsys):
    """--allow-unverified must bypass the guard; test must not load real models."""
    import scripts.eval_multiseed as em
    from scripts.eval_multiseed import main

    # Stub heavy infrastructure so the test stays fast and model-free.
    async def _fake_extract(text_path, doc_id, seed):
        return {"steps": [], "constraints": []}

    monkeypatch.setattr(em, "_extract_one", _fake_extract)
    import src.evaluation.metrics as metrics_mod
    monkeypatch.setattr(metrics_mod, "prepare_evaluator", lambda *a, **kw: (None, None))

    gold_dir = tmp_path / "gold"
    text_dir = tmp_path / "text"
    gold_dir.mkdir()
    text_dir.mkdir()

    (gold_dir / "doc1.json").write_text(json.dumps({
        "procedure": {"doc_id": "doc1", "title": "t"},
        "steps": [{"id": "S1", "label": "step"}],
        "quality": {"review_status": "unreviewed"},
    }))
    (text_dir / "doc1.txt").write_text("step content")

    # No --dry-run: guard must not fire even though gold is unreviewed.
    main([
        "--gold-dir", str(gold_dir),
        "--text-dir", str(text_dir),
        "--seeds", "1",
        "--allow-unverified",
    ])
    captured = capsys.readouterr()
    assert "not reviewed" not in captured.err, (
        "--allow-unverified must bypass the unverified-gold guard"
    )
    assert "quality.review_status != 'reviewed'" not in captured.err


def test_eval_multiseed_allow_unreviewed_alias_sets_allow_unverified():
    from scripts.eval_multiseed import parse_args

    args = parse_args([
        "--gold-dir", "gold",
        "--text-dir", "text",
        "--allow-unreviewed",
    ])

    assert args.allow_unverified is True


def test_eval_multiseed_rejects_malformed_gold(tmp_path):
    """Malformed gold JSON must block the sweep (fail closed), not silently pass."""
    from scripts.eval_multiseed import main

    gold_dir = tmp_path / "gold"
    text_dir = tmp_path / "text"
    gold_dir.mkdir()
    text_dir.mkdir()

    # Truncated JSON — parse will fail
    (gold_dir / "bad.json").write_text("{")
    (text_dir / "bad.txt").write_text("some procedure text")

    rc = main([
        "--gold-dir", str(gold_dir),
        "--text-dir", str(text_dir),
        "--seeds", "1",
    ])
    assert rc == 1


def test_eval_multiseed_dry_run_rejects_malformed_gold(tmp_path):
    """--dry-run must still exit 1 when a gold file is malformed JSON.

    Malformed files are a correctness problem independent of review status;
    --dry-run must not bypass JSON validation.
    """
    from scripts.eval_multiseed import main

    gold_dir = tmp_path / "gold"
    text_dir = tmp_path / "text"
    gold_dir.mkdir()
    text_dir.mkdir()

    (gold_dir / "doc1.json").write_text("{not valid json")
    (text_dir / "doc1.txt").write_text("step content")

    rc = main([
        "--gold-dir", str(gold_dir),
        "--text-dir", str(text_dir),
        "--seeds", "1",
        "--dry-run",
    ])
    assert rc == 1, "--dry-run must exit 1 for malformed gold JSON"


def test_eval_multiseed_allow_unreviewed_still_rejects_malformed_gold(tmp_path):
    """--allow-unreviewed must not bypass malformed-JSON detection.

    The flag relaxes the human-review requirement only; it must not suppress
    failures caused by unreadable or syntactically broken gold files.
    """
    from scripts.eval_multiseed import main

    gold_dir = tmp_path / "gold"
    text_dir = tmp_path / "text"
    gold_dir.mkdir()
    text_dir.mkdir()

    (gold_dir / "doc1.json").write_text("{bad json")
    (text_dir / "doc1.txt").write_text("step content")

    rc = main([
        "--gold-dir", str(gold_dir),
        "--text-dir", str(text_dir),
        "--seeds", "1",
        "--allow-unreviewed",
    ])
    assert rc == 1, "--allow-unreviewed must not suppress malformed-gold detection"


def test_makefile_eval_validate_uses_strict_mode():
    text = Path("Makefile").read_text(encoding="utf-8")
    assert "scripts/validate_paper_gold.py --gold-dir $(PAPER_GOLD) --strict" in text


def test_makefile_has_explicit_paper_evidence_gate():
    text = Path("Makefile").read_text(encoding="utf-8")
    assert "PAPER_MANIFEST := datasets/paper/corpus_manifest.json" in text
    assert "eval-paper-gate:" in text
    assert (
        "scripts/validate_paper_gold.py --gold-dir $(PAPER_GOLD) \\\n"
        "\t\t--manifest $(PAPER_MANIFEST) --require-frozen-manifest \\\n"
        "\t\t--strict --require-human-verified"
    ) in text


def test_makefile_eval_full_requires_paper_evidence_gate():
    text = Path("Makefile").read_text(encoding="utf-8")
    assert "eval-full: eval-paper-gate" in text


def test_makefile_eval_iaa_depends_on_validation():
    text = Path("Makefile").read_text(encoding="utf-8")
    assert "eval-iaa: eval-validate" in text


def test_dataset_readme_uses_existing_validator_command():
    text = Path("datasets/paper/README.md").read_text(encoding="utf-8")
    assert "scripts/validate_gold.py" not in text
    assert "make eval-validate" in text
    assert "make eval-paper-gate" in text
