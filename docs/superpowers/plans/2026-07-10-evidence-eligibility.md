# IPKE Evidence Eligibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent unsigned agent-reviewed gold annotations from entering IPKE paper-evidence runs while preserving an explicit development override.

**Architecture:** Add one small evidence-metadata module under `src/evaluation` and make both the paper validator and multi-seed runner consume it. Structural validation remains the validator's responsibility in this slice; the new module owns only the distinction between declared review and actual human verification. The CLI keeps the old `--allow-unreviewed` spelling as a compatibility alias while documenting the accurate `--allow-unverified` name.

**Tech Stack:** Python 3.12, frozen dataclasses, regular expressions, argparse, pytest, Make.

## Global Constraints

- Agent review must never be represented as human verification.
- `quality.review_status = "reviewed"` alone is insufficient for paper evidence.
- A human-verified annotation requires a non-empty `+ human-verified:<handle>` marker and must not retain `(pending human sign-off)`.
- Dry runs may plan unsigned data but must still reject malformed JSON.
- Full runs fail closed unless `--allow-unverified` is passed explicitly for development.
- Preserve `--allow-unreviewed` as a backward-compatible alias.
- Do not change gold annotations or generated result files in this plan.
- Follow one-test-at-a-time red-green-refactor and run GitNexus impact before editing existing symbols.

---

### Task 1: Evidence metadata interface

**Files:**
- Create: `src/evaluation/evidence.py`
- Create: `tests/evaluation/test_evidence.py`

**Interfaces:**
- Consumes: annotation mappings with an optional `quality` mapping.
- Produces: `AnnotationEvidence` and `assess_annotation_evidence(annotation)`.

- [ ] **Step 1: Write the failing classification behavior test**

Create `tests/evaluation/test_evidence.py` with:

```python
from __future__ import annotations

import pytest

from src.evaluation.evidence import assess_annotation_evidence


def _annotation(annotator: str, *, status: str = "reviewed") -> dict:
    return {
        "quality": {
            "review_status": status,
            "annotator": annotator,
            "review_date": "2026-07-10",
        }
    }


@pytest.mark.parametrize(
    ("annotation", "expected"),
    [
        (
            _annotation(
                "model-assisted:qwen + agent-adjudicated (pending human sign-off)"
            ),
            (
                True,
                False,
                False,
                ("quality.annotator still contains pending human sign-off",),
            ),
        ),
        (
            _annotation(
                "model-assisted:qwen + agent-adjudicated + human-verified:imad"
            ),
            (True, True, True, ()),
        ),
        (
            _annotation("agent-adjudicated"),
            (
                True,
                False,
                False,
                ("quality.annotator lacks a + human-verified:<handle> marker",),
            ),
        ),
        (
            {},
            (
                False,
                False,
                False,
                (
                    "quality.review_status must be 'reviewed'",
                    "quality.annotator missing",
                    "quality.review_date missing",
                ),
            ),
        ),
    ],
)
def test_assess_annotation_evidence_classifies_review_provenance(
    annotation: dict, expected: tuple[bool, bool, bool, tuple[str, ...]]
) -> None:
    result = assess_annotation_evidence(annotation)

    assert (
        result.declared_reviewed,
        result.human_verified,
        result.evidence_eligible,
        result.issues,
    ) == expected
```

- [ ] **Step 2: Run the test and watch it fail**

Run: `uv run pytest tests/evaluation/test_evidence.py::test_assess_annotation_evidence_classifies_review_provenance -q`

Expected: FAIL during collection with `ModuleNotFoundError: No module named 'src.evaluation.evidence'`.

- [ ] **Step 3: Implement the minimal public interface**

Create `src/evaluation/evidence.py` with:

```python
"""Evidence eligibility metadata for research annotations."""
from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

PENDING_HUMAN_SIGN_OFF = "(pending human sign-off)"
HUMAN_VERIFIED_RE = re.compile(
    r"(?:^|\s)\+\s*human-verified:([A-Za-z0-9][A-Za-z0-9._-]*)"
)


@dataclass(frozen=True, slots=True)
class AnnotationEvidence:
    declared_reviewed: bool
    human_verified: bool
    evidence_eligible: bool
    issues: tuple[str, ...]


def assess_annotation_evidence(annotation: Mapping[str, Any]) -> AnnotationEvidence:
    raw_quality = annotation.get("quality")
    quality = raw_quality if isinstance(raw_quality, Mapping) else {}
    declared_reviewed = quality.get("review_status") == "reviewed"
    raw_annotator = quality.get("annotator")
    annotator = raw_annotator.strip() if isinstance(raw_annotator, str) else ""
    review_date = quality.get("review_date")
    pending = PENDING_HUMAN_SIGN_OFF in annotator.lower()
    human_verified = bool(HUMAN_VERIFIED_RE.search(annotator)) and not pending

    issues: list[str] = []
    if not declared_reviewed:
        issues.append("quality.review_status must be 'reviewed'")
    if not annotator:
        issues.append("quality.annotator missing")
    elif pending:
        issues.append("quality.annotator still contains pending human sign-off")
    elif not human_verified:
        issues.append("quality.annotator lacks a + human-verified:<handle> marker")
    if not review_date:
        issues.append("quality.review_date missing")

    return AnnotationEvidence(
        declared_reviewed=declared_reviewed,
        human_verified=human_verified,
        evidence_eligible=not issues,
        issues=tuple(issues),
    )
```

- [ ] **Step 4: Run the first test and watch it pass**

Run: `uv run pytest tests/evaluation/test_evidence.py::test_assess_annotation_evidence_classifies_review_provenance -q`

Expected: `4 passed` because pytest reports one pass for each classified case.

- [ ] **Step 5: Commit**

```bash
git add src/evaluation/evidence.py tests/evaluation/test_evidence.py
git commit -m "Add annotation evidence eligibility"
```

### Task 2: Paper validator human-verification mode

**Files:**
- Modify: `scripts/validate_paper_gold.py`
- Modify: `tests/test_validate_paper_gold.py`

**Interfaces:**
- Consumes: `assess_annotation_evidence` from Task 1.
- Produces: `validate_file(path, require_human_verified=False)` and CLI `--require-human-verified`.

- [ ] **Step 1: Run GitNexus impact before editing**

Run impact for `Function:scripts/validate_paper_gold.py:validate_file` in the upstream direction, including tests. Read every depth-1 caller before editing. Stop and report if risk is HIGH or CRITICAL.

- [ ] **Step 2: Write the failing validator test**

Append to `tests/test_validate_paper_gold.py`:

```python
def test_human_verified_mode_rejects_agent_review(tmp_path: Path) -> None:
    annotation = json.loads(json.dumps(VALID_ANNOTATION))
    annotation["quality"]["annotator"] = (
        "model-assisted:qwen + agent-adjudicated (pending human sign-off)"
    )

    messages = validate_file(
        _write(tmp_path, annotation), require_human_verified=True
    )

    assert any("pending human sign-off" in error for error in _errors(messages))
```

- [ ] **Step 3: Run the test and watch it fail**

Run: `uv run pytest tests/test_validate_paper_gold.py::test_human_verified_mode_rejects_agent_review -q`

Expected: FAIL with `TypeError: validate_file() got an unexpected keyword argument 'require_human_verified'`.

- [ ] **Step 4: Implement the validator option**

Import `assess_annotation_evidence`. Change the public signature to:

```python
def validate_file(path: Path, *, require_human_verified: bool = False) -> list[str]:
```

After the existing review metadata checks, add:

```python
    if require_human_verified:
        evidence = assess_annotation_evidence(d)
        if not evidence.human_verified:
            errors.extend(
                issue
                for issue in evidence.issues
                if "human" in issue or "pending human sign-off" in issue
            )
```

Add the parser flag:

```python
    ap.add_argument(
        "--require-human-verified",
        action="store_true",
        help="Require a non-pending + human-verified:<handle> marker.",
    )
```

Pass it from `main()`:

```python
        msgs = validate_file(
            f, require_human_verified=args.require_human_verified
        )
```

- [ ] **Step 5: Run the validator tests**

Run: `uv run pytest tests/test_validate_paper_gold.py -q`

Expected: all tests pass.

- [ ] **Step 6: Add and prove the signed behavior**

Append:

```python
def test_human_verified_mode_accepts_signed_annotation(tmp_path: Path) -> None:
    annotation = json.loads(json.dumps(VALID_ANNOTATION))
    annotation["quality"]["annotator"] = (
        "model-assisted:qwen + agent-adjudicated + human-verified:imad"
    )

    messages = validate_file(
        _write(tmp_path, annotation), require_human_verified=True
    )

    assert _errors(messages) == []
```

Run the new test before implementation to observe failure, then add the signed-marker
handling shown in Task 1 and rerun `uv run pytest tests/test_validate_paper_gold.py -q`.

- [ ] **Step 7: Commit**

```bash
git add scripts/validate_paper_gold.py tests/test_validate_paper_gold.py
git commit -m "Require explicit human gold verification"
```

### Task 3: Multi-seed paper-evidence guard

**Files:**
- Modify: `scripts/eval_multiseed.py`
- Modify: `tests/test_new_tools.py`

**Interfaces:**
- Consumes: `assess_annotation_evidence` from Task 1.
- Produces: default full-run failure for unsigned annotations and `--allow-unverified` with legacy alias `--allow-unreviewed`.

- [ ] **Step 1: Run GitNexus impact before editing**

Run upstream impact for `Function:scripts/eval_multiseed.py:main` and its argument parser, including tests. Read all depth-1 callers. Stop and report if risk is HIGH or CRITICAL.

- [ ] **Step 2: Write the failing reviewed-but-unsigned behavior test**

Append to `tests/test_new_tools.py`:

```python
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
```

- [ ] **Step 3: Run the test and watch it fail**

Run: `uv run pytest tests/test_new_tools.py::test_eval_multiseed_rejects_reviewed_but_unsigned_gold -q`

Expected: the old guard accepts `review_status="reviewed"` and proceeds toward extraction instead of returning the expected eligibility error.

- [ ] **Step 4: Replace the guard with evidence assessment**

Import `assess_annotation_evidence`. Replace the parser option with:

```python
    parser.add_argument(
        "--allow-unverified",
        "--allow-unreviewed",
        dest="allow_unverified",
        action="store_true",
        help=(
            "Allow gold without explicit human verification. Development only; "
            "results cannot be used as paper evidence."
        ),
    )
```

Replace the non-dry-run review-status block with:

```python
    if not args.dry_run and not args.allow_unverified:
        ineligible: dict[str, tuple[str, ...]] = {}
        for gf, _ in pairs:
            evidence = assess_annotation_evidence(loaded_gold[gf.stem])
            if not evidence.evidence_eligible:
                ineligible[gf.stem] = evidence.issues
        if ineligible:
            print(
                "ERROR: gold files are not eligible for paper evidence:",
                file=sys.stderr,
            )
            for name, issues in ineligible.items():
                print(f"  {name}: {'; '.join(issues)}", file=sys.stderr)
            print(
                "Complete human verification, or pass --allow-unverified "
                "for development use only.",
                file=sys.stderr,
            )
            return 1
```

- [ ] **Step 5: Run the targeted runner tests**

Run:

```bash
uv run pytest \
  tests/test_new_tools.py::test_eval_multiseed_rejects_unreviewed_gold \
  tests/test_new_tools.py::test_eval_multiseed_rejects_reviewed_but_unsigned_gold \
  tests/test_new_tools.py::test_eval_multiseed_dry_run_skips_guard \
  tests/test_new_tools.py::test_eval_multiseed_allow_unreviewed_bypasses_guard \
  tests/test_eval_multiseed.py -q
```

Expected: all selected tests pass. The legacy alias test remains green.

- [ ] **Step 6: Add the accurate override spelling test**

Rename the existing development-override test to use `--allow-unverified`, then duplicate
one small parse-level assertion using `--allow-unreviewed` to preserve compatibility.
Run the targeted command from Step 5 again and expect all tests to pass.

- [ ] **Step 7: Commit**

```bash
git add scripts/eval_multiseed.py tests/test_new_tools.py tests/test_eval_multiseed.py
git commit -m "Block paper runs on unsigned gold"
```

### Task 4: Explicit paper-evidence command and documentation

**Files:**
- Modify: `Makefile`
- Modify: `REPRODUCIBILITY.md`
- Modify: `docs/annotation/SIGN_OFF_ISSUE.md`
- Modify: `tests/test_new_tools.py`

**Interfaces:**
- Consumes: validator CLI from Task 2.
- Produces: `make eval-paper-gate`, an intentionally failing command until human sign-off is complete.

- [ ] **Step 1: Write the failing Make contract test**

Append to `tests/test_new_tools.py`:

```python
def test_makefile_has_explicit_paper_evidence_gate():
    text = Path("Makefile").read_text(encoding="utf-8")
    assert "eval-paper-gate:" in text
    assert (
        "scripts/validate_paper_gold.py --gold-dir $(PAPER_GOLD) "
        "--strict --require-human-verified"
    ) in text
```

- [ ] **Step 2: Run the test and watch it fail**

Run: `uv run pytest tests/test_new_tools.py::test_makefile_has_explicit_paper_evidence_gate -q`

Expected: FAIL because `eval-paper-gate` does not exist.

- [ ] **Step 3: Add the Make target**

Add `eval-paper-gate` to `.PHONY` and define:

```make
eval-paper-gate:
	$(PYTHON) scripts/validate_paper_gold.py --gold-dir $(PAPER_GOLD) --strict --require-human-verified
```

Do not add it to default unit-test or development dry-run targets while the corpus is
unsigned. Paper-result commands must depend on it when the canonical experiment runner
is introduced.

- [ ] **Step 4: Run the Make contract test and target**

Run: `uv run pytest tests/test_new_tools.py::test_makefile_has_explicit_paper_evidence_gate -q`

Expected: `1 passed`.

Run: `make eval-paper-gate`

Expected now: non-zero with all unsigned gold files named. This expected failure proves
the gate is active; it is not a test-suite failure.

- [ ] **Step 5: Correct the documentation**

In `REPRODUCIBILITY.md`, distinguish:

```markdown
- `make eval-validate`: structural and annotation-contract validation.
- `make eval-paper-gate`: structural validation plus explicit human verification; this
  intentionally fails until the human sign-off issue is complete.
```

In `docs/annotation/SIGN_OFF_ISSUE.md`, replace any statement that strict validation alone
makes gold safe for paper evidence with `make eval-paper-gate` as the release criterion.
Do not change the human-only sign-off instructions.

- [ ] **Step 6: Run focused and full tests**

Run:

```bash
uv run pytest tests/evaluation/test_evidence.py tests/test_validate_paper_gold.py tests/test_eval_multiseed.py tests/test_new_tools.py -q
uv run pytest -q
```

Expected: all tests pass. Record the exact totals.

- [ ] **Step 7: Detect change impact and commit**

Run GitNexus change detection over staged changes and confirm only evidence validation,
the multi-seed guard, Make, and documentation are affected.

```bash
git add Makefile REPRODUCIBILITY.md docs/annotation/SIGN_OFF_ISSUE.md tests/test_new_tools.py
git commit -m "Add paper evidence release gate"
```

## Self-review

- Spec coverage: the plan separates declared review, human verification, development
  override, validator mode, runner behavior, and an explicit release command.
- Placeholder scan: every code-changing step contains the concrete implementation or
  exact documentation text it requires.
- Type consistency: every task uses `AnnotationEvidence`,
  `assess_annotation_evidence`, `require_human_verified`, and `allow_unverified` with the
  same spelling and semantics.
