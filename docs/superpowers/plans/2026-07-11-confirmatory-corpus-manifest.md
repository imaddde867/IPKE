# Confirmatory Corpus Manifest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Make one typed manifest control which paper-corpus files validation and evaluation consume, while preserving excluded historical annotations in place.

**Architecture:** A small Pydantic v2 module validates manifest metadata and exact classification of a gold directory. The validator and multi-seed runner both call it before reading selected annotations. Make targets pass the same manifest; the paper gate additionally requires a frozen manifest and human verification.

**Tech Stack:** Python 3.12, Pydantic v2, argparse, JSON, pytest, Make.

## Global Constraints

- Do not move, delete, or rewrite annotation JSON files.
- Every JSON in the selected gold directory must be classified exactly once.
- Excluded annotations must never reach paper-evidence validation or model execution.
- The current five included entries remain candidates, not paper evidence.
- A paper gate requires a frozen manifest and explicit human verification.
- Development dry runs may use a provisional manifest and must state that status.
- Follow one-test-at-a-time red-green-refactor.
- Run GitNexus impact before editing existing functions.

---

### Task 1: Typed manifest boundary and current classification

**Files:**
- Create: src/evaluation/corpus_manifest.py
- Create: tests/evaluation/test_corpus_manifest.py
- Create: datasets/paper/corpus_manifest.json

**Interfaces:**
- Produces CorpusDocument, CorpusManifest, load_corpus_manifest(path), and select_manifest_gold_files(manifest, gold_dir).

- [ ] **Step 1: Write the failing selection test**

~~~python
def test_manifest_selects_only_included_gold(tmp_path: Path) -> None:
    gold_dir = tmp_path / "gold"
    gold_dir.mkdir()
    (gold_dir / "include.json").write_text("{}", encoding="utf-8")
    (gold_dir / "exclude.json").write_text("{}", encoding="utf-8")
    path = _write_manifest(
        tmp_path / "manifest.json",
        [
            _entry("include", include=True),
            _entry(
                "exclude",
                include=False,
                role="requirements_stress_test",
                status="excluded_wrong_genre",
            ),
        ],
    )

    manifest = load_corpus_manifest(path)
    selected = select_manifest_gold_files(manifest, gold_dir)

    assert [item.stem for item in selected] == ["include"]
~~~

Run: uv run pytest tests/evaluation/test_corpus_manifest.py::test_manifest_selects_only_included_gold -q

Expected: collection fails because src.evaluation.corpus_manifest does not exist.

- [ ] **Step 2: Implement the minimal typed boundary**

~~~python
"""Typed membership contract for the IPKE evaluation corpus."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CorpusDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    doc_id: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9_]*$")
    source_family: str = Field(min_length=1)
    role: Literal["procedure_candidate", "requirements_stress_test"]
    status: Literal[
        "candidate",
        "excluded_wrong_genre",
        "excluded_pending_reannotation",
    ]
    include_for_evaluation: bool
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_classification(self) -> CorpusDocument:
        if self.include_for_evaluation:
            if self.role != "procedure_candidate" or self.status != "candidate":
                raise ValueError(
                    "included documents must be procedure candidates with candidate status"
                )
        elif self.status == "candidate":
            raise ValueError("excluded documents must use an excluded status")
        return self


class CorpusManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    manifest_status: Literal["provisional", "frozen"]
    documents: tuple[CorpusDocument, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_membership(self) -> CorpusManifest:
        ids = [document.doc_id for document in self.documents]
        if len(ids) != len(set(ids)):
            raise ValueError("manifest document IDs must be unique")
        if not any(document.include_for_evaluation for document in self.documents):
            raise ValueError("manifest must include at least one document")
        return self

    @property
    def included_doc_ids(self) -> tuple[str, ...]:
        return tuple(
            document.doc_id
            for document in self.documents
            if document.include_for_evaluation
        )


def load_corpus_manifest(path: Path) -> CorpusManifest:
    return CorpusManifest.model_validate_json(path.read_text(encoding="utf-8"))


def select_manifest_gold_files(
    manifest: CorpusManifest,
    gold_dir: Path,
) -> tuple[Path, ...]:
    actual = {path.stem: path for path in sorted(gold_dir.glob("*.json"))}
    declared = {document.doc_id for document in manifest.documents}
    missing = declared - actual.keys()
    unclassified = actual.keys() - declared
    if missing or unclassified:
        details: list[str] = []
        if missing:
            details.append(f"missing files: {', '.join(sorted(missing))}")
        if unclassified:
            details.append(f"unclassified files: {', '.join(sorted(unclassified))}")
        raise ValueError(
            "manifest does not match gold directory; " + "; ".join(details)
        )
    return tuple(actual[doc_id] for doc_id in manifest.included_doc_ids)
~~~

Run the first test. Expected: 1 passed.

- [ ] **Step 3: Add invalid-contract tests one at a time**

Prove duplicate IDs, inconsistent included status, empty inclusion, missing files, and unclassified files each fail. Run every new test red before the smallest implementation adjustment.

- [ ] **Step 4: Add the repository manifest**

Classify all eight current stems. Include the three EPA and two USGS files. Classify NASA as requirements_stress_test / excluded_wrong_genre. Classify OLSK and NIOSH as procedure_candidate / excluded_pending_reannotation. Keep manifest_status provisional.

Add a repository test:

~~~python
def test_repository_manifest_classifies_legacy_candidates() -> None:
    manifest = load_corpus_manifest(Path("datasets/paper/corpus_manifest.json"))
    selected = select_manifest_gold_files(manifest, Path("datasets/paper/gold"))
    by_id = {document.doc_id: document for document in manifest.documents}

    assert len(manifest.documents) == 8
    assert len(selected) == 5
    assert by_id["nasa_npr_8715_3d_general_safety"].role == "requirements_stress_test"
    assert by_id["olsk_small_cnc_v1_workbook"].status == "excluded_pending_reannotation"
    assert by_id["niosh_nmam_5th_edition_ebook"].status == "excluded_pending_reannotation"
~~~

Run: uv run pytest tests/evaluation/test_corpus_manifest.py -q

- [ ] **Step 5: Detect scope and commit**

Run GitNexus staged change detection. Commit only the module, tests, and JSON manifest with title Add confirmatory corpus manifest.

### Task 2: Manifest-scoped paper validator

**Files:**
- Modify: scripts/validate_paper_gold.py
- Modify: tests/test_validate_paper_gold.py
- Modify: Makefile
- Modify: tests/test_new_tools.py

**Interfaces:**
- Produces validator flags --manifest and --require-frozen-manifest.

- [ ] **Step 1: Run GitNexus impact**

Run upstream impact for the exact validator main symbol, including tests. Read every depth-1 caller. Warn before editing on HIGH or CRITICAL risk.

- [ ] **Step 2: Write the failing CLI test**

Write one signed included annotation and one malformed excluded annotation, plus a frozen manifest. Call:

~~~python
rc = main([
    "--gold-dir", str(gold_dir),
    "--manifest", str(manifest_path),
    "--strict",
    "--require-frozen-manifest",
    "--require-human-verified",
])
assert rc == 0
~~~

Expected red: main does not accept argv or manifest flags.

- [ ] **Step 3: Implement CLI selection**

Change the signature to main(argv: list[str] | None = None) -> int and parse argv. Add Path flags --manifest and --require-frozen-manifest. Load the manifest and select files before validation. Return 1 on parse, schema, directory-classification, or missing-manifest errors. If frozen status is required but manifest_status is provisional, print a clear error, continue validating the selected files for complete diagnostics, and return non-zero.

- [ ] **Step 4: Add provisional fail-closed test**

With signed included data and a provisional manifest, assert return 1 and stderr/stdout contains manifest is provisional. Run red then green.

- [ ] **Step 5: Wire Make**

Add PAPER_MANIFEST := datasets/paper/corpus_manifest.json. Change eval-paper-gate to pass:

~~~make
--manifest $(PAPER_MANIFEST) --require-frozen-manifest \
--strict --require-human-verified
~~~

Update the Make contract test red then green.

- [ ] **Step 6: Verify and commit**

Run:

~~~bash
uv run pytest tests/evaluation/test_corpus_manifest.py tests/test_validate_paper_gold.py tests/test_new_tools.py -q
make eval-paper-gate
~~~

Tests pass. The Make target fails on provisional status and five unsigned included files, without treating NASA, OLSK, or NIOSH as evidence failures.

Run GitNexus staged detection. Commit only Task 2 files with title Scope paper gate to corpus manifest.

### Task 3: Manifest-scoped multi-seed evaluation

**Files:**
- Modify: scripts/eval_multiseed.py
- Modify: tests/test_new_tools.py
- Modify: tests/test_eval_multiseed.py
- Modify: Makefile
- Modify: REPRODUCIBILITY.md
- Modify: datasets/paper/README.md

**Interfaces:**
- Produces runner flag --manifest PATH and shared Make selection.

- [ ] **Step 1: Run GitNexus impact**

Run upstream impact for exact parse_args and main symbols in scripts/eval_multiseed.py, including tests. Read every depth-1 caller and warn before editing on HIGH or CRITICAL risk.

- [ ] **Step 2: Write the failing excluded-malformed test**

Create included good.json and excluded malformed bad.json, matching text files, and a provisional manifest. Call the runner with --manifest and --dry-run. Assert return 0 and output Plan: 1 docs. Expected red: argparse rejects --manifest.

- [ ] **Step 3: Implement selection before annotation parsing**

Add parser.add_argument("--manifest", type=Path). When supplied, load and select through Task 1 before building pairs. Catch OSError, ValueError, and Pydantic ValidationError, print ERROR: invalid corpus manifest, and return 1 before config/model work. Print a development-only warning when manifest_status is provisional. Keep malformed-JSON failure for every selected file; never load excluded files.

- [ ] **Step 4: Add manifest failure tests one at a time**

Prove an unclassified gold and a missing declared gold return 1 before config/model work. Run each test red then green.

- [ ] **Step 5: Wire evaluation targets**

Add --manifest $(PAPER_MANIFEST) to eval and eval-full. Extend the Make contract test to assert both target bodies contain it. Keep eval-full dependent on eval-paper-gate.

- [ ] **Step 6: Update durable documentation**

Document that eval-validate checks all legacy files structurally, while the paper gate and evaluation targets consume corpus_manifest.json. Record provisional status, five included candidates, three exclusions, and frozen-plus-human requirements.

- [ ] **Step 7: Verify and commit**

Run:

~~~bash
uv run pytest tests/evaluation/test_corpus_manifest.py tests/evaluation/test_evidence.py tests/test_validate_paper_gold.py tests/test_eval_multiseed.py tests/test_new_tools.py -q
uv run pytest -q
make eval
make eval-paper-gate
~~~

Expected: tests and make eval pass. The paper gate intentionally fails on provisional status and five unsigned included files, not the three excluded files.

Run GitNexus staged detection. Commit only Task 3 files with title Use corpus manifest for evaluation.

## Self-review

- Spec coverage: typed classification, exact directory matching, frozen paper gate, validator selection, runner selection, Make wiring, exclusions, and docs each have a task.
- Placeholder scan: no TODO, TBD, or unspecified behavior remains.
- Type consistency: both consumers use CorpusManifest, load_corpus_manifest, and select_manifest_gold_files with the Task 1 signatures.

