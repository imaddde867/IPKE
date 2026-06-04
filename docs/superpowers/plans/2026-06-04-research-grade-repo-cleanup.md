# Research Grade Repo Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn IPKE into a clean, minimal, reproducible research repository that supports the thesis results and the ECIR 2027 paper push.

**Architecture:** Keep the existing extraction, chunking, graph, and evaluation code intact unless a specific reproducibility defect requires a narrow change. Add a small reproducibility surface around it: consistent run directories, one-command targets, metadata capture, paper dataset layout, and documentation that states what the code actually does. Split packaging concerns so the research core is installable without the app/demo stack.

**Tech Stack:** Python 3.12, uv, pytest, ruff-compatible Python style, Makefile, Pydantic v2, SentenceTransformers, spaCy, llama-cpp-python, GitNexus for impact checks before symbol edits.

---

## Scope Check

This plan addresses the repo-readiness findings from the 2026-06-04 audit:

- branch hygiene and generated/untracked file policy
- duplicated pytest configuration
- `dsc` CLI alias mismatch in `scripts/run_pkg_extraction.py`
- inconsistent `logs/` and `results/` output roots
- missing one-command `make` entry points
- missing paper dataset directory scaffold
- incomplete run metadata capture
- heavy dependency set mixed into base install
- thesis/paper wording risk around DSC implementation

This plan does not run new LLM experiments or mutate gold annotations. Those should be separate experiment plans.

## File Structure

- Modify: `.gitignore`
  - Stop ignoring `AGENTS.md`.
  - Keep generated data, logs, model files, and local PDFs out of normal commits.
- Modify: `pyproject.toml`
  - Keep only core runtime dependencies in base.
  - Move app, LLM, document extras, Neo4j, and dev tools into optional extras.
  - Remove duplicate pytest config after preserving it in `pytest.ini`.
- Modify: `pytest.ini`
  - Keep the canonical pytest configuration.
- Modify: `scripts/run_pkg_extraction.py`
  - Accept `dsc` as a documented alias.
  - Default outputs to `runs/pkg_extraction/<doc_id>`.
  - Write run metadata next to prediction outputs.
- Modify: `scripts/experiments/experiment_utils.py`
  - Change default experiment root from `results/experiments` to `runs/experiments`.
  - Add reusable metadata helpers for reproducibility.
- Modify: `scripts/experiments/run_all_chunking_experiments.py`
  - Change master run root from `results/` to `runs/chunking_sweeps/`.
  - Keep mirrored summaries under `runs/latest/`, not git-ignored ad hoc locations.
- Create: `Makefile`
  - Add `test`, `eval`, `smoke-extract`, `paper-table`, `clean-artifacts`.
- Create: `datasets/paper/text/.gitkeep`
  - Scaffold paper dataset text intake location.
- Create: `datasets/paper/gold/.gitkeep`
  - Scaffold paper Tier A gold location.
- Create: `datasets/paper/README.md`
  - Document intake rules and gold annotation policy.
- Create: `docs/reproducibility.md`
  - Document fresh-clone setup, commands, outputs, metadata, and limitations.
- Create: `docs/methods/dsc-implementation.md`
  - State the implemented DSC algorithm precisely and flag the thesis/paper wording decision.
- Create or modify tests:
  - `tests/test_run_pkg_extraction_cli.py`
  - `tests/test_experiment_paths.py`
  - `tests/test_reproducibility_metadata.py`
  - `tests/test_packaging_config.py`

## Required Git Setup

- [ ] **Step 1: Confirm branch**

Run:

```bash
git status --short --branch
```

Expected:

```text
## chore/research-grade-repo-plan
?? .agents/
?? Elmouss_Imad Eddine.pdf
```

If not on `chore/research-grade-repo-plan`, create or switch to the implementation branch:

```bash
git checkout -b chore/research-grade-repo-cleanup
```

- [ ] **Step 2: Preserve author configuration**

Run:

```bash
git config user.name "Imad"
git config user.email "imad.e.elmouss@turkuamk.fi"
```

Expected: no output.

---

### Task 1: Normalize Git Hygiene

**Files:**
- Modify: `.gitignore`
- Test: shell checks only

- [ ] **Step 1: Inspect current ignore behavior**

Run:

```bash
git check-ignore -v AGENTS.md
git check-ignore -v "Elmouss_Imad Eddine.pdf"
git check-ignore -v runs/example/output.json
```

Expected before change:

```text
.gitignore:...:AGENTS.md	AGENTS.md
```

The thesis PDF and `runs/` may or may not be ignored before this task.

- [ ] **Step 2: Update `.gitignore`**

Edit `.gitignore` so the relevant sections read exactly:

```gitignore
# Logs, Results, and Run Artifacts
logs/
results/
runs/
*.log

# Local research/private documents
*.pdf
!datasets/Samples/*.pdf

# Large model files
models/

# Uploaded files
uploaded_files/*
!uploaded_files/.gitkeep

# Extras
thesis/
PRE_DEPLOYMENT_CHECKLIST.md
```

Remove the `AGENTS.md` line from `.gitignore`.

- [ ] **Step 3: Verify ignore behavior**

Run:

```bash
git check-ignore -v AGENTS.md || true
git check-ignore -v "Elmouss_Imad Eddine.pdf"
git check-ignore -v runs/example/output.json
git status --short
```

Expected:

```text
AGENTS.md is not ignored by the first command; the command exits through `|| true`.
Elmouss_Imad Eddine.pdf is ignored by the `*.pdf` rule.
runs/example/output.json is ignored by the `runs/` rule.
```

`git status --short` should no longer show `Elmouss_Imad Eddine.pdf`.

- [ ] **Step 4: Commit**

Run:

```bash
git add .gitignore
git commit -m "Fix repository ignore rules"
```

Expected: commit succeeds.

---

### Task 2: Consolidate Pytest Configuration

**Files:**
- Modify: `pyproject.toml`
- Keep: `pytest.ini`
- Test: `pytest.ini`, `uv run pytest tests/test_config.py -v`

- [ ] **Step 1: Verify the duplicate config warning**

Run:

```bash
uv run pytest tests/test_config.py -v
```

Expected before change includes:

```text
configfile: pytest.ini (WARNING: ignoring pytest config in pyproject.toml!)
```

- [ ] **Step 2: Remove pytest config from `pyproject.toml`**

Delete this complete block from `pyproject.toml`:

```toml
[tool.pytest.ini_options]
filterwarnings = [
    "ignore:builtin type (SwigPyPacked|SwigPyObject|swigvarlink) has no __module__ attribute:DeprecationWarning",
]
markers = [
    "integration: marks tests that require heavy models (deselect with '-m \"not integration\"')",
]
addopts = "-m \"not integration\" -v"
```

Keep `pytest.ini` unchanged:

```ini
[pytest]
filterwarnings =
    ignore:builtin type (SwigPyPacked|SwigPyObject|swigvarlink) has no __module__ attribute:DeprecationWarning
markers =
    integration: marks tests that require heavy models (deselect with '-m \"not integration\"')
addopts = -m "not integration" -v
```

- [ ] **Step 3: Verify warning is gone**

Run:

```bash
uv run pytest tests/test_config.py -v
```

Expected:

```text
configfile: pytest.ini
```

No warning about ignored pytest config.

- [ ] **Step 4: Commit**

Run:

```bash
git add pyproject.toml pytest.ini
git commit -m "Consolidate pytest configuration"
```

Expected: commit succeeds.

---

### Task 3: Fix `dsc` CLI Alias and Run Output Root

**Files:**
- Modify: `scripts/run_pkg_extraction.py`
- Create: `tests/test_run_pkg_extraction_cli.py`

Before editing `parse_args` or `_apply_env_overrides`, run GitNexus impact:

```bash
# GitNexus MCP equivalent:
# impact({target: "parse_args", direction: "upstream", repo: "IPKE"})
# impact({target: "_apply_env_overrides", direction: "upstream", repo: "IPKE"})
```

If impact is HIGH or CRITICAL, stop and report the blast radius before editing.

- [ ] **Step 1: Write failing CLI tests**

Create `tests/test_run_pkg_extraction_cli.py`:

```python
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
```

- [ ] **Step 2: Run failing tests**

Run:

```bash
uv run pytest tests/test_run_pkg_extraction_cli.py -v
```

Expected before implementation:

```text
FAILED tests/test_run_pkg_extraction_cli.py::test_parse_args_accepts_dsc_alias
```

The default output assertion may also fail because the script defaults to `logs/pkg_runs/...`.

- [ ] **Step 3: Modify `parse_args` to accept argv and alias choices**

In `scripts/run_pkg_extraction.py`, change:

```python
def parse_args() -> argparse.Namespace:
```

to:

```python
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
```

Change the chunking argument to:

```python
    parser.add_argument(
        "--chunking-method",
        default="dsc",
        choices=sorted(CHUNKING_METHOD_CHOICES | {"dsc"}),
        help="Chunking method override (default: dsc).",
    )
```

Change the output default to:

```python
    parser.add_argument(
        "--output-dir",
        default="runs/pkg_extraction/3m_marine_oem_sop",
        help="Directory for saving extraction artifacts.",
    )
```

Change the return line from:

```python
    return parser.parse_args()
```

to:

```python
    return parser.parse_args(argv)
```

- [ ] **Step 4: Run targeted tests**

Run:

```bash
uv run pytest tests/test_run_pkg_extraction_cli.py -v
```

Expected:

```text
3 passed
```

- [ ] **Step 5: Run script help smoke test**

Run:

```bash
uv run python scripts/run_pkg_extraction.py --help
```

Expected includes:

```text
--chunking-method
```

and exits with code 0.

- [ ] **Step 6: Commit**

Run:

```bash
git add scripts/run_pkg_extraction.py tests/test_run_pkg_extraction_cli.py
git commit -m "Fix extraction CLI defaults"
```

Expected: commit succeeds.

---

### Task 4: Add Reproducibility Metadata for Single Extraction

**Files:**
- Modify: `scripts/run_pkg_extraction.py`
- Create: `tests/test_reproducibility_metadata.py`

Before editing `run_extraction`, run GitNexus impact:

```bash
# GitNexus MCP equivalent:
# impact({target: "run_extraction", direction: "upstream", repo: "IPKE"})
```

If impact is HIGH or CRITICAL, stop and report the blast radius before editing.

- [ ] **Step 1: Write metadata helper tests**

Create `tests/test_reproducibility_metadata.py`:

```python
import json
from pathlib import Path

from scripts.run_pkg_extraction import build_run_metadata


class DummyConfig:
    chunking_method = "dual_semantic"
    prompting_strategy = "P3"
    llm_backend = "llama_cpp"
    llm_model_path = "models/llm/mistral.gguf"
    llm_model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    llm_quantization = "Q4_K_M"
    llm_temperature = 0.1
    llm_random_seed = 42
    gpu_backend = "cuda"


def test_build_run_metadata_contains_reproducibility_fields(tmp_path):
    doc_path = tmp_path / "doc.txt"
    doc_path.write_text("Inspect valve.", encoding="utf-8")
    metadata = build_run_metadata(
        config=DummyConfig(),
        doc_id="DOC1",
        input_path=doc_path,
        flat_path=tmp_path / "DOC1_extracted.json",
        pkg_path=tmp_path / "DOC1_pkg.json",
    )
    assert metadata["doc_id"] == "DOC1"
    assert metadata["input_sha256"]
    assert metadata["chunking_method"] == "dual_semantic"
    assert metadata["prompting_strategy"] == "P3"
    assert metadata["llm_temperature"] == 0.1
    assert metadata["llm_random_seed"] == 42
    assert metadata["git_sha"]
    json.dumps(metadata)
```

- [ ] **Step 2: Run failing test**

Run:

```bash
uv run pytest tests/test_reproducibility_metadata.py -v
```

Expected before implementation:

```text
ImportError: cannot import name 'build_run_metadata'
```

- [ ] **Step 3: Add metadata helpers**

In `scripts/run_pkg_extraction.py`, add imports:

```python
import hashlib
import platform
import subprocess
from datetime import datetime, timezone
from typing import Any
```

Add these helpers above `run_extraction`:

```python
def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def build_run_metadata(
    *,
    config: Any,
    doc_id: str,
    input_path: Path,
    flat_path: Path,
    pkg_path: Path,
) -> dict[str, Any]:
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "doc_id": doc_id,
        "input_path": str(input_path),
        "input_sha256": _file_sha256(input_path),
        "flat_output_path": str(flat_path),
        "pkg_output_path": str(pkg_path),
        "chunking_method": getattr(config, "chunking_method", None),
        "prompting_strategy": getattr(config, "prompting_strategy", None),
        "llm_backend": getattr(config, "llm_backend", None),
        "llm_model_path": getattr(config, "llm_model_path", None),
        "llm_model_id": getattr(config, "llm_model_id", None),
        "llm_quantization": getattr(config, "llm_quantization", None),
        "llm_temperature": getattr(config, "llm_temperature", None),
        "llm_random_seed": getattr(config, "llm_random_seed", None),
        "gpu_backend": getattr(config, "gpu_backend", None),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }
```

- [ ] **Step 4: Write metadata from `run_extraction`**

After writing `pkg_path`, add:

```python
    metadata = build_run_metadata(
        config=config,
        doc_id=args.doc_id,
        input_path=doc_path,
        flat_path=flat_path,
        pkg_path=pkg_path,
    )
    metadata_path = output_dir / "config.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
```

Add this print line after the existing output prints:

```python
    print(f"Run metadata          : {metadata_path}")
```

- [ ] **Step 5: Run tests**

Run:

```bash
uv run pytest tests/test_reproducibility_metadata.py tests/test_run_pkg_extraction_cli.py -v
```

Expected:

```text
4 passed
```

- [ ] **Step 6: Commit**

Run:

```bash
git add scripts/run_pkg_extraction.py tests/test_reproducibility_metadata.py
git commit -m "Record extraction run metadata"
```

Expected: commit succeeds.

---

### Task 5: Standardize Experiment Output Roots

**Files:**
- Modify: `scripts/experiments/experiment_utils.py`
- Modify: `scripts/experiments/run_all_chunking_experiments.py`
- Create: `tests/test_experiment_paths.py`

Before editing `main` in `run_all_chunking_experiments.py`, run GitNexus impact:

```bash
# GitNexus MCP equivalent:
# impact({target: "main", direction: "upstream", repo: "IPKE", maxDepth: 2})
```

If impact is HIGH or CRITICAL, stop and report the blast radius before editing.

- [ ] **Step 1: Write failing path tests**

Create `tests/test_experiment_paths.py`:

```python
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
```

- [ ] **Step 2: Run failing tests**

Run:

```bash
uv run pytest tests/test_experiment_paths.py -v
```

Expected before implementation:

```text
FAILED tests/test_experiment_paths.py::test_default_experiment_root_uses_runs
FAILED tests/test_experiment_paths.py::test_master_run_roots_use_runs
```

- [ ] **Step 3: Change default experiment root**

In `scripts/experiments/experiment_utils.py`, change:

```python
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results" / "experiments"
```

to:

```python
DEFAULT_RESULTS_ROOT = REPO_ROOT / "runs" / "experiments"
```

- [ ] **Step 4: Add master root helper**

In `scripts/experiments/run_all_chunking_experiments.py`, add above `main`:

```python
def build_run_roots(timestamp: str) -> Dict[str, Path]:
    return {
        "run_root": REPO_ROOT / "runs" / "chunking_sweeps" / f"full_run_{timestamp}",
        "log_dir": REPO_ROOT / "runs" / "master_logs" / timestamp,
        "global_errors": REPO_ROOT / "runs" / "master_logs" / "errors.txt",
        "latest_summary": REPO_ROOT / "runs" / "latest" / "all_chunking_summary.csv",
    }
```

Then replace:

```python
    run_root = REPO_ROOT / "results" / f"full_run_{timestamp}"
    log_dir = REPO_ROOT / "results" / "master_logs" / timestamp
```

with:

```python
    roots = build_run_roots(timestamp)
    run_root = roots["run_root"]
    log_dir = roots["log_dir"]
```

Replace:

```python
    error_logger = ErrorLogger(run_specific=log_dir / "errors.txt", global_errors=REPO_ROOT / "results" / "master_logs" / "errors.txt")
```

with:

```python
    error_logger = ErrorLogger(run_specific=log_dir / "errors.txt", global_errors=roots["global_errors"])
```

Replace:

```python
    latest_summary = REPO_ROOT / "results" / "all_chunking_summary.csv"
```

with:

```python
    latest_summary = roots["latest_summary"]
```

- [ ] **Step 5: Run targeted tests**

Run:

```bash
uv run pytest tests/test_experiment_paths.py -v
```

Expected:

```text
2 passed
```

- [ ] **Step 6: Commit**

Run:

```bash
git add scripts/experiments/experiment_utils.py scripts/experiments/run_all_chunking_experiments.py tests/test_experiment_paths.py
git commit -m "Standardize experiment run roots"
```

Expected: commit succeeds.

---

### Task 6: Add Makefile Reproducibility Commands

**Files:**
- Create: `Makefile`
- Modify: `README.md`
- Create or modify: no Python tests required

- [ ] **Step 1: Create `Makefile`**

Create `Makefile`:

```makefile
.PHONY: test eval smoke-extract paper-table clean-artifacts

PYTHON := uv run python

test:
	uv run pytest

eval:
	$(PYTHON) -m src.evaluation.metrics \
		--gold_dir datasets/archive/gold_human \
		--pred_dir runs/pkg_extraction/3m_marine_oem_sop \
		--out_file runs/eval/latest_tier_a.json \
		--tier A

smoke-extract:
	$(PYTHON) scripts/run_pkg_extraction.py \
		--input-path datasets/archive/test_data/text/3m_marine_oem_sop.txt \
		--doc-id 3M_OEM_SOP \
		--output-dir runs/pkg_extraction/3m_marine_oem_sop \
		--chunking-method dsc \
		--prompting-strategy P3 \
		--gpu-backend cpu

paper-table:
	$(PYTHON) scripts/experiments/build_summary_csv.py

clean-artifacts:
	$(PYTHON) -c "from pathlib import Path; [p.unlink() for p in Path('runs').rglob('.DS_Store')] if Path('runs').exists() else None"
```

- [ ] **Step 2: Verify Makefile targets list**

Run:

```bash
make -n test
make -n smoke-extract
make -n eval
make -n paper-table
```

Expected: commands are printed without executing model inference.

- [ ] **Step 3: Update README command block**

In `README.md`, replace the run section commands with:

```markdown
## Reproducible Commands

```bash
uv sync
make test
make smoke-extract
make eval
```

Experiment artifacts are written under `runs/` and are intentionally ignored by git.
```
```

- [ ] **Step 4: Run test target**

Run:

```bash
make test
```

Expected:

```text
93 passed
```

The exact selected count may be higher after this plan's tests are added.

- [ ] **Step 5: Commit**

Run:

```bash
git add Makefile README.md
git commit -m "Add reproducibility make targets"
```

Expected: commit succeeds.

---

### Task 7: Scaffold Paper Dataset Layout

**Files:**
- Create: `datasets/paper/README.md`
- Create: `datasets/paper/text/.gitkeep`
- Create: `datasets/paper/gold/.gitkeep`

- [ ] **Step 1: Create dataset directories**

Run:

```bash
mkdir -p datasets/paper/text datasets/paper/gold
```

Expected: no output.

- [ ] **Step 2: Add `.gitkeep` files**

Create empty files:

```text
datasets/paper/text/.gitkeep
datasets/paper/gold/.gitkeep
```

- [ ] **Step 3: Add paper dataset README**

Create `datasets/paper/README.md`:

```markdown
# Paper Dataset Workspace

This directory is for ECIR 2027 paper documents added after the thesis archive.

## Layout

- `text/` - plain-text public SOP/manual documents used for experiments.
- `gold/` - Tier A gold annotations using the same schema as `datasets/archive/test_data/gold/`.

## Rules

- Do not mutate `datasets/archive/` gold annotations.
- Do not commit partner-private SOPs.
- Prefer public documents with stable URLs and licenses.
- Record source URL, access date, license, conversion command, and any extraction caveat in the experiment run metadata.
- Keep document IDs stable once gold annotation starts.

## Validation

Before running paper experiments, each new gold file must parse as JSON and include:

- `steps`
- `constraints`
- stable step IDs
- stable constraint IDs
```

- [ ] **Step 4: Verify files are tracked**

Run:

```bash
git status --short datasets/paper
```

Expected:

```text
?? datasets/paper/
```

- [ ] **Step 5: Commit**

Run:

```bash
git add datasets/paper
git commit -m "Add paper dataset scaffold"
```

Expected: commit succeeds.

---

### Task 8: Document Reproducibility and DSC Implementation

**Files:**
- Create: `docs/reproducibility.md`
- Create: `docs/methods/dsc-implementation.md`
- Modify: `README.md`

- [ ] **Step 1: Create methods directory**

Run:

```bash
mkdir -p docs/methods
```

Expected: no output.

- [ ] **Step 2: Add reproducibility documentation**

Create `docs/reproducibility.md`:

```markdown
# Reproducibility

## Fresh Clone

```bash
uv sync
make test
```

## Single-Document Smoke Extraction

```bash
make smoke-extract
```

Outputs are written to:

```text
runs/pkg_extraction/3m_marine_oem_sop/
```

The run directory contains:

- `3M_OEM_SOP_extracted.json`
- `3M_OEM_SOP_pkg.json`
- `config.json`

## Evaluation

```bash
make eval
```

This evaluates predictions against `datasets/archive/gold_human`.

## Required Metadata

Each final experiment run must record:

- git SHA
- document ID
- input SHA256
- model name or path
- quantization
- temperature
- random seed
- backend
- hardware backend
- Python version
- platform

## Known Limits

- Thesis archive results cover three documents.
- ECIR paper experiments require `datasets/paper/` expansion.
- ConstraintAttachmentF1 remains strict unless the fuzzy metric task is implemented.
```

- [ ] **Step 3: Add DSC implementation note**

Create `docs/methods/dsc-implementation.md`:

```markdown
# Dual Semantic Chunking Implementation

The current implementation is in `src/processors/chunkers/dual_semantic.py`.

## Implemented Algorithm

IPKE's current Dual Semantic Chunker:

1. Splits document text into sentences with spaCy.
2. Embeds sentences with SentenceTransformers.
3. Computes adjacent sentence cosine distances.
4. Builds parent boundaries using local distance thresholds and optional heading detection.
5. Refines each parent block with `BreakpointSemanticChunker._compute_boundaries`.
6. Enforces the configured character cap.

## Paper Wording Constraint

The thesis describes DSC using a global objective and heading bonus. The code currently implements a practical hierarchical heuristic rather than a single global DSC dynamic program with explicit heading bonus.

For ECIR 2027, choose one:

- Update the method text to describe the implemented heuristic exactly.
- Or implement the global DSC objective and validate it against the existing chunker tests and experiment results.

Until that decision is made, paper drafts should avoid claiming the current code solves a single global DSC objective with heading bonus.
```

- [ ] **Step 4: Link docs from README**

Add this block near the README run instructions:

```markdown
## Research Reproducibility

- [Reproducibility guide](docs/reproducibility.md)
- [Implemented DSC method note](docs/methods/dsc-implementation.md)
- [Paper dataset workspace](datasets/paper/README.md)
```

- [ ] **Step 5: Commit**

Run:

```bash
git add docs/reproducibility.md docs/methods/dsc-implementation.md README.md
git commit -m "Document reproducibility workflow"
```

Expected: commit succeeds.

---

### Task 9: Split Packaging Extras

**Files:**
- Modify: `pyproject.toml`
- Create: `tests/test_packaging_config.py`

- [ ] **Step 1: Write packaging tests**

Create `tests/test_packaging_config.py`:

```python
import tomllib
from pathlib import Path


def load_pyproject():
    return tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))


def test_base_dependencies_exclude_app_stack():
    deps = "\n".join(load_pyproject()["project"]["dependencies"])
    assert "streamlit" not in deps
    assert "fastapi" not in deps
    assert "uvicorn" not in deps


def test_base_dependencies_exclude_heavy_llm_backends():
    deps = "\n".join(load_pyproject()["project"]["dependencies"])
    assert "llama-cpp-python" not in deps
    assert "transformers" not in deps
    assert "bitsandbytes" not in deps


def test_expected_optional_extras_exist():
    extras = load_pyproject()["project"]["optional-dependencies"]
    for name in ["app", "llm", "dev", "extras", "neo4j"]:
        assert name in extras
```

- [ ] **Step 2: Run failing packaging tests**

Run:

```bash
uv run pytest tests/test_packaging_config.py -v
```

Expected before implementation:

```text
FAILED tests/test_packaging_config.py::test_base_dependencies_exclude_app_stack
FAILED tests/test_packaging_config.py::test_base_dependencies_exclude_heavy_llm_backends
```

- [ ] **Step 3: Rewrite dependencies in `pyproject.toml`**

Set `[project].dependencies` to:

```toml
dependencies = [
    "numpy>=1.24.4,<2.0",
    "scipy>=1.11.3",
    "networkx>=3.1",
    "tqdm>=4.66.1",
    "spacy>=3.7.2",
    "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl",
    "sentence-transformers>=2.2.2",
    "torch>=2.1.0",
    "requests>=2.31.0",
    "jsonschema==4.20.0",
    "python-dotenv>=1.0.0",
    "pandas>=2.0.0",
]
```

Set `[project.optional-dependencies]` to include:

```toml
app = [
    "fastapi>=0.104.0",
    "python-multipart>=0.0.20",
    "uvicorn>=0.24.0",
    "streamlit>=1.38.0",
    "pyvis>=0.3.2",
]
llm = [
    "llama-cpp-python==0.3.23",
    "transformers>=4.36.0",
    "accelerate>=0.25.0",
    "bitsandbytes>=0.41.3",
]
dev = [
    "pytest>=7.4.0",
    "pytest-asyncio>=0.23.0",
]
neo4j = [
    "neo4j==5.14.0",
]
extras = [
    "PyMuPDF>=1.23.0",
    "python-pptx>=0.6.23",
    "python-docx>=0.8.11",
    "easyocr>=1.7.0",
    "openai-whisper>=20231117",
]
```

- [ ] **Step 4: Update README setup commands**

Replace setup text with:

```markdown
```bash
uv sync --extra dev
```

Use extras as needed:

```bash
uv sync --extra dev --extra llm
uv sync --extra dev --extra app
uv sync --extra dev --extra extras
uv sync --extra dev --extra neo4j
```
```

- [ ] **Step 5: Update lockfile**

Run:

```bash
uv lock
```

Expected: `uv.lock` updates successfully.

- [ ] **Step 6: Run packaging tests and full tests**

Run:

```bash
uv run --extra dev pytest tests/test_packaging_config.py -v
uv run --extra dev pytest
```

Expected:

```text
tests/test_packaging_config.py passes
full non-integration suite passes
```

- [ ] **Step 7: Commit**

Run:

```bash
git add pyproject.toml uv.lock README.md tests/test_packaging_config.py
git commit -m "Split optional dependency groups"
```

Expected: commit succeeds.

---

### Task 10: Final Validation and GitNexus Change Review

**Files:**
- No new files unless validation reveals a defect.

- [ ] **Step 1: Run full fast suite**

Run:

```bash
uv run --extra dev pytest
```

Expected:

```text
all non-integration tests pass
```

- [ ] **Step 2: Run Makefile dry runs**

Run:

```bash
make -n test
make -n smoke-extract
make -n eval
make -n paper-table
```

Expected: all commands print without Makefile syntax errors.

- [ ] **Step 3: Run GitNexus detect changes**

Run GitNexus MCP:

```text
detect_changes({repo: "IPKE", scope: "all"})
```

Expected:

```text
Changed symbols and affected flows match expected scope:
- extraction CLI
- experiment path helpers
- docs and packaging
```

If GitNexus reports unexpected high-risk flows, inspect them before finalizing.

- [ ] **Step 4: Inspect git diff**

Run:

```bash
git status --short
git diff --stat
```

Expected:

```text
Only files from this plan are modified.
No model files, run artifacts, private PDFs, logs, or gold annotation mutations are staged.
```

- [ ] **Step 5: Final commit if needed**

If validation required small fixes in the planned files, commit them:

```bash
git add .gitignore pyproject.toml pytest.ini Makefile README.md docs datasets/paper scripts/run_pkg_extraction.py scripts/experiments tests
git commit -m "Fix research repo validation"
```

Expected: commit succeeds.

---

## Self-Review

Spec coverage:

- Branch move: covered in Required Git Setup.
- Clean repo hygiene: Task 1.
- Pytest config duplication: Task 2.
- CLI alias mismatch: Task 3.
- Run metadata: Task 4.
- Output root standardization: Task 5.
- One-command rerun surface: Task 6.
- Paper dataset layout: Task 7.
- DSC thesis/code alignment: Task 8.
- Minimal dependency packaging: Task 9.
- Full verification and GitNexus scope check: Task 10.

No destructive operation is required. Gold annotations are not modified. Existing untracked `.agents/` and thesis PDF are intentionally left alone.
