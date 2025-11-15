# Repository Guidelines

## Project Structure & Module Organization
Core runtime logic lives in `src/`: `api/app.py` hosts the FastAPI interface, `ai/knowledge_engine.py` selects the Metal/CUDA/CPU inference strategy, `processors/streamlined_processor.py` performs document ingestion, and `core/unified_config.py` plus `graph/` hold shared settings and schemas. Supporting data and checkpoints reside under `datasets/`, `models/`, `uploaded_files/`, and metrics land in `logs/`. Automation helpers stay in `scripts/` (preflight, batch evaluation, plotting) and `tools/` (e.g., `tools/evaluate.py` for Tier A/B scoring). The API boots from `main.py`, the Streamlit UI from `streamlit_app.py`, and tests mirror this layout beneath `tests/`.

## Build, Test, and Development Commands
```bash
python -m venv .venv && source .venv/bin/activate  # create an isolated 3.10 env
pip install -r requirements.txt                   # install FastAPI/LLM/test deps
python scripts/baseline_preflight.py              # verify spaCy + local assets
python main.py                                    # start the FastAPI server on :8000
streamlit run streamlit_app.py                    # launch the upload/preview UI
pytest                                            # execute unit + async integration tests
python scripts/run_baseline_loops.py --runs 3     # batch extract and evaluate datasets/
```
Configure `GPU_BACKEND` and download weights before invoking the engine.

## Coding Style & Naming Conventions
Write Python that follows PEP 8 with four-space indents, snake_case functions, and PascalCase classes (`ExtractedEntity`, request models). Favor type hints and dataclasses for structured outputs, keep modules cohesive (one processor or strategy per file), and route configuration through `UnifiedConfig` instead of hard-coded paths. Keep async functions await-safe like `knowledge_engine.extract`.

## Testing Guidelines
`pytest` with `pytest-asyncio` powers the suite; keep files named `test_<module>.py` and mirror `src/` so failures map directly. Mock heavy dependencies (LLM calls, OCR) and provide deterministic fixtures in `tests/fixtures/` or inline sample docs. Extend tests whenever you touch extraction logic, adapters, or public API schemas, and capture evaluator regressions via `tools/evaluate.py` when graph outputs change.

## Commit & Pull Request Guidelines
Commits follow short, imperative subjects (`refactor: improve processor chunking`), as seen in `git log`. Each PR should summarize intent, list validation commands (`pytest`, `python scripts/baseline_preflight.py`), and link the driving issue. Include screenshots or sample JSON when API or UI payloads shift, and update README/AGENTS.md when flags, scripts, or datasets move.

## Configuration & Security Notes
Copy `.env.example` to `.env`, set `GPU_BACKEND` (`metal|cuda|auto`), and provide `LLM_MODEL_PATH` for GGUF or `LLM_MODEL_ID` for remote downloads. Keep credentials, gold datasets, and model binaries out of Git—they belong in the ignored asset folders. Prefer anonymized samples in `uploaded_files/` for reproduction.
