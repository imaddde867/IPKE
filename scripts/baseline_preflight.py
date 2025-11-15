#!/usr/bin/env python3
"""Quick readiness check for running scripts/run_baseline_loops.py."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.unified_config import get_config
from src.pipelines.baseline import DEFAULT_EMBEDDING_MODEL, DEFAULT_GOLD_DIR, TIER_A_TEST_DOCS


@dataclass
class CheckResult:
    name: str
    target: str
    status: str
    hint: str = ""

    def as_dict(self) -> dict:
        return {"name": self.name, "target": self.target, "status": self.status, "hint": self.hint}


def _exists_check(name: str, path: Path) -> CheckResult:
    expanded = path.expanduser()
    if expanded.exists():
        return CheckResult(name, str(expanded), "ok")
    return CheckResult(
        name,
        str(expanded),
        "missing",
        hint=f"Expected file/directory not found at {expanded}",
    )


def _import_check(name: str, module: str) -> CheckResult:
    try:
        importlib.import_module(module)
        return CheckResult(name, module, "ok")
    except Exception as exc:  # noqa: BLE001
        return CheckResult(name, module, "missing", hint=f"Import failed: {exc}")


def _check_spacy_model(model_name: str) -> CheckResult:
    """Check that a spaCy model is available across spaCy versions.

    Uses multiple strategies to avoid version-specific util APIs:
    1) importlib spec check
    2) optional util.get_data_path() if present (older spaCy)
    3) attempt a lightweight spacy.load(model_name)
    """
    try:
        import importlib.util
        import spacy

        # 1) Check if the package can be imported directly
        if importlib.util.find_spec(model_name) is not None:
            return CheckResult("spaCy model", model_name, "ok")

        # 2) Older spaCy exposed util.get_data_path(); guard via getattr
        get_data_path = getattr(spacy.util, "get_data_path", None)
        if callable(get_data_path):
            data_path = get_data_path()
            try:
                if data_path and Path(data_path, model_name).exists():
                    return CheckResult("spaCy model", model_name, "ok")
            except Exception:
                # Ignore path issues and fall through to load()
                pass

        # 3) Try to load the model; success implies installation
        try:
            _nlp = spacy.load(model_name)
            # Best-effort cleanup
            try:
                _ = _nlp.pipe_names  # touch to ensure it's initialized
            except Exception:
                pass
            return CheckResult("spaCy model", model_name, "ok")
        except Exception:
            pass

        return CheckResult(
            "spaCy model",
            model_name,
            "missing",
            hint=f"Install it via: python -m spacy download {model_name}",
        )
    except Exception as exc:  # noqa: BLE001
        return CheckResult("spaCy availability", "spacy", "missing", hint=str(exc))


def checks() -> List[CheckResult]:
    config = get_config()
    results: List[CheckResult] = []

    for doc_id, path in TIER_A_TEST_DOCS.items():
        results.append(_exists_check(f"Document {doc_id}", Path(path)))

    results.append(_exists_check("Gold annotations", DEFAULT_GOLD_DIR))
    results.append(_exists_check("Embedding model", Path(DEFAULT_EMBEDDING_MODEL)))
    chunking_method = (config.chunking_method or "fixed").lower()
    if chunking_method in {"breakpoint_semantic", "dsc"}:
        results.append(
            _exists_check(
                f"Chunking embedder ({chunking_method})",
                Path(config.embedding_model_path),
            )
        )
    results.append(_exists_check("LLM model", Path(config.llm_model_path)))

    results.append(_import_check("llama-cpp-python", "llama_cpp"))
    results.append(_import_check("sentence-transformers", "sentence_transformers"))
    results.append(_check_spacy_model("en_core_web_sm"))

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify assets required for scripts/run_baseline_loops.py")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON output.")
    args = parser.parse_args()

    results = checks()
    missing = [item for item in results if item.status != "ok"]

    if args.json:
        print(json.dumps([item.as_dict() for item in results], indent=2))
    else:
        print("Baseline readiness checks:")
        for item in results:
            status = "LOCKED N LOADED, vamos" if item.status == "ok" else "Naaaaaahhhh this aint working.. lol"
            line = f"  {status} {item.name:20} -> {item.target}"
            if item.hint and item.status != "ok":
                line += f"\n      hint: {item.hint}"
            print(line)

    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
