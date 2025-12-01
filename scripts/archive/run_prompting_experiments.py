#!/usr/bin/env python3
"""Prompting experiment runner for IPKE.

Example:
    python scripts/run_prompting_experiments.py --config configs/prompting_grid.yaml \
        --out-root logs/prompting_grid --evaluate true
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.core.unified_config import UnifiedConfig
from src.processors.streamlined_processor import StreamlinedDocumentProcessor
from scripts.experiment_lib import (
    EvaluationContext,
    apply_chunk_settings,
    apply_llm_settings,
    apply_prompt_settings,
    bool_arg,
    ensure_out_root,
    evaluate_document,
    load_spec,
    normalize_documents,
    prepare_evaluation_context,
    process_document,
    serialize_config_snapshot,
    write_summary_csv,
)

LOGGER = logging.getLogger("run_prompting_experiments")


async def run_prompting_experiment(
    experiment: Dict[str, Any],
    base_config: UnifiedConfig,
    docs: List[Dict[str, Any]],
    out_root: Path,
    evaluate_flag: bool,
    eval_ctx: Optional[EvaluationContext],
) -> List[Dict[str, Any]]:
    config = copy.deepcopy(base_config)
    apply_prompt_settings(config, experiment)
    exp_name = experiment.get("name") or experiment.get("strategy")
    experiment_dir = out_root / exp_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    (experiment_dir / "config_snapshot.json").write_text(
        json.dumps(config.__dict__, indent=2, default=str), encoding="utf-8"
    )
    processor = StreamlinedDocumentProcessor(config)
    summary_rows: List[Dict[str, Any]] = []

    for doc in docs:
        doc_out_dir = experiment_dir / doc["id"]
        result, prediction_path, tierb_path = await process_document(processor, doc, doc_out_dir)
        metrics = None
        if evaluate_flag:
            metrics = evaluate_document(doc, prediction_path, eval_ctx, tierb_path=tierb_path)
        summary_row = {
            "experiment": exp_name,
            "prompting_strategy": config.prompting_strategy,
            "document_id": doc["id"],
            "document_type": doc.get("type", "unknown"),
        }
        if metrics:
            summary_row.update(metrics)
        summary_rows.append(summary_row)

    processor.knowledge_engine.clear_cache()
    return summary_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Execute prompting experiments (e.g., python scripts/run_prompting_experiments.py "
            "--config configs/prompting_grid.yaml --out-root logs/prompting_grid --evaluate true)."
        )
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the experiment configuration (YAML/JSON), e.g., configs/prompting_grid.yaml.",
    )
    parser.add_argument("--out-root", default=None, help="Optional override for the output directory.")
    parser.add_argument("--evaluate", default="true", help="Whether to run evaluation (true/false).")
    parser.add_argument("--tier", default="both", choices=["A", "B", "both"], help="Tier(s) to evaluate (overrides config)")
    parser.add_argument("--log-level", default="INFO", help="Logging verbosity.")
    return parser.parse_args()


async def main_async() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    spec_path = Path(args.config)
    if not spec_path.exists():
        raise FileNotFoundError(f"Config not found: {spec_path}")
    spec = load_spec(spec_path)
    out_root = ensure_out_root(spec, args.out_root, ROOT)

    if args.tier:
        spec.setdefault("evaluation", {})["tier"] = args.tier

    docs = normalize_documents(spec, ROOT)
    prompting_experiments = spec.get("prompting_experiments")
    if not prompting_experiments:
        raise ValueError("Config must include a 'prompting_experiments' list.")

    base_config = UnifiedConfig.from_environment()
    apply_llm_settings(base_config, spec.get("llm", {}), spec_path.parent)
    apply_chunk_settings(base_config, spec.get("chunking", {}))
    serialize_config_snapshot(base_config, out_root / "base_config_snapshot.json")

    evaluation_flag = bool_arg(args.evaluate, default=True)
    eval_ctx = prepare_evaluation_context(spec.get("evaluation", {})) if evaluation_flag else None

    all_rows: List[Dict[str, Any]] = []
    for experiment in prompting_experiments:
        rows = await run_prompting_experiment(
            experiment=experiment,
            base_config=base_config,
            docs=docs,
            out_root=out_root,
            evaluate_flag=evaluation_flag,
            eval_ctx=eval_ctx,
        )
        all_rows.extend(rows)

    if evaluation_flag:
        fieldnames = [
            "experiment",
            "prompting_strategy",
            "document_id",
            "document_type",
            "StepF1",
            "AdjacencyF1",
            "Kendall",
            "ConstraintCoverage",
            "ConstraintAttachmentF1",
            "A_score",
            "GraphPrecision",
            "GraphRecall",
            "GraphF1",
            "NEXT_EdgeF1",
            "Logic_EdgeF1",
            "ConstraintAttachmentF1_TierB",
            "B_score",
        ]
        write_summary_csv(out_root, all_rows, fieldnames)
    LOGGER.info("Artifacts available under %s", out_root)


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
