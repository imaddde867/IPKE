#!/usr/bin/env python3
"""
Run extraction and evaluation on test documents.

Usage:
    python scripts/run_baseline_loops.py [--runs N] [--out DIR] [--visualize]

Examples:
    python scripts/run_baseline_loops.py                    # Single run
    python scripts/run_baseline_loops.py --runs 5           # Multiple runs for statistics
    python scripts/run_baseline_loops.py --visualize        # Include visualizations
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

# Ensure project root is on sys.path when running from scripts/
import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Configure unlimited chunks
os.environ['LLM_MAX_CHUNKS'] = '0'

from src.pipelines.baseline import (
    TIER_A_TEST_DOCS,
    accumulate_metrics,
    evaluate_predictions,
    extract_documents,
    summarise_metrics,
)
from src.processors.streamlined_processor import StreamlinedDocumentProcessor
from src.graph.builder import ProceduralGraphBuilder


async def run_extraction(run_dir: Path) -> None:
    """Extract baseline predictions for all documents into ``run_dir``."""
    processor = StreamlinedDocumentProcessor()
    
    # Print configuration
    config = processor.config
    print(f"Configuration: max_chunks={config.llm_max_chunks} (0=unlimited), "
          f"gpu_layers={config.llm_n_gpu_layers}")
    
    await extract_documents(
        processor,
        doc_sources=TIER_A_TEST_DOCS,
        run_dir=run_dir,
        status_callback=lambda doc_id, payload: print(
            f"[extract] {doc_id}: {len(payload['steps'])} steps, "
            f"{len(payload['constraints'])} constraints, {len(payload['entities'])} entities"
        ),
    )


def run_evaluation(pred_dir: Path, out_path: Path) -> Dict[str, Dict[str, float | None]]:
    """Run the evaluation CLI against ``pred_dir`` and return the parsed JSON results."""
    print(f"[evaluate] -> {out_path}")
    return evaluate_predictions(pred_dir=pred_dir, out_path=out_path)


def generate_visualizations(run_dir: Path) -> None:
    """Generate visualizations if matplotlib is available."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        
        eval_file = run_dir / "evaluation_report.json"
        if not eval_file.exists():
            return
        
        with open(eval_file) as f:
            metrics = json.load(f)
        
        plots_dir = run_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        docs = [k for k in metrics.keys() if k != "macro_avg"]
        available_metrics = ["StepF1", "AdjacencyF1", "Kendall", 
                            "ConstraintCoverage", "ConstraintAttachmentF1", "A_score"]
        
        # Macro average bar chart
        if "macro_avg" in metrics:
            fig, ax = plt.subplots(figsize=(10, 6))
            values = [metrics["macro_avg"].get(m, 0.0) or 0.0 for m in available_metrics]
            bars = ax.bar(available_metrics, values, color='steelblue', edgecolor='navy', linewidth=1.5)
            ax.set_ylabel('Score', fontsize=12)
            ax.set_title('Macro Average Metrics', fontsize=14, fontweight='bold')
            ax.set_ylim(0, 1.05)
            ax.grid(axis='y', alpha=0.3)
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                       f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
            
            plt.xticks(rotation=15, ha='right')
            plt.tight_layout()
            plt.savefig(plots_dir / "macro_average.png", dpi=200, bbox_inches='tight')
            plt.close()
            
            print(f"[visualize] Saved plot to {plots_dir / 'macro_average.png'}")
    
    except ImportError:
        print("[visualize] matplotlib not available, skipping plots")
    except Exception as e:
        print(f"[visualize] Failed to generate plots: {e}")


def collect_constraint_reports(run_dir: Path, doc_ids: Iterable[str]) -> Dict[str, Dict[str, List]]:
    builder = ProceduralGraphBuilder()
    reports: Dict[str, Dict[str, List]] = {}
    for doc_id in doc_ids:
        doc_path = run_dir / f"{doc_id}.json"
        if not doc_path.exists():
            continue
        payload = json.loads(doc_path.read_text(encoding="utf-8"))
        builder.build_from_payload(payload)
        reports[doc_id] = builder.get_validation_report().to_dict()
    return reports


def write_validation_report(run_dir: Path, documents: Dict[str, Dict[str, List]]) -> None:
    entry = {
        "run_dir": str(run_dir),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "documents": documents,
    }
    report_path = Path("results/validation_report.json")
    existing: List[Dict[str, object]]
    if report_path.exists():
        try:
            data = json.loads(report_path.read_text(encoding="utf-8"))
            existing = data if isinstance(data, list) else [data]
        except json.JSONDecodeError:
            existing = []
    else:
        existing = []
    existing.append(entry)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")


async def orchestrate(runs: int, out_root: Path, visualize: bool = False) -> None:
    """Execute multiple extraction/evaluation runs and summarise the results."""
    accumulator: Dict[str, Dict[str, List[float]]] = {}
    for run_idx in range(1, runs + 1):
        run_dir = out_root / f"run_{run_idx}"
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"=== Run {run_idx}/{runs} ===")
        await run_extraction(run_dir)
        report_path = run_dir / "evaluation_report.json"
        metrics = run_evaluation(run_dir, report_path)
        accumulate_metrics(accumulator, metrics)

        doc_reports = collect_constraint_reports(run_dir, TIER_A_TEST_DOCS.keys())
        write_validation_report(run_dir, doc_reports)

        # Generate visualizations if requested and on last run
        if visualize and run_idx == runs:
            generate_visualizations(run_dir)

    average = summarise_metrics(accumulator)
    summary_path = out_root / "aggregate_metrics.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(average, indent=2), encoding="utf-8")
    print(f"\nAggregate metrics saved to {summary_path}")
    print(json.dumps(average, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run extraction and evaluation on test documents",
        epilog="Examples:\n"
               "  python scripts/run_baseline_loops.py\n"
               "  python scripts/run_baseline_loops.py --runs 5\n"
               "  python scripts/run_baseline_loops.py --visualize\n",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--runs", type=int, default=1, help="Number of runs to execute (default: 1)")
    parser.add_argument("--out", type=Path, default=Path("logs/extraction"), help="Output directory (default: logs/extraction)")
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations")
    parser.add_argument(
        "--strict-validation",
        action="store_true",
        help="Fail the run if schema validation errors occur",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.strict_validation:
        os.environ["STRICT_SCHEMA_VALIDATION"] = "1"
    asyncio.run(orchestrate(args.runs, args.out, args.visualize))


if __name__ == "__main__":
    main()
