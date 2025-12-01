#!/usr/bin/env python3
"""Re-evaluate existing extraction results with the updated A_score formula (0.5*ConstraintCoverage + 0.3*StepF1 + 0.2*Kendall)."""

import json
import sys
from pathlib import Path
from collections import defaultdict

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.metrics import run_evaluation
from src.graph.adapter import flat_to_tierb
from scripts.experiments.experiment_utils import write_summary_table

def get_gold_paths(doc_id):
    """Return the path to the gold standard file based on the document id."""
    gold_tier_a_dir = REPO_ROOT / "datasets" / "archive" / "gold_human"
    gold_tier_b_dir = REPO_ROOT / "datasets" / "archive" / "gold_human_tierb"
    
    doc_map = {
        "3M_OEM_SOP": "3M_OEM_SOP.json",
        "DOA_Food_Proc": "DOA_Food_Man_Proc_Stor.json",
        "op_firesafety_guideline": "op_firesafety_guideline.json"
    }
    
    doc_name = doc_map.get(doc_id)
    if not doc_name:
        raise ValueError(f"Unknown document id: {doc_id}")

    return gold_tier_a_dir / doc_name, gold_tier_b_dir / doc_name

def main():
    """Main function to re-evaluate metrics."""
    base_dir = REPO_ROOT / "logs" / "chunking_grid_core_v2"
    experiments = [
        "fixed_2000_no_overlap",
        "fixed_2000_overlap200",
        "breakpoint_semantic_lambda015",
        "dual_semantic_block16"
    ]
    documents = [
        "3M_OEM_SOP",
        "DOA_Food_Proc",
        "op_firesafety_guideline"
    ]
    
    all_metrics_rows = []
    
    # For before/after comparison
    first_metrics_before = None
    first_metrics_after = None
    first_metrics_path = None

    for experiment in experiments:
        for document in documents:
            exp_dir = base_dir / experiment / document
            pred_path = exp_dir / "predictions.json"
            metrics_path = exp_dir / "metrics.json"

            if not pred_path.exists():
                print(f"⚠️  Missing predictions: {pred_path}")
                continue

            gold_path_a, gold_path_b = get_gold_paths(document)

            if metrics_path.exists() and first_metrics_before is None:
                with open(metrics_path, 'r') as f:
                    first_metrics_before = json.load(f)
                first_metrics_path = metrics_path

            # Tier A
            metrics_a = run_evaluation(
                gold=str(gold_path_a),
                pred=str(pred_path),
                tiers=("A",),
            )
            
            # Tier B
            with open(pred_path, 'r') as f:
                predictions = json.load(f)
            tierb_pred = flat_to_tierb(predictions)
            
            metrics_b = run_evaluation(
                gold=str(gold_path_b),
                pred=tierb_pred,
                tiers=("B",),
                preserve_conflicts=True,
            )
            
            # Merge metrics
            final_metrics = metrics_a.copy()
            final_metrics.update(metrics_b)

            # Save updated metrics
            with open(metrics_path, 'w') as f:
                json.dump(final_metrics, f, indent=4)
            
            if first_metrics_after is None:
                first_metrics_after = final_metrics

            print(f"✅ Updated: {experiment}/{document}")
            print(f"   New A_score: {final_metrics.get('A_score'):.3f}")
            
            # Prepare row for summary csv
            chunk_method = "unknown"
            if "fixed_2000_no_overlap" in experiment:
                chunk_method = "fixed_2000_no_overlap"
            elif "fixed_2000_overlap200" in experiment:
                chunk_method = "fixed_2000_overlap200"
            elif "breakpoint_semantic_lambda015" in experiment:
                chunk_method = "breakpoint_semantic_lambda015"
            elif "dual_semantic_block16" in experiment:
                chunk_method = "dual_semantic_block16"

            row = {
                "experiment": experiment,
                "chunk_method": chunk_method,
                "document_id": document,
                "document_type": "industrial",
            }
            row.update(final_metrics)
            all_metrics_rows.append(row)

    # Regenerate summary.csv
    summary_path = base_dir / "summary.csv"
    
    headers = [
        "experiment", "chunk_method", "document_id", "document_type",
        "StepF1", "AdjacencyF1", "Kendall", "ConstraintCoverage", 
        "ConstraintAttachmentF1", "A_score", "GraphF1", "NEXT_EdgeF1", 
        "Logic_EdgeF1", "ConstraintAttachmentF1_TierB", "B_score"
    ]
    
    filtered_rows = []
    for row in all_metrics_rows:
        filtered_row = {h: row.get(h) for h in headers}
        filtered_rows.append(filtered_row)

    write_summary_table(filtered_rows, summary_path)
    print("\n✅ Regenerated summary.csv")
    
    if first_metrics_before and first_metrics_after:
        print(f"\n🔍 Example metrics.json before/after for {first_metrics_path}:")
        print("\n--- BEFORE ---")
        print(json.dumps(first_metrics_before, indent=2))
        print("\n--- AFTER ---")
        print(json.dumps(first_metrics_after, indent=2))

    macro_scores = defaultdict(list)
    for row in all_metrics_rows:
        macro_scores[row['chunk_method']].append(row.get('A_score') or 0.0)
    
    print("\n📈 Macro-average A_scores:")
    for method, scores in macro_scores.items():
        avg_score = sum(scores) / len(scores) if scores else 0.0
        print(f"- {method}: {avg_score:.3f}")


if __name__ == "__main__":
    main()