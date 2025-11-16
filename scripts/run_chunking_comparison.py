# scripts/run_chunking_comparison.py
"""
A focused script to run the scientific chunking experiments for the thesis.

This script compares three chunking strategies (fixed, semantic, dsc)
across the three core documents of the thesis dataset.

It runs the extraction and evaluation for each combination and prints a
summary of the results.
"""
import os
import sys
import json
import subprocess
import gc
from typing import List, Dict, Any

# Ensure the src directory is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.unified_config import UnifiedConfig
from src.processors.streamlined_processor import StreamlinedDocumentProcessor
from tools.evaluate import evaluate_extraction

# --- Experiment Configuration ---

CHUNK_METHODS_TO_TEST = ["fixed", "semantic", "dsc"]
DOCUMENTS_TO_TEST = [
    "3M_OEM_SOP",
    "DOA_Food_Man_Proc_Stor",
    "op_firesafety_guideline",
]

# --- Helper Functions ---

def run_single_experiment(
    document_name: str,
    chunk_method: str,
    config: UnifiedConfig,
) -> Dict[str, Any]:
    """
    Runs the extraction and evaluation for a single document and chunking method.
    """
    print(f"--- Running: Document='{document_name}', Chunker='{chunk_method}' ---")

    # Override the chunker configuration for this run
    config.set('chunking_strategy', chunk_method)
    
    # HACK: We need to re-initialize the processor for the config change to take effect
    # This is because the chunker is instantiated in the processor's __init__
    processor = StreamlinedDocumentProcessor(config=config)

    input_pdf_path = os.path.join(config.get('data_dir'), "Samples", f"{document_name}.pdf")
    gold_json_path = os.path.join(config.get('data_dir'), "archive", "gold_human", f"{document_name}.json")
    output_dir = os.path.join("results", "chunking_experiments", chunk_method)
    os.makedirs(output_dir, exist_ok=True)
    output_json_path = os.path.join(output_dir, f"{document_name}.json")

    if not os.path.exists(input_pdf_path):
        print(f"ERROR: Input PDF not found at {input_pdf_path}")
        return {"error": "Input file not found"}

    if not os.path.exists(gold_json_path):
        print(f"ERROR: Gold JSON not found at {gold_json_path}")
        return {"error": "Gold file not found"}

    print(f"Processing {input_pdf_path}...")
    extraction_result = processor.process_document(input_pdf_path)

    print(f"Saving extraction to {output_json_path}...")
    with open(output_json_path, "w") as f:
        json.dump(extraction_result, f, indent=4)

    print("Evaluating extraction...")
    metrics = evaluate_extraction(
        gold_file=gold_json_path,
        prediction_file=output_json_path,
        align_threshold=config.get('eval_align_threshold'),
    )
    
    # Select and format the key metrics for the report
    main_metrics = metrics.get("TierA", {})
    report = {
        "Step F1": main_metrics.get("step_f1", 0),
        "Adjacency F1": main_metrics.get("adjacency_f1", 0),
        "Kendall τ": main_metrics.get("kendall_tau", {}).get("tau_b", 0),
        "Constraint Coverage": main_metrics.get("constraint_coverage", 0),
        "A-score": 0.7 * main_metrics.get("step_f1", 0) + 0.3 * main_metrics.get("adjacency_f1", 0),
    }

    # --- Memory Cleanup ---
    print("Cleaning up processor and running garbage collection...")
    del processor
    gc.collect()
    # --- End Cleanup ---

    print("--- Done. ---")
    return report


def print_results_table(results: Dict[str, Dict[str, Any]]):
    """
    Prints a formatted table of the experiment results.
    """
    # Header
    header = f"{'Document':<30} | {'Chunker':<10} | {'A-score':<10} | {'Step F1':<10} | {'Adj. F1':<10} | {'Kendall τ':<10}"
    print("\n" + "="*len(header))
    print("Chunking Experiment Results")
    print("="*len(header))
    print(header)
    print("-"*len(header))

    # Body
    for doc_name, chunker_results in results.items():
        for chunker_name, metrics in chunker_results.items():
            if "error" in metrics:
                print(f"{doc_name:<30} | {chunker_name:<10} | {'ERROR':<10}")
                continue
            
            a_score = metrics.get('A-score', 0)
            step_f1 = metrics.get('Step F1', 0)
            adj_f1 = metrics.get('Adjacency F1', 0)
            kendall = metrics.get('Kendall τ', 0)
            
            print(
                f"{doc_name:<30} | {chunker_name:<10} | {a_score:<10.3f} | "
                f"{step_f1:<10.3f} | {adj_f1:<10.3f} | {kendall:<10.3f}"
            )
    print("-"*len(header))


def main():
    """
    Main function to run the chunking comparison experiment.
    """
    print("Starting Thesis Chunking Experiment...")
    
    # Load base configuration
    # The script will override the chunking strategy for each run
    config = UnifiedConfig()
    
    # Store results for the final table
    all_results = {doc: {} for doc in DOCUMENTS_TO_TEST}

    for doc_name in DOCUMENTS_TO_TEST:
        for chunker_name in CHUNK_METHODS_TO_TEST:
            try:
                report = run_single_experiment(doc_name, chunker_name, config)
                all_results[doc_name][chunker_name] = report
            except Exception as e:
                print(f"FATAL ERROR during experiment: Document='{doc_name}', Chunker='{chunker_name}'")
                print(e)
                all_results[doc_name][chunker_name] = {"error": str(e)}

    print_results_table(all_results)
    print("\nExperiment finished.")


if __name__ == "__main__":
    main()

