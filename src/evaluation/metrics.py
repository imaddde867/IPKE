import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tqdm import tqdm

from src.evaluation.core import (
    compute_macro_average,
    load_json,
    load_pairs,
)
from src.evaluation.alignment import (
    EmbeddingCache,
    TextPreprocessor,
    prepare_evaluator,
)
from src.evaluation.metrics_tier_a import evaluate_tier_a_document
from src.evaluation.metrics_tier_b import evaluate_tier_b_document

# Backward-compatible re-exports
from src.evaluation.core import (  # noqa: F401
    HEADLINE_METRICS_ORDER,
    CONSTRAINT_LINK_KEYS,
    collect_constraint_links,
    compute_macro_average,
    compute_prf,
    extract_constraint_text,
    extract_node_label,
    extract_step_text,
    load_json,
    load_pairs,
    normalize_field,
    round3,
    safe_ratio,
)
from src.evaluation.alignment import (  # noqa: F401
    AlignmentResult,
    EmbeddingCache,
    TextPreprocessor,
    align_by_text,
    alignment_to_id_map,
    cosine_similarity_matrix,
    prepare_evaluator,
)
from src.evaluation.metrics_tier_a import (  # noqa: F401
    compute_phi,
    compute_step_metrics,
    derive_sequence_adjacency,
    derive_sequence_order,
    evaluate_tier_a_document,
    NESTED_CONSTRAINT_KINDS,
    normalize_doc_constraints,
    tier_a_constraints_metrics,
)
from src.evaluation.metrics_tier_b import (  # noqa: F401
    adjacency_from_edges,
    align_constraint_nodes,
    build_step_lookup,
    build_tier_b_graph,
    compute_constraint_attachment_f1_tier_b,
    compute_edge_metrics,
    derive_order_from_next_edges,
    evaluate_tier_b_document,
    expand_node_mapping,
    get_edge_endpoints,
    graph_to_smatch_triples,
    is_constraint_type,
    is_step_type,
    sort_steps_for_graph,
)


def pretty_print(results: Dict[str, Dict[str, Optional[float]]]) -> None:
    metrics_present: List[str] = []
    for metric in HEADLINE_METRICS_ORDER:
        if any(metric in doc_metrics for doc_metrics in results.values()):
            metrics_present.append(metric)

    header = ["doc_id"] + metrics_present
    column_widths = {key: max(len(key), 10) for key in header}

    rows: List[List[str]] = []
    for doc_id, metrics in results.items():
        row = [doc_id]
        for metric in metrics_present:
            value = metrics.get(metric)
            if value is None:
                cell = "-"
            else:
                cell = f"{value:.3f}"
            column_widths[metric] = max(column_widths[metric], len(cell))
            row.append(cell)
        column_widths["doc_id"] = max(column_widths["doc_id"], len(doc_id))
        rows.append(row)

    def format_row(values: List[str]) -> str:
        return "  ".join(
            value.ljust(column_widths[key])
            for value, key in zip(values, header)
        )

    print(format_row(header))
    print("-" * sum(column_widths[key] + 2 for key in header))
    for row in rows:
        print(format_row(row))


def run_evaluation(
    gold: Any,
    pred: Any,
    *,
    tiers: Iterable[str] = ("A", "B"),
    threshold: float = 0.75,
    preprocessor: Optional[TextPreprocessor] = None,
    embedder: Optional[EmbeddingCache] = None,
    spacy_model: str = "en_core_web_sm",
    embedding_model: str = "all-mpnet-base-v2",
    device: Optional[str] = None,
    preserve_conflicts: bool = False,
    strict_paper: bool = False,
) -> Dict[str, Optional[float]]:
    """Evaluate a single prediction/gold pair for the requested tiers."""
    gold_doc = load_json(Path(gold)) if isinstance(gold, (str, Path)) else gold
    pred_doc = load_json(Path(pred)) if isinstance(pred, (str, Path)) else pred
    tiers_upper = {tier.upper() for tier in tiers} or {"A", "B"}
    evaluator_pre = preprocessor or TextPreprocessor(spacy_model)
    evaluator_emb = embedder or EmbeddingCache(model_name=embedding_model, device=device)

    metrics: Dict[str, Optional[float]] = {}
    step_alignment_map: Optional[Dict[str, str]] = None
    if "A" in tiers_upper:
        tier_a_result = evaluate_tier_a_document(
            gold_doc,
            pred_doc,
            evaluator_pre,
            evaluator_emb,
            threshold,
            return_alignment_map="B" in tiers_upper,
            strict_paper=strict_paper,
        )
        if isinstance(tier_a_result, tuple):
            tier_a_metrics, step_alignment_map = tier_a_result
        else:
            tier_a_metrics = tier_a_result
        metrics.update(tier_a_metrics)
    if "B" in tiers_upper:
        tier_b_metrics = evaluate_tier_b_document(
            gold_doc,
            pred_doc,
            evaluator_pre,
            evaluator_emb,
            threshold,
            step_id_map=step_alignment_map,
        )
        for key, value in tier_b_metrics.items():
            if preserve_conflicts and key in metrics:
                metrics[f"{key}_TierB"] = value
            else:
                metrics[key] = value
    return metrics


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate procedural extraction quality.")
    parser.add_argument("--run-dir", type=str, default=None, help="Run directory containing predictions/gold.")
    parser.add_argument("--gold_dir", type=str, default=None, help="Directory or file containing gold JSONs.")
    parser.add_argument("--pred_dir", type=str, default=None, help="Directory or file containing prediction JSONs.")
    parser.add_argument("--tier", type=str, choices=["A", "B", "both"], default="both", help="Which tier to evaluate.")
    parser.add_argument("--threshold", type=float, default=0.75, help="Similarity threshold for alignment.")
    parser.add_argument("--subset", type=float, default=None, help="Optional fraction to sample for sanity checks.")
    parser.add_argument("--seed", type=int, default=13, help="Random seed used when sampling a subset.")
    parser.add_argument("--out_file", type=str, default=None, help="Path to save JSON report.")
    parser.add_argument("--spacy_model", type=str, default="en_core_web_sm", help="spaCy model name for lemmatization.")
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="all-mpnet-base-v2",
        help="SentenceTransformer model to use for semantic alignment.",
    )
    parser.add_argument("--device", type=str, default=None, help="SentenceTransformer device override.")
    parser.add_argument(
        "--strict-paper",
        action="store_true",
        help="Reject non-IPKE-Bench constraint taxonomy/enforcement before scoring.",
    )
    args = parser.parse_args(argv)
    if not args.run_dir and (args.gold_dir is None or args.pred_dir is None):
        parser.error("Provide --run-dir or both --gold_dir and --pred_dir.")
    return args


def _guess_path(root: Path, candidates: Sequence[str]) -> Optional[Path]:
    for name in candidates:
        candidate = root / name
        if candidate.exists():
            return candidate
        json_candidate = root / f"{name}.json"
        if json_candidate.exists():
            return json_candidate
    return None


def resolve_io_paths(args: argparse.Namespace) -> Tuple[Path, Path, Path]:
    run_dir = Path(args.run_dir).resolve() if args.run_dir else None
    gold_path = Path(args.gold_dir).resolve() if args.gold_dir else None
    pred_path = Path(args.pred_dir).resolve() if args.pred_dir else None

    if run_dir:
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        if gold_path is None:
            gold_path = _guess_path(run_dir, ["gold", "references", "labels", "ground_truth"])
        if pred_path is None:
            pred_path = _guess_path(run_dir, ["predictions", "preds", "outputs", "results"])
        out_file = Path(args.out_file) if args.out_file else run_dir / f"evaluation_{args.tier.lower()}.json"
    else:
        out_file = Path(args.out_file or "evaluation_report.json")

    if gold_path is None or pred_path is None:
        raise FileNotFoundError("Unable to resolve gold/prediction paths; specify --gold_dir and --pred_dir explicitly.")
    return gold_path, pred_path, out_file


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        gold_dir, pred_dir, out_file = resolve_io_paths(args)
    except FileNotFoundError as exc:
        logging.error("%s", exc)
        return 1
    args.gold_dir = str(gold_dir)
    args.pred_dir = str(pred_dir)
    args.out_file = str(out_file)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    pairs = load_pairs(args.gold_dir, args.pred_dir, args.subset, args.seed)
    if not pairs:
        logging.error("No document pairs to evaluate.")
        return 1

    preprocessor, embedder = prepare_evaluator(args.spacy_model, args.embedding_model, device=args.device)

    doc_results: Dict[str, Dict[str, Optional[float]]] = {}
    if args.tier == "both":
        tier_tuple: Tuple[str, ...] = ("A", "B")
    else:
        tier_tuple = (args.tier,)

    for doc_id, gold_doc, pred_doc in tqdm(pairs, desc="Evaluating documents"):
        metrics = run_evaluation(
            gold_doc,
            pred_doc,
            tiers=tier_tuple,
            threshold=args.threshold,
            preprocessor=preprocessor,
            embedder=embedder,
            strict_paper=args.strict_paper,
        )
        doc_results[doc_id] = metrics

    macro = compute_macro_average(doc_results)
    doc_results["macro_avg"] = macro

    output_path = Path(args.out_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(doc_results, handle, indent=2)
    logging.info("Saved evaluation report to %s", output_path)

    pretty_print(doc_results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
