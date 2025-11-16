#!/usr/bin/env python3
"""Bundle predictions (and optional gold) into a compact JSON for human eval."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


LOGGER = logging.getLogger("prepare_human_eval_samples")


def parse_systems(values: List[str]) -> List[Tuple[str, Path]]:
    systems: List[Tuple[str, Path]] = []
    for value in values:
        if "=" not in value:
            raise argparse.ArgumentTypeError(f"Expected label=/path, got '{value}'")
        label, path = value.split("=", 1)
        systems.append((label.strip(), Path(path.strip()).expanduser().resolve()))
    return systems


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a compact JSON bundle for human evaluation.")
    parser.add_argument(
        "--system",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Prediction directory for a system (label=/path/to/predictions).",
    )
    parser.add_argument("--output", type=Path, required=True, help="Destination JSON file.")
    parser.add_argument("--gold-dir", type=Path, help="Optional directory with gold JSON files.")
    parser.add_argument("--max-steps", type=int, help="Trim to at most N steps per prediction (best effort).")
    return parser.parse_args()


def maybe_trim(payload: Dict, max_steps: int | None) -> Dict:
    if not max_steps or max_steps <= 0:
        return payload
    trimmed = json.loads(json.dumps(payload))  # cheap deep copy
    graph = trimmed.get("graph")
    if isinstance(graph, dict):
        nodes = graph.get("nodes")
        if isinstance(nodes, list) and len(nodes) > max_steps:
            graph["nodes"] = nodes[:max_steps]
        edges = graph.get("edges")
        if isinstance(edges, list) and len(edges) > max_steps:
            graph["edges"] = edges[:max_steps]
    steps = trimmed.get("steps")
    if isinstance(steps, list) and len(steps) > max_steps:
        trimmed["steps"] = steps[:max_steps]
    return trimmed


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    systems = parse_systems(args.system)
    gold_dir = args.gold_dir.expanduser().resolve() if args.gold_dir else None

    bundle: List[Dict[str, object]] = []
    for label, pred_dir in systems:
        if not pred_dir.exists():
            raise FileNotFoundError(f"Prediction directory not found: {pred_dir}")
        for pred_path in sorted(pred_dir.glob("*.json")):
            with pred_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            entry: Dict[str, object] = {
                "system": label,
                "document": pred_path.stem,
                "prediction_path": str(pred_path),
                "prediction": maybe_trim(payload, args.max_steps),
            }
            if gold_dir:
                gold_path = gold_dir / f"{pred_path.stem}.json"
                if gold_path.exists():
                    with gold_path.open("r", encoding="utf-8") as handle:
                        entry["gold"] = json.load(handle)
                    entry["gold_path"] = str(gold_path)
                else:
                    LOGGER.warning("Gold file missing for %s", pred_path.stem)
            bundle.append(entry)

    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    LOGGER.info("Wrote %d samples to %s", len(bundle), output_path)


if __name__ == "__main__":
    main()
