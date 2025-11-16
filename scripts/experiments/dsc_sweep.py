#!/usr/bin/env python3
"""Sweep Dual Semantic Chunker (DSC) hyperparameters."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments.experiment_utils import (
    DEFAULT_DOCUMENTS,
    DEFAULT_GOLD_TIER_A,
    DEFAULT_GOLD_TIER_B,
    DEFAULT_RESULTS_ROOT,
    DEFAULT_TIMEOUT,
    EvaluationPaths,
    ServiceConfig,
    aggregate_metric_row,
    capture_docker_logs,
    current_git_sha,
    ensure_documents_exist,
    prepare_run_dirs,
    restart_service,
    run_evaluation_cli,
    run_method_extraction,
    save_metadata,
    wait_for_health,
    write_override_file,
    write_summary_table,
)

LOGGER = logging.getLogger("dsc_sweep")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate DSC parameter sweeps (one dimension at a time).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--documents", nargs="+", default=DEFAULT_DOCUMENTS, help="Documents to process.")
    parser.add_argument("--host", default="http://localhost", help="Base host for the DSC service.")
    parser.add_argument("--port", type=int, default=8002, help="Service port.")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="HTTP timeout.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT / "dsc_sweep",
        help="Directory for results.",
    )
    parser.add_argument("--gold-tier-a", type=Path, default=DEFAULT_GOLD_TIER_A, help="Tier A gold directory.")
    parser.add_argument("--gold-tier-b", type=Path, default=DEFAULT_GOLD_TIER_B, help="Tier B gold directory.")
    parser.add_argument("--min-parent-values", type=int, nargs="+", default=[5, 10, 15], help="DSC_PARENT_MIN_SENTENCES grid.")
    parser.add_argument("--max-parent-values", type=int, nargs="+", default=[80, 120, 160], help="DSC_PARENT_MAX_SENTENCES grid.")
    parser.add_argument("--delta-window-values", type=int, nargs="+", default=[15, 25, 35], help="DSC_DELTA_WINDOW grid.")
    parser.add_argument("--threshold-values", type=float, nargs="+", default=[0.8, 1.0, 1.2], help="DSC_THRESHOLD_K grid.")
    parser.add_argument("--heading-options", nargs="+", default=["true", "false"], help="DSC_USE_HEADINGS options.")
    parser.add_argument("--service-name", default="ipke-dsc", help="docker-compose service to restart.")
    parser.add_argument("--container-name", default="ipke-dsc-chunking", help="Container name for logs.")
    parser.add_argument("--compose-file", type=Path, default=REPO_ROOT / "docker-compose.yml", help="Compose file path.")
    parser.add_argument("--health-timeout", type=int, default=240, help="Seconds to wait for health.")
    parser.add_argument("--health-interval", type=float, default=5.0, help="Seconds between health polls.")
    parser.add_argument("--skip-existing", action="store_true", help="Reuse prediction JSONs if present.")
    parser.add_argument("--doc-id-map", type=Path, help="Optional JSON mapping doc stem -> gold id.")
    return parser.parse_args()


def load_doc_map(path: Path | None) -> Dict[str, str]:
    if not path:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def execute_run(
    config_id: str,
    env_overrides: Dict[str, str],
    service_cfg: ServiceConfig,
    documents: List[Path],
    args: argparse.Namespace,
    doc_id_overrides: Dict[str, str],
    summary_rows: List[Dict[str, object]],
) -> None:
    run_dir = args.output_root / config_id
    run_paths: EvaluationPaths = prepare_run_dirs(run_dir)
    override_file = write_override_file(service_cfg.name, env_overrides, service_cfg.override_dir)
    restart_service(service_cfg, override_file)
    wait_for_health(service_cfg)

    LOGGER.info("Running DSC config %s", config_id)
    run_start = time.time()
    log_since = datetime.now(timezone.utc)
    doc_summaries = run_method_extraction(
        method="dsc",
        service_cfg=service_cfg,
        documents=documents,
        run_paths=run_paths,
        request_timeout=args.timeout,
        skip_existing=args.skip_existing,
        doc_id_overrides=doc_id_overrides,
    )
    tier_a_metrics = run_evaluation_cli("A", args.gold_tier_a, run_paths.predictions_dir, run_paths.metrics_dir / "tier_a.json")
    tier_b_metrics = run_evaluation_cli("B", args.gold_tier_b, run_paths.tierb_dir, run_paths.metrics_dir / "tier_b.json")
    wall_time = time.time() - run_start
    log_file = run_paths.logs_dir / "docker.log"
    capture_docker_logs(args.container_name, log_since, log_file)

    metadata = {
        "config_id": config_id,
        "method": "dsc",
        "env_overrides": env_overrides,
        "documents": [str(path) for path in documents],
        "git_sha": current_git_sha(),
        "command": " ".join(sys.argv),
        "service": service_cfg.name,
        "container_name": args.container_name,
        "host": service_cfg.host,
        "port": service_cfg.port,
        "request_timeout": args.timeout,
        "wall_time_seconds": wall_time,
        "tier_a_report": str(run_paths.metrics_dir / "tier_a.json"),
        "tier_b_report": str(run_paths.metrics_dir / "tier_b.json"),
        "docker_log": str(log_file),
        "doc_summaries": doc_summaries,
    }
    save_metadata(run_paths, metadata)
    summary_rows.append(
        aggregate_metric_row(
            config_id=config_id,
            parameters=env_overrides,
            tier_a_metrics=tier_a_metrics,
            tier_b_metrics=tier_b_metrics,
            wall_time=wall_time,
        )
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args.output_root.mkdir(parents=True, exist_ok=True)
    documents = ensure_documents_exist(args.documents)
    doc_id_overrides = load_doc_map(args.doc_id_map)
    service_cfg = ServiceConfig(
        name=args.service_name,
        container_name=args.container_name,
        port=args.port,
        host=args.host,
        compose_file=args.compose_file,
        health_timeout=args.health_timeout,
        health_interval=args.health_interval,
    )

    defaults = {
        "CHUNKING_METHOD": "dsc",
        "DSC_PARENT_MIN_SENTENCES": "10",
        "DSC_PARENT_MAX_SENTENCES": "120",
        "DSC_DELTA_WINDOW": "25",
        "DSC_THRESHOLD_K": "1.0",
        "DSC_USE_HEADINGS": "true",
    }
    grids = [
        ("DSC_PARENT_MIN_SENTENCES", [str(v) for v in args.min_parent_values]),
        ("DSC_PARENT_MAX_SENTENCES", [str(v) for v in args.max_parent_values]),
        ("DSC_DELTA_WINDOW", [str(v) for v in args.delta_window_values]),
        ("DSC_THRESHOLD_K", [str(v) for v in args.threshold_values]),
        ("DSC_USE_HEADINGS", [str(v).lower() for v in args.heading_options]),
    ]
    summary_rows: List[Dict[str, object]] = []
    seen: set[str] = set()
    for param_name, values in grids:
        for value in values:
            env = defaults.copy()
            env[param_name] = value
            config_id = f"{param_name.lower()}_{value}"
            if config_id in seen:
                continue
            seen.add(config_id)
            execute_run(
                config_id=config_id,
                env_overrides=env,
                service_cfg=service_cfg,
                documents=documents,
                args=args,
                doc_id_overrides=doc_id_overrides,
                summary_rows=summary_rows,
            )

    summary_path = args.output_root / "dsc_sweep_summary.csv"
    write_summary_table(summary_rows, summary_path)
    LOGGER.info("DSC sweep finished. Summary stored at %s", summary_path)


if __name__ == "__main__":
    main()
