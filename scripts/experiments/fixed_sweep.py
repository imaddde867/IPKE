#!/usr/bin/env python3
"""Sweep CHUNK_MAX_CHARS (and optional stride) for the fixed chunker."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

# Ensure repository root on sys.path for imports
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

LOGGER = logging.getLogger("fixed_sweep")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a CHUNK_MAX_CHARS sweep on the fixed chunker service.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        nargs="+",
        default=[1000, 1500, 2000, 3000],
        help="CHUNK_MAX_CHARS values to evaluate.",
    )
    parser.add_argument(
        "--stride-values",
        type=int,
        nargs="*",
        help="Optional CHUNK_STRIDE_CHARS values to pair with each max char.",
    )
    parser.add_argument(
        "--documents",
        nargs="+",
        default=DEFAULT_DOCUMENTS,
        help="Document paths to evaluate.",
    )
    parser.add_argument("--host", default="http://localhost", help="Base host for the service.")
    parser.add_argument("--port", type=int, default=8000, help="Service port for ipke-fixed.")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="HTTP timeout passed to /extract.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT / "fixed_sweep",
        help="Directory to store sweep outputs.",
    )
    parser.add_argument(
        "--gold-tier-a",
        type=Path,
        default=DEFAULT_GOLD_TIER_A,
        help="Gold directory for Tier A evaluation.",
    )
    parser.add_argument(
        "--gold-tier-b",
        type=Path,
        default=DEFAULT_GOLD_TIER_B,
        help="Gold directory for Tier B evaluation.",
    )
    parser.add_argument("--service-name", default="ipke-fixed", help="docker-compose service to restart.")
    parser.add_argument("--container-name", default="ipke-fixed-chunking", help="Container name for log capture.")
    parser.add_argument("--compose-file", type=Path, default=REPO_ROOT / "docker-compose.yml", help="Compose file path.")
    parser.add_argument("--health-timeout", type=int, default=240, help="Seconds to wait for /health readiness.")
    parser.add_argument("--health-interval", type=float, default=5.0, help="Seconds between health polls.")
    parser.add_argument("--skip-existing", action="store_true", help="Reuse predictions if they already exist.")
    parser.add_argument("--doc-id-map", type=Path, help="Optional JSON mapping file: doc stem -> gold id.")
    parser.add_argument(
        "--no-docker",
        action="store_true",
        help="Assume the service is already running and skip docker-compose restarts/log capture.",
    )
    return parser.parse_args()


def load_doc_id_overrides(path: Optional[Path]) -> Dict[str, str]:
    if not path:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run_configuration(
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

    if service_cfg.use_docker:
        override_file = write_override_file(service_cfg.name, env_overrides, service_cfg.override_dir)
        restart_service(service_cfg, override_file)
    else:
        LOGGER.info(
            "Docker disabled; assuming %s is already running at %s:%s",
            service_cfg.name,
            service_cfg.host,
            service_cfg.port,
        )
    wait_for_health(service_cfg)

    LOGGER.info("Starting extraction for %s", config_id)
    run_start = time.time()
    log_since = datetime.now(timezone.utc)
    doc_summaries = run_method_extraction(
        method="fixed",
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
    capture_docker_logs(args.container_name, log_since, log_file, enabled=service_cfg.use_docker)

    metadata = {
        "config_id": config_id,
        "method": "fixed",
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
    doc_id_overrides = load_doc_id_overrides(args.doc_id_map)
    documents = ensure_documents_exist(args.documents)
    service_cfg = ServiceConfig(
        name=args.service_name,
        container_name=args.container_name,
        port=args.port,
        host=args.host,
        compose_file=args.compose_file,
        health_timeout=args.health_timeout,
        health_interval=args.health_interval,
        use_docker=not args.no_docker,
    )

    summary_rows: List[Dict[str, object]] = []
    stride_values = args.stride_values or [None]
    for max_chars in args.max_chars:
        for stride in stride_values:
            env: Dict[str, str] = {
                "CHUNKING_METHOD": "fixed",
                "CHUNK_MAX_CHARS": str(max_chars),
            }
            config_id = f"max{max_chars}"
            if stride is not None:
                env["CHUNK_STRIDE_CHARS"] = str(stride)
                config_id += f"_stride{stride}"
            run_configuration(
                config_id=config_id,
                env_overrides=env,
                service_cfg=service_cfg,
                documents=documents,
                args=args,
                doc_id_overrides=doc_id_overrides,
                summary_rows=summary_rows,
            )

    write_summary_table(summary_rows, args.output_root / "fixed_sweep_summary.csv")
    LOGGER.info("Sweep complete. Summary written to %s", args.output_root / "fixed_sweep_summary.csv")


if __name__ == "__main__":
    main()
