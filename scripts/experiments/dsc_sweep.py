#!/usr/bin/env python3
"""Sweep Dual Semantic Chunker (DSC) hyperparameters."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

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
    request_form_fields_from_env,
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
    parser.add_argument("--timeout", type=int, default=7200, help="HTTP timeout (seconds).")
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
    parser.add_argument(
        "--request-retries",
        type=int,
        default=2,
        help="Number of times to retry each /extract request after a timeout or connection error.",
    )
    parser.add_argument(
        "--request-retry-wait",
        type=float,
        default=15.0,
        help="Seconds to wait between request retries.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop the sweep after the first configuration failure instead of continuing.",
    )
    parser.add_argument(
        "--no-docker",
        action="store_true",
        help="Assume the service is already running and skip docker-compose restarts/log capture.",
    )
    return parser.parse_args()


def load_doc_map(path: Path | None) -> Dict[str, str]:
    if not path:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def process_single_run(
    config_id: str,
    env_overrides: Dict[str, str],
    service_cfg: ServiceConfig,
    documents: List[Path],
    args: argparse.Namespace,
    doc_id_overrides: Dict[str, str],
) -> Dict[str, object]:
    run_dir = args.output_root / config_id
    os.makedirs(run_dir, exist_ok=True)
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

    LOGGER.info("Running DSC config %s", config_id)
    run_start = time.time()
    log_since = datetime.now(timezone.utc)
    form_fields = request_form_fields_from_env("dsc", env_overrides)
    doc_summaries = run_method_extraction(
        method="dsc",
        service_cfg=service_cfg,
        documents=documents,
        run_paths=run_paths,
        request_timeout=args.timeout,
        skip_existing=args.skip_existing,
        doc_id_overrides=doc_id_overrides,
        request_form_fields=form_fields,
        max_attempts=max(1, args.request_retries + 1),
        retry_delay=args.request_retry_wait,
    )
    tier_a_metrics = run_evaluation_cli("A", args.gold_tier_a, run_paths.predictions_dir, run_paths.metrics_dir / "tier_a.json")
    tier_b_metrics = run_evaluation_cli("B", args.gold_tier_b, run_paths.tierb_dir, run_paths.metrics_dir / "tier_b.json")
    wall_time = time.time() - run_start
    log_file = run_paths.logs_dir / "docker.log"
    capture_docker_logs(args.container_name, log_since, log_file, enabled=service_cfg.use_docker)

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
    return aggregate_metric_row(
        config_id=config_id,
        parameters=env_overrides,
        tier_a_metrics=tier_a_metrics,
        tier_b_metrics=tier_b_metrics,
        wall_time=wall_time,
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args.output_root.mkdir(parents=True, exist_ok=True)
    documents = ensure_documents_exist(args.documents)
    for label, gold_path in (("Tier A", args.gold_tier_a), ("Tier B", args.gold_tier_b)):
        if not gold_path.exists():
            raise FileNotFoundError(f"{label} gold directory not found: {gold_path}")
    doc_id_overrides = load_doc_map(args.doc_id_map)
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
    failures: List[Tuple[str, str]] = []
    future_to_config: Dict[concurrent.futures.Future[Dict[str, object]], str] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        for param_name, values in grids:
            for value in values:
                env = defaults.copy()
                env[param_name] = value
                config_id = f"{param_name.lower()}_{value}"
                if config_id in seen:
                    continue
                seen.add(config_id)
                future = executor.submit(
                    process_single_run,
                    config_id,
                    env,
                    service_cfg,
                    documents,
                    args,
                    doc_id_overrides,
                )
                future_to_config[future] = config_id

        for future in concurrent.futures.as_completed(future_to_config):
            config_id = future_to_config[future]
            try:
                summary_rows.append(future.result())
            except KeyboardInterrupt:
                executor.shutdown(wait=False, cancel_futures=True)
                raise
            except Exception as exc:  # noqa: PERF203
                LOGGER.error("Configuration %s failed: %s", config_id, exc)
                LOGGER.debug("Detailed error for %s", config_id, exc_info=True)
                failures.append((config_id, str(exc)))
                if args.fail_fast:
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise

    summary_path = args.output_root / "dsc_sweep_summary.csv"
    write_summary_table(summary_rows, summary_path)
    LOGGER.info("DSC sweep finished. Summary stored at %s", summary_path)
    if failures:
        failed_ids = ", ".join(cfg for cfg, _ in failures)
        LOGGER.warning("Failed configurations (%d): %s", len(failures), failed_ids)


if __name__ == "__main__":
    main()
