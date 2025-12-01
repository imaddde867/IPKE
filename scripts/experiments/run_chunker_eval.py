#!/usr/bin/env python3
"""Run a single chunker + prompt configuration and evaluate Tier A/B."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


# Ensure repository root on sys.path for imports
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments.experiment_utils import (  # noqa: E402
    DEFAULT_DOCUMENTS,
    DEFAULT_GOLD_TIER_A,
    DEFAULT_GOLD_TIER_B,
    DEFAULT_OVERRIDE_DIR,
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
)


LOGGER = logging.getLogger("run_chunker_eval")
SUMMARY_FILENAME = "summary_row.json"


def parse_key_values(pairs: List[str]) -> Dict[str, str]:
    values: Dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise argparse.ArgumentTypeError(f"Expected KEY=VALUE, got '{pair}'")
        key, value = pair.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Restart a chunking service with env overrides, run extraction, and evaluate tiers A/B.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config-id", required=True, help="Identifier for this configuration (used in metadata).")
    parser.add_argument(
        "--method",
        default="fixed",
        help="Chunking method name recorded in metadata (e.g. fixed, breakpoint_semantic, dsc).",
    )
    parser.add_argument("--run-dir", type=Path, required=True, help="Directory where run artifacts will be stored.")
    parser.add_argument("--documents", nargs="+", default=DEFAULT_DOCUMENTS, help="Document paths to evaluate.")
    parser.add_argument("--host", default="http://localhost", help="Base host for the service endpoint.")
    parser.add_argument("--port", type=int, default=8000, help="Port for the service endpoint.")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="HTTP timeout for /extract calls.")
    parser.add_argument("--service-name", default="ipke-fixed", help="docker-compose service to restart.")
    parser.add_argument("--container-name", default="ipke-fixed-chunking", help="Docker container name for log capture.")
    parser.add_argument("--compose-file", type=Path, default=REPO_ROOT / "docker-compose.yml", help="Compose file to use.")
    parser.add_argument(
        "--override-dir",
        type=Path,
        default=DEFAULT_OVERRIDE_DIR,
        help="Directory where temporary docker-compose override files are written.",
    )
    parser.add_argument("--health-timeout", type=int, default=240, help="Seconds to wait for service /health to succeed.")
    parser.add_argument("--health-interval", type=float, default=5.0, help="Seconds between health checks.")
    parser.add_argument("--skip-existing", action="store_true", help="Reuse predictions if JSON outputs already exist.")
    parser.add_argument("--prompt-mode", help="Optional prompt mode env override (sets PROMPT_MODE).")
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Additional environment overrides passed to the docker service.",
    )
    parser.add_argument(
        "--summary-param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Key/value tags recorded in the aggregated summary row.",
    )
    parser.add_argument(
        "--gold-tier-a",
        type=Path,
        default=DEFAULT_GOLD_TIER_A,
        help="Directory containing Tier A gold annotations.",
    )
    parser.add_argument(
        "--gold-tier-b",
        type=Path,
        default=DEFAULT_GOLD_TIER_B,
        help="Directory containing Tier B gold annotations.",
    )
    parser.add_argument("--doc-id-map", type=Path, help="Optional JSON mapping: doc stem -> gold id override.")
    parser.add_argument(
        "--no-docker",
        action="store_true",
        help="Assume the service is already running and skip docker-compose restarts/log capture.",
    )
    return parser.parse_args()


def load_doc_id_overrides(path: Path | None) -> Dict[str, str]:
    if not path:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    run_dir = args.run_dir.expanduser().resolve()
    documents = ensure_documents_exist(args.documents)
    doc_id_overrides = load_doc_id_overrides(args.doc_id_map)

    env_overrides = parse_key_values(args.env)
    env_overrides.setdefault("CHUNKING_METHOD", args.method)
    if args.prompt_mode:
        env_overrides["PROMPT_MODE"] = args.prompt_mode

    summary_params = parse_key_values(args.summary_param)

    service_cfg = ServiceConfig(
        name=args.service_name,
        container_name=args.container_name,
        port=args.port,
        host=args.host,
        compose_file=args.compose_file,
        override_dir=args.override_dir,
        health_timeout=args.health_timeout,
        health_interval=args.health_interval,
        use_docker=not args.no_docker,
    )

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

    LOGGER.info("Running extraction for %s", args.config_id)
    run_start = time.time()
    log_since = datetime.now(timezone.utc)
    form_fields = request_form_fields_from_env(args.method, env_overrides)
    doc_summaries = run_method_extraction(
        method=args.method,
        service_cfg=service_cfg,
        documents=documents,
        run_paths=run_paths,
        request_timeout=args.timeout,
        skip_existing=args.skip_existing,
        doc_id_overrides=doc_id_overrides,
        request_form_fields=form_fields,
    )

    tier_a_metrics = run_evaluation_cli("A", args.gold_tier_a, run_paths.predictions_dir, run_paths.metrics_dir / "tier_a.json")
    tier_b_metrics = run_evaluation_cli("B", args.gold_tier_b, run_paths.tierb_dir, run_paths.metrics_dir / "tier_b.json")
    wall_time = time.time() - run_start

    log_file = run_paths.logs_dir / "docker.log"
    capture_docker_logs(args.container_name, log_since, log_file, enabled=service_cfg.use_docker)

    metadata = {
        "config_id": args.config_id,
        "method": args.method,
        "env_overrides": env_overrides,
        "documents": [str(path) for path in documents],
        "git_sha": current_git_sha(),
        "host": args.host,
        "port": args.port,
        "service": args.service_name,
        "container_name": args.container_name,
        "request_timeout": args.timeout,
        "command": " ".join(sys.argv),
        "wall_time_seconds": wall_time,
        "tier_a_report": str(run_paths.metrics_dir / "tier_a.json"),
        "tier_b_report": str(run_paths.metrics_dir / "tier_b.json"),
        "docker_log": str(log_file),
        "doc_summaries": doc_summaries,
    }
    save_metadata(run_paths, metadata)

    parameters: Dict[str, str] = {"method": args.method, "config": args.config_id}
    if args.prompt_mode:
        parameters["prompt_mode"] = args.prompt_mode
    parameters.update(summary_params)

    summary_row = aggregate_metric_row(
        config_id=args.config_id,
        parameters=parameters,
        tier_a_metrics=tier_a_metrics,
        tier_b_metrics=tier_b_metrics,
        wall_time=wall_time,
    )

    summary_path = run_paths.run_dir / SUMMARY_FILENAME
    summary_path.write_text(json.dumps(summary_row, indent=2), encoding="utf-8")
    LOGGER.info("Wrote summary row to %s", summary_path)


if __name__ == "__main__":
    main()
