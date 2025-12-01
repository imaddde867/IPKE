#!/usr/bin/env python3
"""Shared helpers for chunking sweep experiments."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import requests

# Ensure the repository root is importable when running from scripts/
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_thesis_experiments import (  # noqa: E402
    DEFAULT_DOCUMENTS,
    ensure_documents_exist,
    make_service_url,
    run_single_request,
    save_result,
    env_to_method_form_fields,
)
from src.graph.adapter import flat_to_tierb  # noqa: E402

LOGGER = logging.getLogger("experiment_utils")
DEFAULT_HOST = "http://localhost"
DEFAULT_COMPOSE_FILE = REPO_ROOT / "docker-compose.yml"
DEFAULT_OVERRIDE_DIR = REPO_ROOT / "scripts" / "experiments" / ".overrides"
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results" / "experiments"
DEFAULT_GOLD_TIER_A = REPO_ROOT / "datasets" / "archive" / "gold_human"
DEFAULT_GOLD_TIER_B = REPO_ROOT / "datasets" / "archive" / "gold_human_tierb"
DEFAULT_TIMEOUT = 2000

# Some document stems do not match their gold filenames; normalise them here.
DOC_ID_OVERRIDES = {
    "3m_marine_oem_sop": "3M_OEM_SOP",
}


def request_form_fields_from_env(method: str, env_overrides: Optional[Dict[str, str]]) -> Dict[str, str]:
    """Translate docker env overrides into request form fields expected by /extract."""
    if not env_overrides:
        return {}
    return env_to_method_form_fields(method, env_overrides)


@dataclass
class ServiceConfig:
    """Describe the docker-compose service used for a sweep."""

    name: str
    container_name: str
    port: int
    host: str = DEFAULT_HOST
    compose_file: Path = DEFAULT_COMPOSE_FILE
    override_dir: Path = DEFAULT_OVERRIDE_DIR
    health_path: str = "/health"
    health_timeout: int = 240
    health_interval: float = 5.0


@dataclass
class EvaluationPaths:
    """Convenience bundle for directories produced per configuration."""

    run_dir: Path
    predictions_dir: Path
    tierb_dir: Path
    metrics_dir: Path
    logs_dir: Path
    metadata_path: Path = field(init=False)
    summary_path: Path = field(init=False)

    def __post_init__(self) -> None:
        self.metadata_path = self.run_dir / "metadata.json"
        self.summary_path = self.run_dir / "prediction_summary.json"


def ensure_override_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def canonical_doc_id(doc_path: Path, overrides: Optional[Dict[str, str]] = None) -> str:
    overrides = overrides or {}
    stem = doc_path.stem
    return overrides.get(stem, DOC_ID_OVERRIDES.get(stem, stem))


def write_override_file(service: str, env_overrides: Dict[str, str], override_dir: Path) -> Path:
    override_dir = ensure_override_dir(override_dir)
    override_path = override_dir / f"{service}_override.yml"
    lines = ["services:", f"  {service}:", "    environment:"]
    for key, value in env_overrides.items():
        lines.append(f'      {key}: "{value}"')
    override_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return override_path


def restart_service(service_cfg: ServiceConfig, override_file: Path) -> None:
    cmd = [
        "docker",
        "compose",
        "-f",
        str(service_cfg.compose_file),
        "-f",
        str(override_file),
        "up",
        "-d",
        "--force-recreate",
        service_cfg.name,
    ]
    LOGGER.info("Restarting %s via docker compose", service_cfg.name)
    subprocess.run(cmd, check=True)


def wait_for_health(service_cfg: ServiceConfig) -> None:
    url = f"{service_cfg.host.rstrip('/') }:{service_cfg.port}{service_cfg.health_path}"
    LOGGER.info("Waiting for %s to report healthy at %s", service_cfg.name, url)
    deadline = time.time() + service_cfg.health_timeout
    last_error: Optional[str] = None
    while time.time() < deadline:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                LOGGER.info("%s is healthy.", service_cfg.name)
                return
            last_error = f"HTTP {response.status_code}"
        except requests.RequestException as exc:  # noqa: PERF203
            last_error = str(exc)
        time.sleep(service_cfg.health_interval)
    raise RuntimeError(f"Service {service_cfg.name} did not become healthy within timeout. Last error: {last_error}")


def capture_docker_logs(container_name: str, since: datetime, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    timestamp = since.astimezone(timezone.utc).isoformat()
    cmd = ["docker", "logs", container_name, "--since", timestamp]
    LOGGER.info("Collecting logs for %s since %s", container_name, timestamp)
    with destination.open("w", encoding="utf-8") as handle:
        try:
            subprocess.run(cmd, check=True, stdout=handle, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as exc:  # pragma: no cover - best effort logging
            handle.write(f"Failed to capture logs: {exc}\n")


def current_git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return "unknown"


def prepare_run_dirs(run_dir: Path) -> EvaluationPaths:
    predictions = run_dir / "predictions"
    tierb = run_dir / "tierb"
    metrics = run_dir / "metrics"
    logs_dir = run_dir / "logs"
    for path in (predictions, tierb, metrics, logs_dir):
        path.mkdir(parents=True, exist_ok=True)
    return EvaluationPaths(run_dir=run_dir, predictions_dir=predictions, tierb_dir=tierb, metrics_dir=metrics, logs_dir=logs_dir)


def run_method_extraction(
    method: str,
    service_cfg: ServiceConfig,
    documents: Sequence[Path],
    run_paths: EvaluationPaths,
    request_timeout: int = DEFAULT_TIMEOUT,
    skip_existing: bool = False,
    doc_id_overrides: Optional[Dict[str, str]] = None,
    request_form_fields: Optional[Dict[str, object]] = None,
) -> List[Dict[str, object]]:
    session = requests.Session()
    service_url = make_service_url(service_cfg.host, service_cfg.port)
    summaries: List[Dict[str, object]] = []

    for doc_path in documents:
        doc_id = canonical_doc_id(doc_path, doc_id_overrides)
        output_path = run_paths.predictions_dir / f"{doc_id}.json"
        tierb_path = run_paths.tierb_dir / f"{doc_id}.json"

        if skip_existing and output_path.exists():
            LOGGER.info("Skipping %s; prediction already exists.", doc_id)
            with output_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if not tierb_path.exists():
                tierb_payload = flat_to_tierb(payload)
                tierb_path.write_text(json.dumps(tierb_payload, indent=2), encoding="utf-8")
            summaries.append({"document": doc_id, "status": "skipped"})
            continue

        LOGGER.info("Requesting %s via %s", doc_id, service_url)
        start = time.time()
        payload = run_single_request(
            session,
            service_url,
            doc_path,
            request_timeout,
            method=method,
            form_overrides=request_form_fields,
        )
        elapsed = time.time() - start
        save_result(payload, output_path)
        tierb_payload = flat_to_tierb(payload)
        tierb_path.write_text(json.dumps(tierb_payload, indent=2), encoding="utf-8")
        quality = payload.get("quality_metrics", {}) or {}
        summaries.append(
            {
                "document": doc_id,
                "method": method,
                "service_url": service_url,
                "output_path": str(output_path),
                "tierb_path": str(tierb_path),
                "wall_time": round(elapsed, 2),
                "processing_time_api": payload.get("processing_time"),
                "confidence_score": payload.get("confidence_score"),
                "chunk_count": quality.get("chunk_count"),
                "avg_chunk_size": quality.get("avg_chunk_size"),
                "avg_chunk_cohesion": quality.get("avg_chunk_cohesion"),
                "avg_sentences_per_chunk": quality.get("avg_sentences_per_chunk"),
            }
        )

    run_paths.summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    return summaries


def run_evaluation_cli(
    tier: str,
    gold_dir: Path,
    pred_dir: Path,
    out_path: Path,
) -> Dict[str, float]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "src" / "evaluation" / "metrics.py"),
        "--gold_dir",
        str(gold_dir),
        "--pred_dir",
        str(pred_dir),
        "--tier",
        tier,
        "--out_file",
        str(out_path),
    ]
    LOGGER.info("Running %s evaluation -> %s", tier, out_path)
    subprocess.run(cmd, check=True)
    with out_path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    return report.get("macro_avg", {})


def aggregate_metric_row(
    config_id: str,
    parameters: Dict[str, str],
    tier_a_metrics: Dict[str, float],
    tier_b_metrics: Dict[str, float],
    wall_time: float,
) -> Dict[str, object]:
    row: Dict[str, object] = {
        "config_id": config_id,
        "wall_time": round(wall_time, 2),
    }
    for key, value in parameters.items():
        row[key.lower()] = value
    important_metrics = [
        "StepF1",
        "AdjacencyF1",
        "Kendall",
        "ConstraintCoverage",
        "ConstraintAttachmentF1",
        "A_score",
    ]
    for metric in important_metrics:
        row[f"A_{metric}"] = tier_a_metrics.get(metric)
    for metric in [
        "GraphF1",
        "NEXT_EdgeF1",
        "Logic_EdgeF1",
        "ConstraintAttachmentF1",
        "B_score",
    ]:
        row[f"B_{metric}"] = tier_b_metrics.get(metric)
    return row


def write_summary_table(rows: List[Dict[str, object]], destination: Path) -> None:
    if not rows:
        return
    headers: List[str] = sorted({key for row in rows for key in row.keys()})
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        handle.write(",".join(headers) + "\n")
        for row in rows:
            values = []
            for header in headers:
                value = row.get(header)
                values.append("" if value is None else str(value))
            handle.write(",".join(values) + "\n")


def save_metadata(run_paths: EvaluationPaths, metadata: Dict[str, object]) -> None:
    run_paths.metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


__all__ = [
    "ServiceConfig",
    "EvaluationPaths",
    "DEFAULT_DOCUMENTS",
    "DEFAULT_GOLD_TIER_A",
    "DEFAULT_GOLD_TIER_B",
    "DEFAULT_RESULTS_ROOT",
    "DEFAULT_TIMEOUT",
    "ensure_documents_exist",
    "canonical_doc_id",
    "current_git_sha",
    "prepare_run_dirs",
    "write_override_file",
    "restart_service",
    "wait_for_health",
    "capture_docker_logs",
    "run_method_extraction",
    "run_evaluation_cli",
    "aggregate_metric_row",
    "write_summary_table",
    "save_metadata",
]
