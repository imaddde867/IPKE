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
from typing import Any, Dict, List, Optional, Sequence

import requests

# Ensure the repository root is importable when running from scripts/
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.graph.adapter import flat_to_tierb  # noqa: E402

LOGGER = logging.getLogger("experiment_utils")
DEFAULT_HOST = "http://localhost"
DEFAULT_COMPOSE_FILE = REPO_ROOT / "docker-compose.yml"
DEFAULT_OVERRIDE_DIR = REPO_ROOT / "scripts" / "experiments" / ".overrides"
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results" / "experiments"
DEFAULT_GOLD_TIER_A = REPO_ROOT / "datasets" / "archive" / "gold_human"
DEFAULT_GOLD_TIER_B = REPO_ROOT / "datasets" / "archive" / "gold_human_tierb"
DEFAULT_TIMEOUT = 2000
# Canonical documents used across sweeps
DEFAULT_DOCUMENTS = [
    REPO_ROOT / "datasets" / "archive" / "test_data" / "text" / "3m_marine_oem_sop.txt",
    REPO_ROOT / "datasets" / "archive" / "test_data" / "text" / "DOA_Food_Man_Proc_Stor.txt",
    REPO_ROOT / "datasets" / "archive" / "test_data" / "text" / "op_firesafety_guideline.txt",
]

METHOD_FORM_FIELD_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "fixed": {
        "chunking_method": "fixed",
        "chunk_max_chars": 2000,
    },
    "breakpoint_semantic": {
        "chunking_method": "breakpoint_semantic",
        "chunk_max_chars": 2000,
        "sem_lambda": 0.15,
        "sem_window_w": 30,
        "sem_min_sentences_per_chunk": 2,
        "sem_max_sentences_per_chunk": 40,
    },
    "dsc": {
        "chunking_method": "dsc",
        "chunk_max_chars": 2000,
        "dsc_parent_min_sentences": 10,
        "dsc_parent_max_sentences": 120,
        "dsc_delta_window": 25,
        "dsc_threshold_k": 1.0,
        "dsc_use_headings": True,
    },
}

ENV_TO_METHOD_FORM_FIELD_MAPPING: Dict[str, Dict[str, str]] = {
    "fixed": {
        "CHUNK_MAX_CHARS": "chunk_max_chars",
        "CHUNK_STRIDE_CHARS": "chunk_stride_chars",
    },
    "breakpoint_semantic": {
        "CHUNK_MAX_CHARS": "chunk_max_chars",
        "SEM_LAMBDA": "sem_lambda",
        "SEM_WINDOW_W": "sem_window_w",
        "SEM_MIN_SENTENCES_PER_CHUNK": "sem_min_sentences_per_chunk",
        "SEM_MAX_SENTENCES_PER_CHUNK": "sem_max_sentences_per_chunk",
    },
    "dsc": {
        "CHUNK_MAX_CHARS": "chunk_max_chars",
        "DSC_PARENT_MIN_SENTENCES": "dsc_parent_min_sentences",
        "DSC_PARENT_MAX_SENTENCES": "dsc_parent_max_sentences",
        "DSC_DELTA_WINDOW": "dsc_delta_window",
        "DSC_THRESHOLD_K": "dsc_threshold_k",
        "DSC_USE_HEADINGS": "dsc_use_headings",
    },
    "parent_only": {
        "CHUNK_MAX_CHARS": "chunk_max_chars",
        "DSC_PARENT_MIN_SENTENCES": "dsc_parent_min_sentences",
        "DSC_PARENT_MAX_SENTENCES": "dsc_parent_max_sentences",
        "DSC_DELTA_WINDOW": "dsc_delta_window",
        "DSC_THRESHOLD_K": "dsc_threshold_k",
        "DSC_USE_HEADINGS": "dsc_use_headings",
    },
}

# Some document stems do not match their gold filenames; normalise them here.
DOC_ID_OVERRIDES = {
    "3m_marine_oem_sop": "3M_OEM_SOP",
}


def ensure_documents_exist(documents: Sequence[str | Path]) -> List[Path]:
    """Resolve ``documents`` relative to the repo and ensure each exists."""
    resolved: List[Path] = []
    for doc in documents:
        path = Path(doc).expanduser()
        if not path.is_absolute():
            path = (REPO_ROOT / path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {path}")
        resolved.append(path)
    return resolved


def make_service_url(host: str, port: int) -> str:
    """Normalise host/port to a fully qualified base URL."""
    base = host.rstrip("/")
    if base.endswith(f":{port}"):
        return base
    if "://" not in base:
        base = f"http://{base}"
    return f"{base}:{port}"


def _stringify_form_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def build_chunk_request_fields(method: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    base = METHOD_FORM_FIELD_DEFAULTS.get(method, {"chunking_method": method})
    payload: Dict[str, Any] = dict(base)
    payload.setdefault("chunking_method", method)
    if overrides:
        payload.update(overrides)
    return {key: _stringify_form_value(value) for key, value in payload.items() if value is not None}


def env_to_method_form_fields(method: str, env_overrides: Dict[str, str]) -> Dict[str, str]:
    mapping = ENV_TO_METHOD_FORM_FIELD_MAPPING.get(method, {})
    return {target: env_overrides[key] for key, target in mapping.items() if key in env_overrides}


def request_form_fields_from_env(method: str, env_overrides: Optional[Dict[str, str]]) -> Dict[str, str]:
    """Translate docker env overrides into request form fields expected by /extract."""
    if not env_overrides:
        return {}
    return env_to_method_form_fields(method, env_overrides)


def run_single_request(
    session: requests.Session,
    url: str,
    doc_path: Path,
    timeout: int,
    method: str,
    form_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Send ``doc_path`` to ``url`` and return the parsed JSON response."""
    endpoint = f"{url}/extract"
    LOGGER.debug("Posting %s to %s", doc_path, endpoint)
    form_fields = build_chunk_request_fields(method, form_overrides)
    request_data = form_fields or None
    path_obj = Path(doc_path)
    with path_obj.open("rb") as stream:
        files = {"file": (path_obj.name, stream, "text/plain")}
        response = session.post(endpoint, files=files, data=request_data, timeout=timeout)
    response.raise_for_status()
    return response.json()


def save_result(payload: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


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
    use_docker: bool = True


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
    if not service_cfg.use_docker:
        LOGGER.info("Docker disabled; skipping restart for %s.", service_cfg.name)
        return
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


def capture_docker_logs(container_name: str, since: datetime, destination: Path, *, enabled: bool = True) -> None:
    if not enabled:
        LOGGER.info("Docker disabled; skipping log capture for %s.", container_name)
        return
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
    max_attempts: int = 1,
    retry_delay: float = 10.0,
) -> List[Dict[str, object]]:
    """Send documents to the chunking API, optionally retrying on transient errors."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
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
        attempt = 0
        last_error: Optional[Exception] = None
        while attempt < max_attempts:
            attempt += 1
            start = time.time()
            try:
                payload = run_single_request(
                    session,
                    service_url,
                    doc_path,
                    request_timeout,
                    method=method,
                    form_overrides=request_form_fields,
                )
                break
            except requests.Timeout as exc:
                last_error = exc
                LOGGER.warning(
                    "Request for %s timed out (attempt %s/%s, timeout=%ss).",
                    doc_id,
                    attempt,
                    max_attempts,
                    request_timeout,
                )
            except requests.RequestException as exc:  # noqa: PERF203
                last_error = exc
                LOGGER.warning(
                    "Request for %s failed (attempt %s/%s): %s",
                    doc_id,
                    attempt,
                    max_attempts,
                    exc,
                )
            if attempt >= max_attempts:
                raise RuntimeError(f"Failed to process {doc_id} after {max_attempts} attempts.") from last_error
            if retry_delay > 0:
                LOGGER.info("Retrying %s in %.1f seconds...", doc_id, retry_delay)
                time.sleep(retry_delay)
        else:  # pragma: no cover - loop always breaks or raises
            continue
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
        "Phi",
    ]
    for metric in important_metrics:
        row[f"A_{metric}"] = tier_a_metrics.get(metric)
    for metric in [
        "GraphF1",
        "NEXT_EdgeF1",
        "Logic_EdgeF1",
        "ConstraintAttachmentF1",
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
