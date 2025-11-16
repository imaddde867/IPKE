#!/usr/bin/env python3
"""
Run the full thesis experiment grid (3 documents × 3 chunking methods).

For each combination the script:
1. Uploads the document to the respective service (fixed/semantic/DSC).
2. Saves the raw API payload to results/<document>_<method>.json.
3. Logs a concise summary with chunking metrics and confidence scores.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


DEFAULT_DOCUMENTS = [
    "datasets/archive/test_data/text/3m_marine_oem_sop.txt",
    "datasets/archive/test_data/text/DOA_Food_Man_Proc_Stor.txt",
    "datasets/archive/test_data/text/op_firesafety_guideline.txt",
]

METHOD_PORT_DEFAULTS = {
    "fixed": 8000,
    "breakpoint_semantic": 8001,
    "dsc": 8002,
}

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

ENV_TO_FORM_FIELD_MAPPING: Dict[str, Dict[str, str]] = {
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
}

LOGGER = logging.getLogger("thesis_runner")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run thesis experiments across chunking strategies.")
    parser.add_argument(
        "--host",
        default="http://localhost",
        help="Base host where the docker-compose services listen (default: http://localhost).",
    )
    parser.add_argument(
        "--documents",
        nargs="*",
        default=DEFAULT_DOCUMENTS,
        help="List of document paths to process. Defaults to the 3 thesis PDFs.",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory to store JSON outputs (default: results).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=900,
        help="HTTP request timeout in seconds (default: 900).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip combinations whose output file already exists.",
    )
    parser.add_argument(
        "--fixed-port",
        type=int,
        default=METHOD_PORT_DEFAULTS["fixed"],
        help="Port for the fixed chunking service (default: 8000).",
    )
    parser.add_argument(
        "--semantic-port",
        type=int,
        default=METHOD_PORT_DEFAULTS["breakpoint_semantic"],
        help="Port for the breakpoint_semantic service (default: 8001).",
    )
    parser.add_argument(
        "--dsc-port",
        type=int,
        default=METHOD_PORT_DEFAULTS["dsc"],
        help="Port for the DSC chunking service (default: 8002).",
    )
    return parser.parse_args()


def build_method_port_map(args: argparse.Namespace) -> Dict[str, int]:
    return {
        "fixed": args.fixed_port,
        "breakpoint_semantic": args.semantic_port,
        "dsc": args.dsc_port,
    }


def ensure_documents_exist(documents: List[str]) -> List[Path]:
    resolved = []
    for doc in documents:
        path = Path(doc).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {path}")
        resolved.append(path)
    return resolved


def make_service_url(host: str, port: int) -> str:
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
    mapping = ENV_TO_FORM_FIELD_MAPPING.get(method, {})
    return {target: env_overrides[key] for key, target in mapping.items() if key in env_overrides}


def run_single_request(
    session: requests.Session,
    url: str,
    doc_path: Path,
    timeout: int,
    method: str,
    form_overrides: Optional[Dict[str, Any]] = None,
) -> Dict:
    endpoint = f"{url}/extract"
    LOGGER.debug("Posting %s to %s", doc_path, endpoint)
    form_fields = build_chunk_request_fields(method, form_overrides)
    request_data = form_fields or None
    path_obj = Path(doc_path)
    with path_obj.open("rb") as stream:
        files = {"file": (path_obj.name, stream, "text/plain")}
        response = session.post(endpoint, files=files, data=request_data, timeout=timeout)
    try:
        response.raise_for_status()
    except requests.HTTPError:
        print(f"HTTP error {response.status_code}")
        print(response.text[:1000])
        raise
    return response.json()


def save_result(payload: Dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    method_ports = build_method_port_map(args)
    documents = ensure_documents_exist(args.documents)
    output_dir = Path(args.output_dir).expanduser().resolve()
    summaries: List[Dict[str, object]] = []

    session = requests.Session()

    for doc_path in documents:
        for method, port in method_ports.items():
            output_path = output_dir / f"{doc_path.stem}_{method}.json"
            if args.skip_existing and output_path.exists():
                LOGGER.info("Skipping %s (%s) because output exists.", doc_path.name, method)
                continue

            service_url = make_service_url(args.host, port)
            LOGGER.info("Processing %s with %s via %s", doc_path.name, method, service_url)
            start = time.time()
            payload = run_single_request(session, service_url, doc_path, args.timeout, method=method)
            elapsed = time.time() - start

            save_result(payload, output_path)
            quality = payload.get("quality_metrics", {}) or {}
            summary = {
                "document": doc_path.name,
                "method": method,
                "service_url": service_url,
                "result_path": str(output_path),
                "processing_time_api": payload.get("processing_time"),
                "wall_time": round(elapsed, 2),
                "confidence_score": payload.get("confidence_score"),
                "chunk_count": quality.get("chunk_count"),
                "avg_chunk_size": quality.get("avg_chunk_size"),
                "avg_chunk_cohesion": quality.get("avg_chunk_cohesion"),
                "avg_sentences_per_chunk": quality.get("avg_sentences_per_chunk"),
            }
            summaries.append(summary)

            LOGGER.info(
                "Saved %s (confidence %.2f, chunks %s, api %.2fs)",
                output_path.name,
                summary["confidence_score"] or 0,
                summary["chunk_count"],
                summary["processing_time_api"] or 0,
            )

    if summaries:
        summary_path = output_dir / "thesis_summary.json"
        summary_path.write_text(json.dumps(summaries, indent=2))
        LOGGER.info("Wrote summary table to %s", summary_path)
    else:
        LOGGER.warning("No experiments were executed.")


if __name__ == "__main__":
    main()
