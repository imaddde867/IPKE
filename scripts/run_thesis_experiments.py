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
from typing import Dict, List

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


def run_single_request(
    session: requests.Session,
    url: str,
    doc_path: Path,
    timeout: int,
) -> Dict:
    endpoint = f"{url}/extract"
    LOGGER.debug("Posting %s to %s", doc_path, endpoint)
    with doc_path.open("rb") as stream:
        files = {"file": (doc_path.name, stream, "application/pdf")}
        response = session.post(endpoint, files=files, timeout=timeout)
    response.raise_for_status()
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
            payload = run_single_request(session, service_url, doc_path, args.timeout)
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
