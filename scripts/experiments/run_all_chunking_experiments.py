#!/usr/bin/env python3
"""Master orchestrator that runs every chunking sweep sequentially."""

from __future__ import annotations

import argparse
import csv
import datetime
import importlib
import logging
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib import error as urllib_error
from urllib import request as urllib_request

import requests


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_DOCUMENTS = [
    REPO_ROOT / "datasets" / "archive" / "test_data" / "text" / "3m_marine_oem_sop.txt",
    REPO_ROOT / "datasets" / "archive" / "test_data" / "text" / "DOA_Food_Man_Proc_Stor.txt",
    REPO_ROOT / "datasets" / "archive" / "test_data" / "text" / "op_firesafety_guideline.txt",
]


@dataclass(frozen=True)
class SweepDefinition:
    """Describe how to invoke a sweep script."""

    name: str
    method: str
    script: Path
    output_subdir: str
    summary_filename: str
    base_args: Sequence[str]
    health_port: int

    def build_command(self, output_root: Path, documents: Sequence[Path]) -> List[str]:
        command = [sys.executable, str(self.script)]
        command.extend(self.base_args)
        command.extend(["--output-root", str(output_root)])
        command.append("--documents")
        command.extend(str(doc) for doc in documents)
        command.append("--skip-existing")
        return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all chunking sweeps sequentially with safety checks.")
    parser.add_argument(
        "--host",
        default="http://localhost",
        help="Base host used for /health checks (default: http://localhost).",
    )
    parser.add_argument(
        "--health-timeout",
        type=int,
        default=90,
        help="Seconds to wait for an individual service health check.",
    )
    parser.add_argument(
        "--health-interval",
        type=float,
        default=5.0,
        help="Seconds between health check attempts.",
    )
    parser.add_argument(
        "--documents",
        nargs="+",
        default=[str(path) for path in DEFAULT_DOCUMENTS],
        help="Document paths for every sweep (defaults to the standard three).",
    )
    return parser.parse_args()


def ensure_scripts_importable(modules: Iterable[str]) -> None:
    for module in modules:
        importlib.import_module(module)


def verify_documents_exist(documents: Sequence[str]) -> List[Path]:
    resolved: List[Path] = []
    for doc in documents:
        path = Path(doc).expanduser()
        if not path.is_absolute():
            path = (REPO_ROOT / path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {path}")
        resolved.append(path)
    return resolved


def verify_docker_running() -> None:
    try:
        subprocess.run(
            ["docker", "info"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("Docker CLI is not installed or not on PATH.") from exc
    except subprocess.CalledProcessError as exc:  # pragma: no cover - requires docker at runtime
        raise RuntimeError("Docker daemon does not appear to be running.") from exc


def normalize_host(host: str) -> str:
    stripped = host.rstrip("/")
    if "://" not in stripped:
        stripped = f"http://{stripped}"
    return stripped


def check_service_health(
    host: str,
    port: int,
    timeout: int,
    interval: float,
) -> None:
    base = normalize_host(host)
    if base.endswith(f":{port}"):
        url = f"{base}/health"
    else:
        url = f"{base}:{port}/health"
    deadline = time.time() + timeout
    last_error: Optional[str] = None
    while time.time() < deadline:
        try:
            with urllib_request.urlopen(url, timeout=5) as response:
                if response.status == 200:
                    return
                last_error = f"HTTP {response.status}"
        except urllib_error.URLError as exc:  # pragma: no cover - network required
            last_error = str(exc.reason)
        except Exception as exc:  # pragma: no cover - safety net
            last_error = str(exc)
        time.sleep(interval)
    raise RuntimeError(f"{url} did not report healthy within {timeout}s (last error: {last_error})")


class ErrorLogger:
    """Helper that keeps errors in both run-specific and global logs."""

    def __init__(self, run_specific: Path, global_errors: Path) -> None:
        self.paths = [run_specific, global_errors]
        for path in self.paths:
            path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, message: str) -> None:
        timestamp = datetime.datetime.utcnow().isoformat()
        entry = f"[{timestamp}] {message}\n"
        for path in self.paths:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(entry)


def run_with_logging(command: List[str], log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    cmd_str = shlex.join(command)
    tee_cmd = f"set -o pipefail; {cmd_str} 2>&1 | tee {shlex.quote(str(log_file))}"
    subprocess.run(["bash", "-lc", tee_cmd], check=True)


def run_sweep(
    sweep: SweepDefinition,
    documents: Sequence[Path],
    output_root: Path,
    log_dir: Path,
    host: str,
    health_timeout: int,
    health_interval: float,
    record_error: Callable[[str], None],
) -> bool:
    log_path = log_dir / f"{sweep.name}.log"
    logging.info("Preparing %s sweep.", sweep.name)
    health_ok = True
    try:
        check_service_health(host, sweep.health_port, timeout=health_timeout, interval=health_interval)
    except Exception as exc:  # pragma: no cover - network required
        message = f"{sweep.name} health check failed: {exc}"
        logging.warning(message)
        record_error(message)
        health_ok = False

    command = sweep.build_command(output_root=output_root / sweep.output_subdir, documents=documents)
    logging.info("Starting %s sweep. Logs -> %s", sweep.name, log_path)
    try:
        run_with_logging(command, log_path)
        logging.info("%s sweep finished successfully.", sweep.name)
        return True
    except subprocess.CalledProcessError as exc:
        message = f"{sweep.name} sweep failed with exit code {exc.returncode}. See {log_path}."
        logging.error(message)
        record_error(message)
        return False


def combine_summaries(
    summaries: Sequence[Tuple[str, Path]],
    destination: Path,
    mirror_destination: Path,
    record_error: Callable[[str], None],
) -> bool:
    rows: List[Dict[str, str]] = []
    headers_seen: Dict[str, None] = {}
    for method, summary_path in summaries:
        if not summary_path.exists():
            record_error(f"Summary missing for {method} sweep: {summary_path}")
            continue
        with summary_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                record_error(f"Summary at {summary_path} lacks headers.")
                continue
            for row in reader:
                combined: Dict[str, str] = {"method": method}
                for key, value in row.items():
                    if key is None:
                        continue
                    combined[key] = value or ""
                    headers_seen[key] = None
                rows.append(combined)
    if not rows:
        logging.warning("No summary rows collected; aggregated CSV will not be written.")
        return False

    base_headers = ["method", "config_id", "wall_time"]
    other_headers = [header for header in headers_seen.keys() if header not in base_headers]
    ordered_headers = base_headers + sorted(other_headers)

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered_headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in ordered_headers})

    if mirror_destination != destination:
        mirror_destination.parent.mkdir(parents=True, exist_ok=True)
        mirror_destination.write_text(destination.read_text(encoding="utf-8"), encoding="utf-8")

    logging.info("Aggregated summary written to %s (and mirrored at %s).", destination, mirror_destination)
    return True


def build_sweeps() -> List[SweepDefinition]:
    scripts_root = REPO_ROOT / "scripts" / "experiments"
    return [
        SweepDefinition(
            name="fixed_chunker",
            method="fixed",
            script=scripts_root / "fixed_sweep.py",
            output_subdir="fixed_sweep_main",
            summary_filename="fixed_sweep_summary.csv",
            base_args=["--max-chars", "1000", "1500", "2000", "2500", "3000"],
            health_port=8000,
        ),
        SweepDefinition(
            name="semantic_chunker",
            method="semantic",
            script=scripts_root / "semantic_sweep.py",
            output_subdir="semantic_sweep_main",
            summary_filename="semantic_sweep_summary.csv",
            base_args=[
                "--chunk-max-chars",
                "2000",
                "--lambda-values",
                "0.05",
                "0.15",
                "0.25",
                "--window-values",
                "20",
                "30",
                "40",
                "--min-sent-values",
                "2",
                "--max-sent-values",
                "40",
            ],
            health_port=8001,
        ),
        SweepDefinition(
            name="dsc_chunker",
            method="dsc",
            script=scripts_root / "dsc_sweep.py",
            output_subdir="dsc_sweep_main",
            summary_filename="dsc_sweep_summary.csv",
            base_args=[
                "--min-parent-values",
                "8",
                "12",
                "--max-parent-values",
                "100",
                "140",
                "--delta-window-values",
                "25",
                "--threshold-values",
                "0.8",
                "1.0",
                "1.2",
                "--heading-options",
                "true",
                "false",
            ],
            health_port=8002,
        ),
    ]


def notify_discord(webhook_url: str, message: str) -> None:
    payload = {"content": message}
    try:
        requests.post(webhook_url, json=payload, timeout=10)
    except Exception as exc:
        print(f"Failed to send Discord notification: {exc}")


def write_done_file(path: Path, statuses: Dict[str, bool], summary_path: Optional[Path]) -> Path:
    lines = [
        f"Completed at {datetime.datetime.utcnow().isoformat()}Z",
        "Sweep statuses:",
    ]
    for name, status in statuses.items():
        state = "SUCCESS" if status else "FAILED"
        lines.append(f"  - {name}: {state}")
    if summary_path:
        lines.append(f"Summary: {summary_path}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ensure_scripts_importable(
        [
            "scripts.experiments.fixed_sweep",
            "scripts.experiments.semantic_sweep",
            "scripts.experiments.dsc_sweep",
        ]
    )
    documents = verify_documents_exist(args.documents)
    verify_docker_running()

    timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_root = REPO_ROOT / "results" / f"full_run_{timestamp}"
    log_dir = REPO_ROOT / "results" / "master_logs" / timestamp
    run_root.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    error_logger = ErrorLogger(run_specific=log_dir / "errors.txt", global_errors=REPO_ROOT / "results" / "master_logs" / "errors.txt")

    sweeps = build_sweeps()
    statuses: Dict[str, bool] = {}
    for sweep in sweeps:
        output_subdir = run_root / sweep.output_subdir
        output_subdir.mkdir(parents=True, exist_ok=True)
        success = run_sweep(
            sweep=sweep,
            documents=documents,
            output_root=run_root,
            log_dir=log_dir,
            host=args.host,
            health_timeout=args.health_timeout,
            health_interval=args.health_interval,
            record_error=error_logger.log,
        )
        statuses[sweep.name] = success

    summary_inputs = [
        (sweep.method, run_root / sweep.output_subdir / sweep.summary_filename) for sweep in sweeps
    ]
    summary_path = run_root / "all_chunking_summary.csv"
    latest_summary = REPO_ROOT / "results" / "all_chunking_summary.csv"
    summary_written = combine_summaries(summary_inputs, summary_path, latest_summary, error_logger.log)
    done_file = write_done_file(run_root / "DONE.txt", statuses, summary_path if summary_written else None)
    fixed_status = "SUCCESS" if statuses.get("fixed_chunker") else "FAILED"
    semantic_status = "SUCCESS" if statuses.get("semantic_chunker") else "FAILED"
    dsc_status = "SUCCESS" if statuses.get("dsc_chunker") else "FAILED"
    notify_discord(
        "https://discord.com/api/webhooks/1439588394404413510/fMrb1wRV8t6KNu9x-L0MwgNXjIuXAlXxKKUcNNshVP1kYnOg_5IWBddGM2cS0tO3X3gE",
        f"🎉 IPKE EXPERIMENT FINISHED at {datetime.datetime.utcnow()} UTC\n"
        f"Status: Fixed={fixed_status}, Semantic={semantic_status}, DSC={dsc_status}",
    )
    logging.info("All sweeps finished. DONE file written to %s.", done_file)


if __name__ == "__main__":
    main()
