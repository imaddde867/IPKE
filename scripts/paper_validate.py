"""End-to-end validation for the public paper dataset.

This is the single command a reviewer runs to verify the paper corpus.
It is also the body of `make paper-validate`.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

GOLD_DIR = REPO_ROOT / "datasets/paper/gold"
SECOND_DIR = REPO_ROOT / "datasets/paper/second_pass"
STRICT_SCHEMA = REPO_ROOT / "schemas/ipke_paper_tiera.schema.json"
MANIFEST = REPO_ROOT / "datasets/paper/public_sources_manifest.csv"
TEXT_DIR = REPO_ROOT / "datasets/paper/text"
IAA_SMOKE_REPORT = REPO_ROOT / "datasets/paper/reports/issue_53_iaa_report.smoke.json"
PILOT_STATUS = REPO_ROOT / "datasets/paper/annotation_batches/manifest_pilot_status.json"

DOUBLE_ANNOTATED_DOCS = (
    "epa_field_sampling_measurement_procedure_validation",
    "niosh_nmam_5th_edition_ebook",
    "olsk_small_cnc_v1_workbook",
)


def step(label: str) -> None:
    print(f"== {label} ==", flush=True)


def check_manifest_freshness() -> None:
    step("check manifest freshness")
    subprocess.run(
        [
            sys.executable,
            "scripts/check_manifest_freshness.py",
            "--manifest",
            str(MANIFEST),
            "--text-dir",
            str(TEXT_DIR),
        ],
        cwd=REPO_ROOT,
        check=True,
    )


def check_schema_files() -> None:
    step("validate gold against strict schema + second_pass existence")
    try:
        import jsonschema  # type: ignore[import-not-found]
    except ImportError as exc:
        raise SystemExit("jsonschema is required: install via uv sync") from exc

    with STRICT_SCHEMA.open(encoding="utf-8") as handle:
        schema = json.load(handle)
    validator = jsonschema.Draft202012Validator(schema)

    gold_paths = sorted(GOLD_DIR.glob("*.json"))
    if len(gold_paths) < 8:
        raise SystemExit(f"expected 8 gold files, found {len(gold_paths)}")
    for path in gold_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        errors = sorted(validator.iter_errors(payload), key=lambda e: list(e.path))
        if errors:
            for err in errors:
                print(f"  {path.name}: {err.message}", file=sys.stderr)
            raise SystemExit(f"strict schema failed for {path.name}")

    for doc_id in DOUBLE_ANNOTATED_DOCS:
        second_path = SECOND_DIR / f"{doc_id}.json"
        if not second_path.exists():
            raise SystemExit(f"missing second_pass file: {second_path}")


def check_iaa_smoke() -> None:
    step("run IAA smoke check")
    if IAA_SMOKE_REPORT.exists():
        IAA_SMOKE_REPORT.unlink()
    subprocess.run(
        [sys.executable, "scripts/compute_iaa.py", "--smoke-only"],
        cwd=REPO_ROOT,
        check=True,
    )
    payload = json.loads(IAA_SMOKE_REPORT.read_text(encoding="utf-8"))
    if payload.get("mode") != "smoke":
        raise SystemExit(f"unexpected smoke report: {payload!r}")


def check_pilot_status_consistency() -> None:
    step("verify pilot status")
    status = json.loads(PILOT_STATUS.read_text(encoding="utf-8"))
    double_annotated = {
        entry["document_id"]
        for entry in status
        if entry.get("annotation_status") == "double_annotated"
    }
    missing = set(DOUBLE_ANNOTATED_DOCS) - double_annotated
    if missing:
        raise SystemExit(f"pilot_status missing double_annotated docs: {missing}")


def main() -> int:
    check_manifest_freshness()
    check_schema_files()
    check_pilot_status_consistency()
    check_iaa_smoke()
    n_gold = len(list(GOLD_DIR.glob("*.json")))
    n_second = sum(1 for d in DOUBLE_ANNOTATED_DOCS if (SECOND_DIR / f"{d}.json").exists())
    print(
        f"paper-validate OK: {n_gold} gold + {n_second} second_pass validated against strict schema"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
