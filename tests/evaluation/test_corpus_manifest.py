from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.evaluation.corpus_manifest import (
    load_corpus_manifest,
    select_manifest_gold_files,
)


def _entry(
    doc_id: str,
    *,
    include: bool,
    role: str = "procedure_candidate",
    status: str = "candidate",
) -> dict[str, object]:
    return {
        "doc_id": doc_id,
        "source_family": "test",
        "role": role,
        "status": status,
        "include_for_evaluation": include,
        "reason": "Test fixture.",
    }


def _write_manifest(path: Path, documents: list[dict[str, object]]) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "manifest_status": "provisional",
                "documents": documents,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_manifest_selects_only_included_gold(tmp_path: Path) -> None:
    gold_dir = tmp_path / "gold"
    gold_dir.mkdir()
    (gold_dir / "include.json").write_text("{}", encoding="utf-8")
    (gold_dir / "exclude.json").write_text("{}", encoding="utf-8")
    path = _write_manifest(
        tmp_path / "manifest.json",
        [
            _entry("include", include=True),
            _entry(
                "exclude",
                include=False,
                role="requirements_stress_test",
                status="excluded_wrong_genre",
            ),
        ],
    )

    manifest = load_corpus_manifest(path)
    selected = select_manifest_gold_files(manifest, gold_dir)

    assert [item.stem for item in selected] == ["include"]


def test_manifest_rejects_duplicate_document_ids(tmp_path: Path) -> None:
    path = _write_manifest(
        tmp_path / "manifest.json",
        [
            _entry("duplicate", include=True),
            _entry("duplicate", include=True),
        ],
    )

    with pytest.raises(
        ValidationError,
        match="manifest document IDs must be unique",
    ):
        load_corpus_manifest(path)


@pytest.mark.parametrize(
    ("include", "role", "status", "message"),
    [
        (
            True,
            "requirements_stress_test",
            "candidate",
            "included documents must be procedure candidates with candidate status",
        ),
        (
            True,
            "procedure_candidate",
            "excluded_pending_reannotation",
            "included documents must be procedure candidates with candidate status",
        ),
        (
            False,
            "procedure_candidate",
            "candidate",
            "excluded documents must use an excluded status",
        ),
    ],
)
def test_manifest_rejects_inconsistent_classification(
    tmp_path: Path,
    include: bool,
    role: str,
    status: str,
    message: str,
) -> None:
    path = _write_manifest(
        tmp_path / "manifest.json",
        [
            _entry(
                "inconsistent",
                include=include,
                role=role,
                status=status,
            )
        ],
    )

    with pytest.raises(ValidationError, match=message):
        load_corpus_manifest(path)


def test_manifest_rejects_empty_evaluation_inclusion(tmp_path: Path) -> None:
    path = _write_manifest(
        tmp_path / "manifest.json",
        [
            _entry(
                "excluded",
                include=False,
                role="procedure_candidate",
                status="excluded_pending_reannotation",
            )
        ],
    )

    with pytest.raises(
        ValidationError,
        match="manifest must include at least one document",
    ):
        load_corpus_manifest(path)


def test_selection_rejects_missing_declared_gold_file(tmp_path: Path) -> None:
    gold_dir = tmp_path / "gold"
    gold_dir.mkdir()
    (gold_dir / "present.json").write_text("{}", encoding="utf-8")
    path = _write_manifest(
        tmp_path / "manifest.json",
        [
            _entry("present", include=True),
            _entry(
                "missing",
                include=False,
                status="excluded_pending_reannotation",
            ),
        ],
    )
    manifest = load_corpus_manifest(path)

    with pytest.raises(ValueError, match="missing files: missing"):
        select_manifest_gold_files(manifest, gold_dir)


def test_selection_rejects_unclassified_gold_file(tmp_path: Path) -> None:
    gold_dir = tmp_path / "gold"
    gold_dir.mkdir()
    (gold_dir / "included.json").write_text("{}", encoding="utf-8")
    (gold_dir / "extra.json").write_text("{}", encoding="utf-8")
    path = _write_manifest(
        tmp_path / "manifest.json",
        [_entry("included", include=True)],
    )
    manifest = load_corpus_manifest(path)

    with pytest.raises(ValueError, match="unclassified files: extra"):
        select_manifest_gold_files(manifest, gold_dir)


def test_selection_reports_all_directory_mismatches(tmp_path: Path) -> None:
    gold_dir = tmp_path / "gold"
    gold_dir.mkdir()
    (gold_dir / "included.json").write_text("{}", encoding="utf-8")
    (gold_dir / "extra.json").write_text("{}", encoding="utf-8")
    path = _write_manifest(
        tmp_path / "manifest.json",
        [
            _entry("included", include=True),
            _entry(
                "missing",
                include=False,
                status="excluded_pending_reannotation",
            ),
        ],
    )
    manifest = load_corpus_manifest(path)

    with pytest.raises(
        ValueError,
        match="missing files: missing; unclassified files: extra",
    ):
        select_manifest_gold_files(manifest, gold_dir)


def test_repository_manifest_classifies_legacy_candidates() -> None:
    gold_dir = Path("datasets/paper/gold")
    manifest = load_corpus_manifest(Path("datasets/paper/corpus_manifest.json"))
    selected = select_manifest_gold_files(manifest, gold_dir)
    by_id = {document.doc_id: document for document in manifest.documents}
    included_ids = {
        "epa_field_operations_manual_filter_sampling_sop",
        "epa_field_sampling_measurement_procedure_validation",
        "epa_guidance_preparing_sops_qag6",
        "usgs_groundwater_technical_procedures_tm1_a1",
        "usgs_nfm_collection_water_samples_a4",
    }

    assert manifest.manifest_status == "provisional"
    assert len(manifest.documents) == 8
    assert set(by_id) == {path.stem for path in gold_dir.glob("*.json")}
    assert {path.stem for path in selected} == included_ids
    assert all(
        by_id[doc_id].source_family == "epa"
        for doc_id in included_ids
        if doc_id.startswith("epa_")
    )
    assert all(
        by_id[doc_id].source_family == "usgs"
        for doc_id in included_ids
        if doc_id.startswith("usgs_")
    )
    assert by_id["nasa_npr_8715_3d_general_safety"].role == ("requirements_stress_test")
    assert by_id["nasa_npr_8715_3d_general_safety"].status == ("excluded_wrong_genre")
    assert by_id["olsk_small_cnc_v1_workbook"].status == (
        "excluded_pending_reannotation"
    )
    assert by_id["niosh_nmam_5th_edition_ebook"].status == (
        "excluded_pending_reannotation"
    )
