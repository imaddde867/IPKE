"""Typed membership contract for the IPKE evaluation corpus."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CorpusDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    doc_id: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9_]*$")
    source_family: str = Field(min_length=1)
    role: Literal["procedure_candidate", "requirements_stress_test"]
    status: Literal[
        "candidate",
        "excluded_wrong_genre",
        "excluded_pending_reannotation",
    ]
    include_for_evaluation: bool
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_classification(self) -> CorpusDocument:
        if self.include_for_evaluation:
            if self.role != "procedure_candidate" or self.status != "candidate":
                raise ValueError(
                    "included documents must be procedure candidates with candidate status"
                )
        elif self.status == "candidate":
            raise ValueError("excluded documents must use an excluded status")
        return self


class CorpusManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    manifest_status: Literal["provisional", "frozen"]
    documents: tuple[CorpusDocument, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_membership(self) -> CorpusManifest:
        ids = [document.doc_id for document in self.documents]
        if len(ids) != len(set(ids)):
            raise ValueError("manifest document IDs must be unique")
        if not any(document.include_for_evaluation for document in self.documents):
            raise ValueError("manifest must include at least one document")
        return self

    @property
    def included_doc_ids(self) -> tuple[str, ...]:
        return tuple(
            document.doc_id
            for document in self.documents
            if document.include_for_evaluation
        )


def load_corpus_manifest(path: Path) -> CorpusManifest:
    return CorpusManifest.model_validate_json(path.read_text(encoding="utf-8"))


def select_manifest_gold_files(
    manifest: CorpusManifest,
    gold_dir: Path,
) -> tuple[Path, ...]:
    actual = {path.stem: path for path in sorted(gold_dir.glob("*.json"))}
    declared = {document.doc_id for document in manifest.documents}
    missing = declared - actual.keys()
    unclassified = actual.keys() - declared
    if missing or unclassified:
        details: list[str] = []
        if missing:
            details.append(f"missing files: {', '.join(sorted(missing))}")
        if unclassified:
            details.append(f"unclassified files: {', '.join(sorted(unclassified))}")
        raise ValueError(
            "manifest does not match gold directory; " + "; ".join(details)
        )
    return tuple(actual[doc_id] for doc_id in manifest.included_doc_ids)


def select_manifest_production_files(
    manifest: CorpusManifest,
    annotation_dir: Path,
) -> tuple[Path, ...]:
    """Select only included production artifacts.

    Unlike the legacy-candidate resolver, excluded manifest entries do not need a file
    in the production directory. Extra files still fail closed so directory presence
    cannot silently change the evaluation corpus.
    """
    actual = {
        path.stem: path for path in sorted(annotation_dir.glob("*.json"))
    }
    included = set(manifest.included_doc_ids)
    missing = included - actual.keys()
    unclassified = actual.keys() - included
    if missing or unclassified:
        details: list[str] = []
        if missing:
            details.append(
                f"missing included files: {', '.join(sorted(missing))}"
            )
        if unclassified:
            details.append(
                f"unclassified production files: {', '.join(sorted(unclassified))}"
            )
        raise ValueError(
            "manifest does not match production directory; " + "; ".join(details)
        )
    return tuple(actual[doc_id] for doc_id in manifest.included_doc_ids)
