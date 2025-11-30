"""
Utility builder that turns the flat extractor output (steps + constraints)
into the strongly-typed procedural graph defined in ``src.graph.models``.

The builder is intentionally lightweight – it sanitizes the raw payload,
derives default IDs/text when necessary, stitches together simple procedural
``NEXT`` edges from the ordered steps, and links constraint nodes back to the
steps they guard.  It also exposes a ``get_topology_stats`` helper that is
handy for descriptive analysis inside notebooks or ad-hoc scripts.

This file now also doubles as a CLI utility that can optionally persist a
built graph into a locally running Neo4j instance (see ``--persist-neo4j``).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Union

from .models import Condition, Edge, Graph, Step
from src.validation.constraint_validator import ValidationReport, validate_constraints
from .neo4j_connector import Neo4jConnector


ConditionType = Optional[str]
Payload = Dict[str, Any]

ALLOWED_CONDITION_TYPES = {
    "precondition",
    "postcondition",
    "safety",
    "environment",
    "quality",
    "exception",
}


class ProceduralGraphBuilder:
    """Builds :class:`src.graph.models.Graph` instances from flat JSON payloads."""

    def __init__(self) -> None:
        self.graph: Optional[Graph] = None
        self.validation_report: Optional[ValidationReport] = None

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def build_from_json(self, path: Union[str, Path]) -> Graph:
        """Load a prediction JSON file and build a Graph."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return self.build_from_payload(data)

    def build_from_payload(self, payload: Payload) -> Graph:
        """Build a Graph directly from an in-memory payload."""
        raw_constraints = payload.get("constraints") or []
        steps = self._build_steps(payload.get("steps") or [])
        self.validation_report = validate_constraints(raw_constraints)
        conditions = self._build_conditions(raw_constraints)
        edges = self._build_edges(steps, conditions, raw_constraints)

        raw_metadata = payload.get("metadata")
        metadata = self._coerce_metadata(raw_metadata if isinstance(raw_metadata, dict) else {})

        graph = Graph(
            document_id=str(payload.get("document_id") or "unknown_document"),
            document_type=payload.get("document_type"),
            title=payload.get("title"),
            steps=steps,
            conditions=conditions,
            equipment=[],
            parameters=[],
            edges=edges,
            metadata=metadata,
        )

        self.graph = graph
        return graph

    def get_validation_report(self) -> ValidationReport:
        if self.validation_report is None:
            raise RuntimeError("ProceduralGraphBuilder.build_from_* must be called before accessing validation report.")
        return self.validation_report

    def persist_to_neo4j(
        self,
        graph: Graph,
        clear_existing: bool = False,
        connector: Optional[Neo4jConnector] = None,
    ) -> None:
        """Persist the given graph into the configured Neo4j instance."""

        if connector is None:
            with Neo4jConnector() as managed:
                managed.create_graph(graph, clear_existing=clear_existing)
        else:
            connector.create_graph(graph, clear_existing=clear_existing)

    def get_graph(self) -> Graph:
        """Return the last built Graph or raise if build() has not been called."""
        if self.graph is None:
            raise RuntimeError("ProceduralGraphBuilder.build_from_* must be called before accessing the graph.")
        return self.graph

    def get_topology_stats(self) -> Dict[str, Any]:
        """Return high-level stats for the current graph."""
        graph = self.get_graph()
        step_count = len(graph.steps)
        constraint_count = len(graph.conditions)
        equipment_count = len(graph.equipment)
        parameter_count = len(graph.parameters)
        node_count = step_count + constraint_count + equipment_count + parameter_count
        edge_count = len(graph.edges)

        max_edges = node_count * (node_count - 1)
        density = (edge_count / max_edges) if max_edges else 0.0
        constraint_density = (constraint_count / step_count) if step_count else 0.0

        return {
            "document_id": graph.document_id,
            "node_count": node_count,
            "step_count": step_count,
            "constraint_count": constraint_count,
            "equipment_count": equipment_count,
            "parameter_count": parameter_count,
            "edge_count": edge_count,
            "density": density,
            "constraint_density": constraint_density,
        }

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #
    @staticmethod
    def _build_steps(raw_steps: Sequence[Payload]) -> List[Step]:
        steps: List[Step] = []
        for idx, raw in enumerate(raw_steps):
            sid = ProceduralGraphBuilder._coerce_id(raw.get("id"), prefix="S", index=idx)
            text = ProceduralGraphBuilder._extract_first(
                raw,
                [
                    "text",
                    "description",
                    "statement",
                    "name",
                    "summary",
                ],
                fallback=sid,
            )
            order = ProceduralGraphBuilder._safe_int(raw.get("order"))
            step = Step(
                id=sid,
                text=text,
                order=order,
                section=raw.get("section"),
                context=raw.get("context"),
                references=raw.get("references"),
            )
            steps.append(step)
        return steps

    @staticmethod
    def _build_conditions(raw_constraints: Sequence[Payload]) -> List[Condition]:
        conditions: List[Condition] = []
        for idx, raw in enumerate(raw_constraints):
            cid = ProceduralGraphBuilder._coerce_id(raw.get("id"), prefix="C", index=idx)
            expression = ProceduralGraphBuilder._extract_first(
                raw,
                ["expression", "text", "description", "statement"],
                fallback=cid,
            )
            cond_type = ProceduralGraphBuilder._normalize_condition_type(raw.get("type"))
            condition = Condition(
                id=cid,
                expression=expression,
                type=cond_type,
                context=raw.get("context"),
            )
            conditions.append(condition)
        return conditions

    @staticmethod
    def _build_edges(
        steps: Sequence[Step],
        conditions: Sequence[Condition],
        raw_constraints: Sequence[Payload],
    ) -> List[Edge]:
        edges: List[Edge] = []
        ordered_ids = ProceduralGraphBuilder._ordered_step_ids(steps)
        for current, nxt in zip(ordered_ids, ordered_ids[1:]):
            edges.append(Edge(from_id=current, to_id=nxt, type="NEXT"))

        step_ids = {step.id for step in steps}
        condition_lookup = {cond.id: cond for cond in conditions}
        for idx, raw in enumerate(raw_constraints):
            cid = ProceduralGraphBuilder._coerce_id(raw.get("id"), prefix="C", index=idx)
            if cid not in condition_lookup:
                continue
            targets = ProceduralGraphBuilder._collect_constraint_refs(raw)
            for target in targets:
                if target not in step_ids:
                    continue
                edges.append(Edge(from_id=cid, to_id=target, type="CONDITION_ON"))
        return edges

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _collect_constraint_refs(raw: Payload) -> Set[str]:
        keys = [
            "step",
            "steps",
            "step_id",
            "attached_to",
            "attached_step",
            "attached_steps",
            "applies_to",
            "scope",
            "targets",
        ]
        refs: Set[str] = set()
        for key in keys:
            value = raw.get(key)
            if not value:
                continue
            refs.update(ProceduralGraphBuilder._flatten_ref_value(value))
        return refs

    @staticmethod
    def _flatten_ref_value(value: Any) -> Set[str]:
        if isinstance(value, str):
            return {value}
        if isinstance(value, dict):
            out: Set[str] = set()
            for key in ("id", "step_id"):
                candidate = value.get(key)
                if candidate:
                    out.add(str(candidate))
            return out
        if isinstance(value, Iterable):
            refs: Set[str] = set()
            for item in value:
                refs.update(ProceduralGraphBuilder._flatten_ref_value(item))
            return refs
        candidate = str(value).strip()
        return {candidate} if candidate else set()

    @staticmethod
    def _ordered_step_ids(steps: Sequence[Step]) -> List[str]:
        enumerated = list(enumerate(steps))
        enumerated.sort(
            key=lambda item: (
                item[1].order if item[1].order is not None else float("inf"),
                item[0],
            )
        )
        return [step.id for _, step in enumerated]

    @staticmethod
    def _normalize_condition_type(value: Any) -> ConditionType:
        if value is None:
            return None
        text = str(value).strip().lower()
        if text in ALLOWED_CONDITION_TYPES:
            return text
        return None

    @staticmethod
    def _coerce_id(value: Any, prefix: str, index: int) -> str:
        if isinstance(value, str) and value.strip():
            return value.strip()
        if value is not None:
            candidate = str(value).strip()
            if candidate:
                return candidate
        return f"{prefix}{index + 1}"

    @staticmethod
    def _extract_first(raw: Payload, keys: Sequence[str], fallback: str) -> str:
        for key in keys:
            value = raw.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return fallback

    @staticmethod
    def _safe_int(value: Any) -> Optional[int]:
        try:
            return int(value)
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _coerce_metadata(raw_meta: Payload) -> Dict[str, Optional[str]]:
        meta: Dict[str, Optional[str]] = {}
        for key, value in raw_meta.items():
            if value is None or isinstance(value, str):
                meta[str(key)] = value
            else:
                meta[str(key)] = str(value)
        return meta


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and optionally persist procedural graphs")
    parser.add_argument("input", help="Path to a prediction JSON file")
    parser.add_argument(
        "--persist-neo4j",
        action="store_true",
        help="Persist the built graph to the configured Neo4j instance",
    )
    parser.add_argument(
        "--neo4j-clear",
        action="store_true",
        help="Clear the Neo4j database before persisting (implies --persist-neo4j)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    builder = ProceduralGraphBuilder()
    graph = builder.build_from_json(args.input)
    stats = builder.get_topology_stats()
    print(json.dumps(stats, indent=2))

    if args.persist_neo4j or args.neo4j_clear:
        connector = Neo4jConnector()
        builder.persist_to_neo4j(graph, clear_existing=args.neo4j_clear, connector=connector)
        print(f"Persisted graph to Neo4j at {connector.uri}")


if __name__ == "__main__":  # pragma: no cover - CLI utility
    main()
