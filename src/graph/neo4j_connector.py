"""Neo4j connection utilities for persisting procedural graphs.

This module wraps the official Neo4j Python driver with a lightweight
context-managed connector that reads credentials from environment
variables and exposes a tiny surface area tailored for this project.

Environment variables:
    NEO4J_URI: Bolt URI (default: bolt://localhost:7687)
    NEO4J_USER: Username (default: neo4j)
    NEO4J_PASSWORD: Password (default: password)
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from neo4j import GraphDatabase, Result

from src.graph.models import Graph


class Neo4jConnector:
    """Thin connection manager around the Neo4j driver."""

    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
    ) -> None:
        self.uri = uri or os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.environ.get("NEO4J_USER", "neo4j")
        self.password = password or os.environ.get("NEO4J_PASSWORD", "password")
        self._driver = None

    # ------------------------------------------------------------------ utils
    def connect(self):
        if self._driver is None:
            self._driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        return self._driver

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    def __enter__(self) -> "Neo4jConnector":  # pragma: no cover - simple helper
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - simple helper
        self.close()

    # ----------------------------------------------------------------- actions
    def clear_graph(self) -> None:
        """Detach delete every node/edge in the database."""

        driver = self.connect()
        with driver.session() as session:
            session.execute_write(lambda tx: tx.run("MATCH (n) DETACH DELETE n"))

    def create_graph(self, graph: Graph, clear_existing: bool = False) -> None:
        """Persist a :class:`Graph` into Neo4j using MERGE/CREATE Cypher."""

        driver = self.connect()
        with driver.session() as session:
            if clear_existing:
                session.execute_write(lambda tx: tx.run("MATCH (n) DETACH DELETE n"))

            # Nodes
            for step in graph.steps:
                payload = self._model_to_dict(step)
                session.execute_write(
                    self._merge_node,
                    label="Step",
                    payload=payload,
                )

            for condition in graph.conditions:
                payload = self._model_to_dict(condition)
                session.execute_write(
                    self._merge_node,
                    label="Condition",
                    payload=payload,
                )

            for equipment in graph.equipment:
                payload = self._model_to_dict(equipment)
                session.execute_write(
                    self._merge_node,
                    label="Equipment",
                    payload=payload,
                )

            for parameter in graph.parameters:
                payload = self._model_to_dict(parameter)
                session.execute_write(
                    self._merge_node,
                    label="Parameter",
                    payload=payload,
                )

            # Relationships
            for edge in graph.edges:
                payload = self._model_to_dict(edge)
                session.execute_write(self._create_relationship, payload)

    def query(self, cypher: str, **params: Any) -> List[Dict[str, Any]]:
        """Run an arbitrary Cypher query and return dictionaries of values."""

        driver = self.connect()
        with driver.session() as session:
            result: Result = session.run(cypher, **params)
            records: List[Dict[str, Any]] = []
            for record in result:
                records.append(record.data())
            return records

    # --------------------------------------------------------------- tx helpers
    @staticmethod
    def _merge_node(tx, label: str, payload: Dict[str, Any]) -> None:
        tx.run(
            f"MERGE (n:{label} {{id: $id}}) "
            "SET n += $props",
            id=payload.get("id"),
            props={k: v for k, v in payload.items() if k != "id"},
        )

    @staticmethod
    def _create_relationship(tx, payload: Dict[str, Any]) -> None:
        edge_type = payload.get("type")
        if not edge_type:
            return
        tx.run(
            "MATCH (source {id: $from_id}), (target {id: $to_id}) "
            "MERGE (source)-[r:%s]->(target) "
            "SET r += $props" % edge_type,
            from_id=payload.get("from_id"),
            to_id=payload.get("to_id"),
            props={k: v for k, v in payload.items() if k not in {"id", "from_id", "to_id", "type"}},
        )

    @staticmethod
    def _model_to_dict(model: Any) -> Dict[str, Any]:
        if hasattr(model, "model_dump"):
            data = model.model_dump(exclude_none=True)
        elif isinstance(model, dict):
            data = {k: v for k, v in model.items() if v is not None}
        else:
            data = {k: v for k, v in vars(model).items() if v is not None}
        return data


__all__ = ["Neo4jConnector"]
