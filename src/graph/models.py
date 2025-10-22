"""
Procedural Graph Models : Pydantic models to represent a structured procedural knowledge graph used by the
thesis pipeline. This mirrors the JSON schema in `src/graph/schema.json`.

Core concepts
- Nodes: Step, Condition, Equipment, Parameter
- Edges: typed relations between node IDs (e.g., NEXT, CONDITION_ON, USES)
- Graph: container holding nodes, edges, and metadata

Notes
- Keep IDs unique within a graph.
- Prefer stable step ordering via `order` on Step plus explicit NEXT edges.
- For matching/evaluation, use simple normalization helpers provided here.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Literal
from pydantic import BaseModel, Field


def normalize_text(s: Optional[str]) -> str:
    """Lightweight simple initial normalization for string comparison in evaluation :
    - Lowercase
    - Strip whitespace
    - Collapse multiple spaces
    - Remove common punctuation at ends
    """
    if not s:
        return ""
    import re

    s2 = s.lower().strip()
    s2 = re.sub(r"\s+", " ", s2)
    s2 = re.sub(r"^[\s\-:\.;,]+|[\s\-:\.;,]+$", "", s2)
    return s2


class Step(BaseModel):
    id: str = Field(..., description="Unique step identifier")
    order: Optional[int] = Field(None, ge=1, description="1-based order if available")
    text: str = Field(..., description="Verbatim or paraphrased step text")
    section: Optional[str] = Field(None, description="Section or heading context")
    context: Optional[str] = Field(None, description="Local context excerpt")
    references: Optional[List[str]] = Field(default=None, description="Optional citations or anchors")


class Condition(BaseModel):
    id: str
    type: Optional[Literal[
        "precondition",
        "postcondition",
        "safety",
        "environment",
        "quality",
        "exception",
    ]] = "precondition"
    expression: str = Field(..., description="Condition expression or clause")
    context: Optional[str] = None


class Equipment(BaseModel):
    id: str
    name: str
    model: Optional[str] = None
    category: Optional[str] = None
    vendor: Optional[str] = None
    notes: Optional[str] = None


class Parameter(BaseModel):
    id: str
    name: str
    value: Optional[str] = None
    unit: Optional[str] = None
    range: Optional[str] = None
    notes: Optional[str] = None


class Edge(BaseModel):
    id: Optional[str] = None
    from_id: str = Field(..., description="Source node ID")
    to_id: str = Field(..., description="Target node ID")
    type: Literal[
        "NEXT",
        "CONDITION_ON",
        "USES",
        "HAS_PARAMETER",
        "REQUIRES",
        "PRODUCES",
        "REFERENCES",
        "ALTERNATIVE_TO",
    ]
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    attributes: Optional[Dict[str, str]] = None


class Graph(BaseModel):
    document_id: str
    document_type: Optional[str] = None
    title: Optional[str] = None

    steps: List[Step] = []
    conditions: List[Condition] = []
    equipment: List[Equipment] = []
    parameters: List[Parameter] = []
    edges: List[Edge] = []

    metadata: Dict[str, Optional[str]] = {}

    def id_index(self) -> Dict[str, BaseModel]:
        idx: Dict[str, BaseModel] = {}
        for c in (self.steps + self.conditions + self.equipment + self.parameters):
            idx[c.id] = c
        return idx

    def summarize(self) -> Dict[str, int]:
        return {
            "steps": len(self.steps),
            "conditions": len(self.conditions),
            "equipment": len(self.equipment),
            "parameters": len(self.parameters),
            "edges": len(self.edges),
        }

