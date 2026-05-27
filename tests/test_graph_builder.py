import subprocess
import sys


def test_graph_builder_imports_without_neo4j_dependency():
    program = """
import importlib.abc
import sys


class BlockNeo4j(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "neo4j" or fullname.startswith("neo4j."):
            raise ModuleNotFoundError("No module named 'neo4j'")
        return None


sys.meta_path.insert(0, BlockNeo4j())

from src.graph.builder import ProceduralGraphBuilder

graph = ProceduralGraphBuilder().build_from_payload({
    "document_id": "doc-1",
    "steps": [{"id": "S1", "text": "Inspect valve", "order": 1}],
    "constraints": [{"id": "C1", "text": "Wear gloves", "steps": ["S1"]}],
})

assert graph.document_id == "doc-1"
assert len(graph.steps) == 1
assert len(graph.conditions) == 1
"""

    proc = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
