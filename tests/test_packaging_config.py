import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def load_pyproject():
    return tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))


def dependency_names(dependencies):
    names = set()
    for dependency in dependencies:
        names.add(canonicalize_name(Requirement(dependency).name))
    return names


def test_base_dependencies_exclude_app_stack():
    deps = dependency_names(load_pyproject()["project"]["dependencies"])
    assert "streamlit" not in deps
    assert "fastapi" not in deps
    assert "uvicorn" not in deps


def test_base_dependencies_exclude_heavy_llm_backends():
    deps = dependency_names(load_pyproject()["project"]["dependencies"])
    assert "llama-cpp-python" not in deps
    assert "transformers" not in deps
    assert "bitsandbytes" not in deps


def test_optional_extras_are_runtime_only():
    extras = load_pyproject()["project"]["optional-dependencies"]
    assert {"app", "llm", "extras", "neo4j"}.issubset(extras)
    assert "dev" not in extras


def test_dev_dependency_group_includes_pytest_tools():
    dependency_groups = load_pyproject()["dependency-groups"]
    dev_deps = dependency_names(dependency_groups["dev"])
    assert "pytest" in dev_deps
    assert "pytest-asyncio" in dev_deps
