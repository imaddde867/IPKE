import tomllib
from pathlib import Path


def load_pyproject():
    return tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))


def base_dependency_names():
    names = set()
    for dependency in load_pyproject()["project"]["dependencies"]:
        name = dependency.split("@", 1)[0].split("=", 1)[0].split("<", 1)[0].split(">", 1)[0].strip()
        names.add(name)
    return names


def test_base_dependencies_exclude_app_stack():
    deps = base_dependency_names()
    assert "streamlit" not in deps
    assert "fastapi" not in deps
    assert "uvicorn" not in deps


def test_base_dependencies_exclude_heavy_llm_backends():
    deps = base_dependency_names()
    assert "llama-cpp-python" not in deps
    assert "transformers" not in deps
    assert "bitsandbytes" not in deps


def test_expected_optional_extras_exist():
    extras = load_pyproject()["project"]["optional-dependencies"]
    for name in ["app", "llm", "dev", "extras", "neo4j"]:
        assert name in extras
