"""
Test-wide fixtures and configuration.

Ensures the suite runs in EXPLAINIUM_ENV=testing so that lightweight mock
strategies are used instead of real LLMs/downloads.
"""
import os

import pytest

from src.core import unified_config


def pytest_configure(config: pytest.Config) -> None:  # noqa: D401
    os.environ.setdefault("EXPLAINIUM_ENV", "testing")
    os.environ.setdefault("GPU_BACKEND", "cpu")
    os.environ.setdefault("ENABLE_GPU", "false")
    unified_config.reload_config()
