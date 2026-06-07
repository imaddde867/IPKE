"""
Test-wide fixtures and configuration.

Defaults EXPLAINIUM_ENV, GPU_BACKEND, and ENABLE_GPU to testing-safe values
if the caller has not set them. Caller-provided environment takes precedence.
"""
import os

import pytest

from src.core import unified_config


def pytest_configure(config: pytest.Config) -> None:  # noqa: D401
    os.environ.setdefault("EXPLAINIUM_ENV", "testing")
    os.environ.setdefault("GPU_BACKEND", "cpu")
    os.environ.setdefault("ENABLE_GPU", "false")
    unified_config.reload_config()
