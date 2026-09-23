"""
Pytest configuration and shared fixtures for SGM test suite.
"""

import asyncio
import os
import sys
import pytest

# Ensure root repository directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture
def anyio_backend():
    return "asyncio"
