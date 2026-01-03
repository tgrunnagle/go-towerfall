"""Shared fixtures and utilities for bot2 tests.

This module provides:
- Server availability checking for integration tests
- The `requires_server` decorator to skip tests when server is unavailable
- Shared fixtures for both unit and integration tests
- Utility functions for generating unique room names
- Custom markers for test categorization
"""

import asyncio
import os
import uuid

import httpx
import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers for test categorization."""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests requiring server"
    )
    config.addinivalue_line(
        "markers", "websocket: marks tests as WebSocket-specific tests"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "stress: marks stress/scalability tests")


# Default server URL, can be overridden via environment variable
DEFAULT_SERVER_URL = os.environ.get("TOWERFALL_SERVER_URL", "http://localhost:4000")


async def _check_server_available(base_url: str, timeout: float = 2.0) -> bool:
    """Check if the go-towerfall server is available.

    Args:
        base_url: The base URL of the server to check.
        timeout: Timeout in seconds for the health check request.

    Returns:
        True if the server is available, False otherwise.
    """
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(f"{base_url}/api/maps")
            if response.status_code != 200:
                return False
            # Verify it's actually the go-towerfall server by checking response format
            data = response.json()
            return isinstance(data, dict) and "maps" in data
    except (httpx.ConnectError, httpx.TimeoutException, Exception):
        return False


def _run_check() -> bool:
    """Helper function to run the async check in a new event loop."""
    return asyncio.run(_check_server_available(DEFAULT_SERVER_URL))


def _is_server_available() -> bool:
    """Synchronous wrapper to check server availability."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're in an async context, run in a separate thread
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(_run_check)
                return future.result(timeout=5.0)
        return loop.run_until_complete(_check_server_available(DEFAULT_SERVER_URL))
    except Exception:
        # If there's no event loop, create one
        try:
            return asyncio.run(_check_server_available(DEFAULT_SERVER_URL))
        except Exception:
            return False


# Skip decorator for integration tests that require a running server
requires_server = pytest.mark.skipif(
    not _is_server_available(),
    reason=f"Integration test requires running go-towerfall server at {DEFAULT_SERVER_URL}",
)


@pytest.fixture(scope="session")
def server_url() -> str:
    """Provide the server URL for integration tests.

    Returns:
        The base URL of the go-towerfall server.
    """
    return DEFAULT_SERVER_URL


def unique_room_name(prefix: str = "IntegrationTest") -> str:
    """Generate a unique room name for integration tests.

    Uses UUID to ensure no conflicts between parallel test runs.

    Args:
        prefix: Prefix for the room name.

    Returns:
        A unique room name string.
    """
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def room_name() -> str:
    """Provide a unique room name for each test.

    Returns:
        A unique room name string.
    """
    return unique_room_name()
