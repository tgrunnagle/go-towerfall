"""
Pytest configuration and fixtures for server integration tests.

This configuration sets up the server tests to run against a deployed game server,
similar to the main integration tests.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

import pytest
import pytest_asyncio

# Add the bot directory to Python path for imports
bot_dir = Path(__file__).parent.parent.parent.parent
if str(bot_dir) not in sys.path:
    sys.path.insert(0, str(bot_dir))


# Configure pytest-asyncio
pytest_plugins = ('pytest_asyncio',)


@pytest_asyncio.fixture(scope="session")
async def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(autouse=True)
def setup_logging():
    """Setup logging for tests."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create a temporary directory for test data."""
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()
    return data_dir


# Server configuration for integration tests
@pytest.fixture(scope="session")
def server_config():
    """Configuration for test servers."""
    return {
        "go_server_port": 4000,
        "bot_server_port": 4002,
        "go_server_url": "http://localhost:4000",
        "bot_server_url": "http://localhost:4002",
        "ws_url": "ws://localhost:4000/ws"
    }


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test requiring servers"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "server_integration: mark test as requiring deployed game server"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers."""
    for item in items:
        # Mark server integration tests
        if any(name in item.nodeid for name in ["test_bot_lifecycle", "test_bot_server", "test_bot_lifecycle_api"]):
            item.add_marker(pytest.mark.server_integration)
            item.add_marker(pytest.mark.slow)


# Skip integration tests if servers are not available
def pytest_runtest_setup(item):
    """Setup for individual test runs."""
    if "server_integration" in [mark.name for mark in item.iter_markers()]:
        # Check if we should skip integration tests
        if os.environ.get("SKIP_INTEGRATION_TESTS", "").lower() in ("true", "1", "yes"):
            pytest.skip("Server integration tests disabled by environment variable")