"""Integration tests for server connection.

Tests cover:
- Verifying server is reachable
- Fetching available maps via /api/maps
"""

import pytest

from tests.conftest import requires_server


@pytest.mark.integration
class TestServerConnection:
    """Tests for basic server connectivity."""

    @requires_server
    @pytest.mark.asyncio
    async def test_server_is_reachable(self, server_url: str) -> None:
        """Verify server is reachable at the configured URL."""
        import httpx

        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{server_url}/api/maps")

            assert response.status_code == 200

    @requires_server
    @pytest.mark.asyncio
    async def test_fetch_available_maps(self, server_url: str) -> None:
        """Fetch available maps via /api/maps endpoint."""
        import httpx

        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{server_url}/api/maps")

            assert response.status_code == 200
            data = response.json()

            # Should return a dict with a "maps" key containing a list
            assert isinstance(data, dict)
            assert "maps" in data
            maps = data["maps"]
            assert isinstance(maps, list)
            assert len(maps) > 0
            # Each map should be a dict with required fields
            for map_info in maps:
                assert isinstance(map_info, dict)
                assert "name" in map_info
                assert "type" in map_info
