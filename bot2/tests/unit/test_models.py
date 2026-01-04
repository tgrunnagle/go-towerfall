"""Unit tests for bot Pydantic models.

Tests cover:
- Model validation with real game state JSON samples
- Serialization produces valid JSON for server
- GameState.from_update() correctly parses GameUpdate into typed objects
- All imports work correctly
"""

import pytest

from bot.models import (
    GAME_CONSTANTS,
    ArrowState,
    BlockState,
    BotActionRequest,
    CreateGameRequest,
    CreateGameResponse,
    GameConstants,
    GameState,
    GameUpdate,
    JoinGameRequest,
    KeyStatusRequest,
    MessageTypes,
    ObjectTypes,
    PlayerClickRequest,
    PlayerState,
    Point,
    RejoinGameRequest,
    RejoinGameResponse,
    StateKeys,
    WebSocketMessage,
)


class TestPointModel:
    """Tests for Point model."""

    def test_point_creation(self) -> None:
        """Test basic point creation."""
        # Use alias names (X, Y) for type checker compatibility
        point = Point(X=100.0, Y=200.0)
        assert point.x == 100.0
        assert point.y == 200.0

    def test_point_from_json(self) -> None:
        """Test point parsing from JSON."""
        data = {"x": 150.5, "y": 250.5}
        point = Point.model_validate(data)
        assert point.x == 150.5
        assert point.y == 250.5

    def test_point_serialization(self) -> None:
        """Test point serialization to dict."""
        # Use alias names (X, Y) for type checker compatibility
        point = Point(X=100.0, Y=200.0)
        data = point.model_dump()
        assert data == {"x": 100.0, "y": 200.0}


class TestPlayerStateModel:
    """Tests for PlayerState model."""

    @pytest.fixture
    def player_json(self) -> dict:
        """Sample player state JSON from server."""
        return {
            "id": "player-123",
            "objectType": "player",
            "name": "TestBot",
            "x": 400.0,
            "y": 300.0,
            "dx": 5.0,
            "dy": -10.0,
            "dir": 0.0,
            "rad": 20.0,
            "h": 100,
            "dead": False,
            "sht": False,
            "jc": 0,
            "ac": 4,
        }

    def test_player_state_parsing(self, player_json: dict) -> None:
        """Test parsing player state from server JSON."""
        player = PlayerState.model_validate(player_json)

        assert player.id == "player-123"
        assert player.object_type == "player"
        assert player.name == "TestBot"
        assert player.x == 400.0
        assert player.y == 300.0
        assert player.dx == 5.0
        assert player.dy == -10.0
        assert player.direction == 0.0
        assert player.radius == 20.0
        assert player.health == 100
        assert player.dead is False
        assert player.shooting is False
        assert player.jump_count == 0
        assert player.arrow_count == 4

    def test_player_state_serialization(self, player_json: dict) -> None:
        """Test player state serialization with aliases."""
        player = PlayerState.model_validate(player_json)
        data = player.model_dump(by_alias=True)

        # Verify aliases are used
        assert data["objectType"] == "player"
        assert data["dir"] == 0.0
        assert data["rad"] == 20.0
        assert data["h"] == 100
        assert data["sht"] is False
        assert data["jc"] == 0
        assert data["ac"] == 4

    def test_player_state_with_shooting(self) -> None:
        """Test player state while shooting."""
        data = {
            "id": "player-456",
            "objectType": "player",
            "name": "Archer",
            "x": 200.0,
            "y": 400.0,
            "dx": 0.0,
            "dy": 0.0,
            "dir": 1.57,
            "rad": 20.0,
            "h": 80,
            "dead": False,
            "sht": True,
            "shts": 1234567.89,
            "jc": 1,
            "ac": 3,
        }
        player = PlayerState.model_validate(data)
        assert player.shooting is True
        assert player.shooting_start_time == 1234567.89


class TestArrowStateModel:
    """Tests for ArrowState model."""

    def test_arrow_state_parsing(self) -> None:
        """Test parsing arrow state from server JSON."""
        data = {
            "id": "arrow-789",
            "objectType": "arrow",
            "x": 300.0,
            "y": 250.0,
            "dx": 15.0,
            "dy": -5.0,
            "dir": -0.32,
            "ag": False,
        }
        arrow = ArrowState.model_validate(data)

        assert arrow.id == "arrow-789"
        assert arrow.object_type == "arrow"
        assert arrow.x == 300.0
        assert arrow.y == 250.0
        assert arrow.dx == 15.0
        assert arrow.dy == -5.0
        assert arrow.direction == -0.32
        assert arrow.grounded is False
        assert arrow.destroyed is False

    def test_arrow_state_grounded(self) -> None:
        """Test arrow state when grounded."""
        data = {
            "id": "arrow-101",
            "objectType": "arrow",
            "x": 500.0,
            "y": 600.0,
            "dx": 0.0,
            "dy": 0.0,
            "ag": True,
        }
        arrow = ArrowState.model_validate(data)
        assert arrow.grounded is True

    def test_arrow_state_serialization(self) -> None:
        """Test arrow state serialization with aliases."""
        arrow = ArrowState.model_validate(
            {
                "id": "arrow-1",
                "objectType": "arrow",
                "x": 100.0,
                "y": 200.0,
                "dx": 10.0,
                "dy": -5.0,
                "ag": False,
            }
        )
        data = arrow.model_dump(by_alias=True)
        assert data["objectType"] == "arrow"
        assert data["ag"] is False


class TestBlockStateModel:
    """Tests for BlockState model."""

    def test_block_state_parsing(self) -> None:
        """Test parsing block state with polygon points."""
        data = {
            "id": "block-001",
            "objectType": "block",
            "pts": [
                {"x": 0.0, "y": 0.0},
                {"x": 100.0, "y": 0.0},
                {"x": 100.0, "y": 20.0},
                {"x": 0.0, "y": 20.0},
            ],
        }
        block = BlockState.model_validate(data)

        assert block.id == "block-001"
        assert block.object_type == "block"
        assert len(block.points) == 4
        assert block.points[0].x == 0.0
        assert block.points[0].y == 0.0
        assert block.points[2].x == 100.0
        assert block.points[2].y == 20.0


class TestGameUpdateModel:
    """Tests for GameUpdate model."""

    @pytest.fixture
    def full_update_json(self) -> dict:
        """Sample full game update from server."""
        return {
            "fullUpdate": True,
            "objectStates": {
                "player-1": {
                    "id": "player-1",
                    "objectType": "player",
                    "name": "Bot1",
                    "x": 100.0,
                    "y": 200.0,
                    "dx": 0.0,
                    "dy": 0.0,
                    "dir": 0.0,
                    "rad": 20.0,
                    "h": 100,
                    "dead": False,
                    "sht": False,
                    "jc": 0,
                    "ac": 4,
                },
                "block-1": {
                    "id": "block-1",
                    "objectType": "block",
                    "pts": [
                        {"x": 300.0, "y": 700.0},
                        {"x": 500.0, "y": 700.0},
                        {"x": 500.0, "y": 720.0},
                        {"x": 300.0, "y": 720.0},
                    ],
                },
            },
            "events": [{"type": "GameStart", "data": {}}],
        }

    def test_game_update_parsing(self, full_update_json: dict) -> None:
        """Test parsing full game update."""
        update = GameUpdate.model_validate(full_update_json)

        assert update.full_update is True
        assert "player-1" in update.object_states
        assert "block-1" in update.object_states
        assert len(update.events) == 1
        assert update.events[0].type == "GameStart"

    def test_game_update_with_destroyed_object(self) -> None:
        """Test game update with destroyed object (null value)."""
        data = {
            "fullUpdate": False,
            "objectStates": {
                "arrow-1": None,  # Destroyed
                "player-1": {
                    "id": "player-1",
                    "objectType": "player",
                    "name": "Test",
                    "x": 100.0,
                    "y": 200.0,
                    "dx": 0.0,
                    "dy": 0.0,
                    "dir": 0.0,
                    "rad": 20.0,
                    "h": 100,
                    "dead": False,
                    "sht": False,
                    "jc": 0,
                    "ac": 4,
                },
            },
            "events": [],
        }
        update = GameUpdate.model_validate(data)
        assert update.object_states["arrow-1"] is None
        assert update.object_states["player-1"] is not None


class TestGameStateModel:
    """Tests for GameState model and from_update method."""

    @pytest.fixture
    def game_update(self) -> GameUpdate:
        """Create a sample game update."""
        return GameUpdate.model_validate(
            {
                "fullUpdate": True,
                "objectStates": {
                    "player-1": {
                        "id": "player-1",
                        "objectType": "player",
                        "name": "Bot1",
                        "x": 100.0,
                        "y": 200.0,
                        "dx": 0.0,
                        "dy": 0.0,
                        "dir": 0.0,
                        "rad": 20.0,
                        "h": 100,
                        "dead": False,
                        "sht": False,
                        "jc": 0,
                        "ac": 4,
                    },
                    "player-2": {
                        "id": "player-2",
                        "objectType": "player",
                        "name": "Bot2",
                        "x": 700.0,
                        "y": 200.0,
                        "dx": 0.0,
                        "dy": 0.0,
                        "dir": 3.14,
                        "rad": 20.0,
                        "h": 50,
                        "dead": False,
                        "sht": False,
                        "jc": 0,
                        "ac": 2,
                    },
                    "arrow-1": {
                        "id": "arrow-1",
                        "objectType": "arrow",
                        "x": 400.0,
                        "y": 200.0,
                        "dx": 20.0,
                        "dy": 0.0,
                        "ag": False,
                    },
                    "block-1": {
                        "id": "block-1",
                        "objectType": "block",
                        "pts": [
                            {"x": 0.0, "y": 780.0},
                            {"x": 800.0, "y": 780.0},
                            {"x": 800.0, "y": 800.0},
                            {"x": 0.0, "y": 800.0},
                        ],
                    },
                },
                "events": [],
            }
        )

    def test_game_state_from_update(self, game_update: GameUpdate) -> None:
        """Test GameState.from_update creates typed objects."""
        state = GameState.from_update(game_update)

        assert len(state.players) == 2
        assert len(state.arrows) == 1
        assert len(state.blocks) == 1
        assert len(state.bullets) == 0

        # Verify typed access
        player1 = state.players["player-1"]
        assert isinstance(player1, PlayerState)
        assert player1.name == "Bot1"
        assert player1.health == 100

        arrow1 = state.arrows["arrow-1"]
        assert isinstance(arrow1, ArrowState)
        assert arrow1.dx == 20.0

    def test_game_state_incremental_update(self, game_update: GameUpdate) -> None:
        """Test incremental update merges with existing state."""
        initial_state = GameState.from_update(game_update)

        # Incremental update: player moved, arrow destroyed
        incremental = GameUpdate.model_validate(
            {
                "fullUpdate": False,
                "objectStates": {
                    "player-1": {
                        "id": "player-1",
                        "objectType": "player",
                        "name": "Bot1",
                        "x": 150.0,  # Moved
                        "y": 200.0,
                        "dx": 10.0,
                        "dy": 0.0,
                        "dir": 0.0,
                        "rad": 20.0,
                        "h": 100,
                        "dead": False,
                        "sht": False,
                        "jc": 0,
                        "ac": 4,
                    },
                    "arrow-1": None,  # Destroyed
                },
                "events": [],
            }
        )

        new_state = GameState.from_update(incremental, existing_state=initial_state)

        # Player updated
        assert new_state.players["player-1"].x == 150.0
        assert new_state.players["player-1"].dx == 10.0

        # Player 2 still exists from previous state
        assert "player-2" in new_state.players

        # Arrow removed
        assert "arrow-1" not in new_state.arrows

        # Block still exists
        assert "block-1" in new_state.blocks

    def test_get_player_by_name(self, game_update: GameUpdate) -> None:
        """Test finding player by name."""
        state = GameState.from_update(game_update)

        bot1 = state.get_player_by_name("Bot1")
        assert bot1 is not None
        assert bot1.id == "player-1"

        unknown = state.get_player_by_name("Unknown")
        assert unknown is None

    def test_get_living_players(self, game_update: GameUpdate) -> None:
        """Test getting living players."""
        state = GameState.from_update(game_update)
        living = state.get_living_players()
        assert len(living) == 2

        # Kill one player
        state.players["player-1"].dead = True
        living = state.get_living_players()
        assert len(living) == 1
        assert living[0].id == "player-2"

    def test_get_active_arrows(self, game_update: GameUpdate) -> None:
        """Test getting active arrows."""
        state = GameState.from_update(game_update)
        active = state.get_active_arrows()
        assert len(active) == 1

        # Destroy the arrow
        state.arrows["arrow-1"].destroyed = True
        active = state.get_active_arrows()
        assert len(active) == 0


class TestHTTPAPIModels:
    """Tests for HTTP API request/response models."""

    def test_create_game_request_serialization(self) -> None:
        """Test CreateGameRequest serializes with correct aliases."""
        request = CreateGameRequest.model_validate(
            {
                "roomName": "Test Room",
                "playerName": "TestBot",
                "mapType": "default",
            }
        )
        data = request.model_dump(by_alias=True)
        assert data["roomName"] == "Test Room"
        assert data["playerName"] == "TestBot"
        assert data["mapType"] == "default"

    def test_create_game_response_parsing(self) -> None:
        """Test CreateGameResponse parsing from server JSON."""
        data = {
            "success": True,
            "roomId": "room-123",
            "roomCode": "ABC123",
            "roomName": "Test Room",
            "roomPassword": "secret-pass",
            "playerId": "player-456",
            "playerToken": "secret-token",
            "canvasSizeX": 800,
            "canvasSizeY": 800,
        }
        response = CreateGameResponse.model_validate(data)
        assert response.success is True
        assert response.room_id == "room-123"
        assert response.room_code == "ABC123"
        assert response.room_password == "secret-pass"
        assert response.player_id == "player-456"
        assert response.player_token == "secret-token"
        assert response.canvas_size_x == 800

    def test_join_game_request_serialization(self) -> None:
        """Test JoinGameRequest serializes correctly."""
        request = JoinGameRequest.model_validate(
            {
                "roomCode": "ABC123",
                "playerName": "Joiner",
                "roomPassword": "SECRET",
            }
        )
        data = request.model_dump(by_alias=True)
        assert data["roomCode"] == "ABC123"
        assert data["playerName"] == "Joiner"
        assert data["roomPassword"] == "SECRET"

    def test_bot_action_request(self) -> None:
        """Test BotActionRequest with multiple actions."""
        request = BotActionRequest.model_validate(
            {
                "actions": [
                    {"type": "key", "key": "W", "isDown": True},
                    {"type": "direction", "direction": 1.57},
                    {
                        "type": "click",
                        "x": 400.0,
                        "y": 300.0,
                        "isDown": True,
                        "button": 0,
                    },
                ]
            }
        )
        data = request.model_dump(by_alias=True)
        assert len(data["actions"]) == 3
        assert data["actions"][0]["key"] == "W"
        assert data["actions"][0]["isDown"] is True
        assert data["actions"][1]["direction"] == 1.57
        assert data["actions"][2]["button"] == 0


class TestWebSocketModels:
    """Tests for WebSocket message models."""

    def test_key_status_request_serialization(self) -> None:
        """Test KeyStatusRequest serializes with correct alias."""
        request = KeyStatusRequest.model_validate({"key": "W", "isDown": True})
        data = request.model_dump(by_alias=True)
        assert data["key"] == "W"
        assert data["isDown"] is True

    def test_player_click_request(self) -> None:
        """Test PlayerClickRequest serialization."""
        request = PlayerClickRequest.model_validate(
            {"x": 400.0, "y": 300.0, "isDown": True, "button": 0}
        )
        data = request.model_dump(by_alias=True)
        assert data["x"] == 400.0
        assert data["y"] == 300.0
        assert data["isDown"] is True
        assert data["button"] == 0

    def test_rejoin_game_request(self) -> None:
        """Test RejoinGameRequest serialization."""
        request = RejoinGameRequest.model_validate(
            {
                "roomId": "room-123",
                "playerId": "player-456",
                "playerToken": "token-789",
            }
        )
        data = request.model_dump(by_alias=True)
        assert data["roomId"] == "room-123"
        assert data["playerId"] == "player-456"
        assert data["playerToken"] == "token-789"

    def test_rejoin_game_response_parsing(self) -> None:
        """Test RejoinGameResponse parsing."""
        data = {
            "success": True,
            "roomName": "Test Room",
            "roomCode": "ABC123",
            "playerName": "Bot",
            "playerId": "player-123",
        }
        response = RejoinGameResponse.model_validate(data)
        assert response.success is True
        assert response.room_name == "Test Room"
        assert response.player_id == "player-123"

    def test_websocket_message(self) -> None:
        """Test WebSocketMessage structure."""
        message = WebSocketMessage(
            type="Key",
            payload={"key": "W", "isDown": True},
        )
        data = message.model_dump()
        assert data["type"] == "Key"
        assert data["payload"]["key"] == "W"

    def test_message_types_constants(self) -> None:
        """Test MessageTypes constants."""
        assert MessageTypes.KEY == "Key"
        assert MessageTypes.CLICK == "Click"
        assert MessageTypes.GAME_UPDATE == "GameUpdate"
        assert MessageTypes.REJOIN_GAME == "RejoinGame"


class TestConstants:
    """Tests for game constants."""

    def test_game_constants_values(self) -> None:
        """Test GameConstants has correct default values."""
        assert GAME_CONSTANTS.GRAVITY_METERS_PER_SEC2 == 20.0
        assert GAME_CONSTANTS.PX_PER_METER == 20.0
        assert GAME_CONSTANTS.ROOM_SIZE_PIXELS_X == 800.0
        assert GAME_CONSTANTS.PLAYER_RADIUS == 20.0
        assert GAME_CONSTANTS.PLAYER_MAX_JUMPS == 2
        assert GAME_CONSTANTS.PLAYER_STARTING_ARROWS == 4
        assert GAME_CONSTANTS.ARROW_MAX_POWER_TIME_SEC == 2.0

    def test_object_types_constants(self) -> None:
        """Test ObjectTypes constants."""
        assert ObjectTypes.PLAYER == "player"
        assert ObjectTypes.ARROW == "arrow"
        assert ObjectTypes.BLOCK == "block"
        assert ObjectTypes.BULLET == "bullet"

    def test_state_keys_constants(self) -> None:
        """Test StateKeys constants."""
        assert StateKeys.ID == "id"
        assert StateKeys.DIRECTION == "dir"
        assert StateKeys.HEALTH == "h"
        assert StateKeys.DEAD == "dead"
        assert StateKeys.ARROW_GROUNDED == "ag"

    def test_game_constants_instantiation(self) -> None:
        """Test GameConstants can be instantiated with custom values."""
        custom = GameConstants(GRAVITY_METERS_PER_SEC2=10.0)
        assert custom.GRAVITY_METERS_PER_SEC2 == 10.0
        # Other values remain default
        assert custom.PX_PER_METER == 20.0


class TestImports:
    """Tests to verify all imports work correctly."""

    def test_all_exports_importable(self) -> None:
        """Test all __all__ exports are importable."""
        from bot.models import __all__

        for name in __all__:
            # This will raise if import fails
            obj = getattr(__import__("bot.models", fromlist=[name]), name)
            assert obj is not None

    def test_direct_module_imports(self) -> None:
        """Test imports from specific modules work."""
        from bot.models.api import CreateGameRequest, JoinGameResponse
        from bot.models.base import BaseObjectState, Point
        from bot.models.constants import GAME_CONSTANTS, GameConstants
        from bot.models.game_objects import ArrowState, BlockState, PlayerState
        from bot.models.game_state import GameState, GameUpdate
        from bot.models.websocket import KeyStatusRequest, WebSocketMessage

        # All imports successful
        assert Point is not None
        assert BaseObjectState is not None
        assert PlayerState is not None
        assert ArrowState is not None
        assert BlockState is not None
        assert GameState is not None
        assert GameUpdate is not None
        assert CreateGameRequest is not None
        assert JoinGameResponse is not None
        assert KeyStatusRequest is not None
        assert WebSocketMessage is not None
        assert GAME_CONSTANTS is not None
        assert GameConstants is not None
