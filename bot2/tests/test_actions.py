"""Unit tests for the discrete action space module.

Tests cover:
- Action enum values and ranges
- aim_action_to_radians() conversion
- radians_to_aim_action() conversion
- Action category checks (is_aim_action, is_movement_action, is_shoot_action)
- execute_action() with mocked GameClient
"""

import math
from unittest.mock import AsyncMock, MagicMock

import pytest

from bot.actions import (
    ACTION_SPACE_SIZE,
    NUM_AIM_BUCKETS,
    RADIANS_PER_AIM_BUCKET,
    Action,
    action_to_string,
    aim_action_to_radians,
    execute_action,
    get_action_category,
    is_aim_action,
    is_movement_action,
    is_shoot_action,
    radians_to_aim_action,
)


class TestActionEnum:
    """Tests for the Action enum."""

    def test_action_space_size(self) -> None:
        """Test that ACTION_SPACE_SIZE matches the enum."""
        assert ACTION_SPACE_SIZE == 27
        assert len(Action) == 27

    def test_movement_action_values(self) -> None:
        """Test movement action values are in range 0-7."""
        assert Action.MOVE_LEFT_PRESS == 0
        assert Action.MOVE_LEFT_RELEASE == 1
        assert Action.MOVE_RIGHT_PRESS == 2
        assert Action.MOVE_RIGHT_RELEASE == 3
        assert Action.JUMP_PRESS == 4
        assert Action.JUMP_RELEASE == 5
        assert Action.DIVE_PRESS == 6
        assert Action.DIVE_RELEASE == 7

    def test_aim_action_values(self) -> None:
        """Test aim action values are in range 8-23."""
        assert Action.AIM_0 == 8
        assert Action.AIM_15 == 23

        # Verify all 16 aim actions exist
        for i in range(16):
            action = Action(8 + i)
            assert action.name == f"AIM_{i}"

    def test_shoot_action_values(self) -> None:
        """Test shoot action values."""
        assert Action.SHOOT_START == 24
        assert Action.SHOOT_RELEASE == 25

    def test_noop_action_value(self) -> None:
        """Test no-op action value."""
        assert Action.NO_OP == 26

    def test_action_values_are_contiguous(self) -> None:
        """Test that action values form a contiguous range from 0 to 26."""
        values = sorted(action.value for action in Action)
        assert values == list(range(27))


class TestAimConstants:
    """Tests for aim-related constants."""

    def test_num_aim_buckets(self) -> None:
        """Test number of aim buckets."""
        assert NUM_AIM_BUCKETS == 16

    def test_radians_per_bucket(self) -> None:
        """Test radians per aim bucket (should be π/8)."""
        expected = math.pi / 8
        assert RADIANS_PER_AIM_BUCKET == pytest.approx(expected)

    def test_full_circle_coverage(self) -> None:
        """Test that all buckets cover a full circle."""
        total_radians = NUM_AIM_BUCKETS * RADIANS_PER_AIM_BUCKET
        assert total_radians == pytest.approx(2 * math.pi)


class TestAimActionToRadians:
    """Tests for aim_action_to_radians() function."""

    def test_aim_0_is_zero_radians(self) -> None:
        """Test AIM_0 maps to 0 radians (right)."""
        assert aim_action_to_radians(Action.AIM_0) == 0.0

    def test_aim_4_is_half_pi(self) -> None:
        """Test AIM_4 maps to π/2 radians (down)."""
        result = aim_action_to_radians(Action.AIM_4)
        assert result == pytest.approx(math.pi / 2)

    def test_aim_8_is_pi(self) -> None:
        """Test AIM_8 maps to π radians (left)."""
        result = aim_action_to_radians(Action.AIM_8)
        assert result == pytest.approx(math.pi)

    def test_aim_12_is_three_halves_pi(self) -> None:
        """Test AIM_12 maps to 3π/2 radians (up)."""
        result = aim_action_to_radians(Action.AIM_12)
        assert result == pytest.approx(3 * math.pi / 2)

    def test_all_aim_actions(self) -> None:
        """Test all aim actions produce expected radians."""
        for i in range(16):
            action = Action(Action.AIM_0 + i)
            expected = i * (math.pi / 8)
            result = aim_action_to_radians(action)
            assert result == pytest.approx(expected), f"AIM_{i} failed"

    def test_accepts_integer(self) -> None:
        """Test function accepts integer values."""
        # AIM_4 = 12
        result = aim_action_to_radians(12)
        assert result == pytest.approx(math.pi / 2)

    def test_rejects_non_aim_action(self) -> None:
        """Test function raises ValueError for non-aim actions."""
        with pytest.raises(ValueError, match="not an aim action"):
            aim_action_to_radians(Action.MOVE_LEFT_PRESS)

        with pytest.raises(ValueError, match="not an aim action"):
            aim_action_to_radians(Action.SHOOT_START)

        with pytest.raises(ValueError, match="not an aim action"):
            aim_action_to_radians(Action.NO_OP)


class TestRadiansToAimAction:
    """Tests for radians_to_aim_action() function."""

    def test_zero_radians(self) -> None:
        """Test 0 radians maps to AIM_0."""
        assert radians_to_aim_action(0.0) == Action.AIM_0

    def test_half_pi(self) -> None:
        """Test π/2 radians maps to AIM_4 (down)."""
        assert radians_to_aim_action(math.pi / 2) == Action.AIM_4

    def test_pi(self) -> None:
        """Test π radians maps to AIM_8 (left)."""
        assert radians_to_aim_action(math.pi) == Action.AIM_8

    def test_three_halves_pi(self) -> None:
        """Test 3π/2 radians maps to AIM_12 (up)."""
        assert radians_to_aim_action(3 * math.pi / 2) == Action.AIM_12

    def test_negative_radians(self) -> None:
        """Test negative radians are normalized."""
        # -π/2 = 3π/2 when normalized
        assert radians_to_aim_action(-math.pi / 2) == Action.AIM_12

    def test_large_radians(self) -> None:
        """Test radians > 2π are normalized."""
        # 3π = π when normalized
        assert radians_to_aim_action(3 * math.pi) == Action.AIM_8

    def test_roundtrip(self) -> None:
        """Test converting action to radians and back."""
        for action in range(Action.AIM_0, Action.AIM_15 + 1):
            radians = aim_action_to_radians(action)
            result = radians_to_aim_action(radians)
            assert result == Action(action)


class TestActionCategoryChecks:
    """Tests for action category check functions."""

    def test_is_movement_action(self) -> None:
        """Test is_movement_action correctly identifies movement actions."""
        # Movement actions
        assert is_movement_action(Action.MOVE_LEFT_PRESS) is True
        assert is_movement_action(Action.MOVE_LEFT_RELEASE) is True
        assert is_movement_action(Action.MOVE_RIGHT_PRESS) is True
        assert is_movement_action(Action.MOVE_RIGHT_RELEASE) is True
        assert is_movement_action(Action.JUMP_PRESS) is True
        assert is_movement_action(Action.JUMP_RELEASE) is True
        assert is_movement_action(Action.DIVE_PRESS) is True
        assert is_movement_action(Action.DIVE_RELEASE) is True

        # Non-movement actions
        assert is_movement_action(Action.AIM_0) is False
        assert is_movement_action(Action.SHOOT_START) is False
        assert is_movement_action(Action.NO_OP) is False

    def test_is_aim_action(self) -> None:
        """Test is_aim_action correctly identifies aim actions."""
        # Aim actions
        for i in range(16):
            assert is_aim_action(Action.AIM_0 + i) is True

        # Non-aim actions
        assert is_aim_action(Action.MOVE_LEFT_PRESS) is False
        assert is_aim_action(Action.SHOOT_START) is False
        assert is_aim_action(Action.NO_OP) is False

    def test_is_shoot_action(self) -> None:
        """Test is_shoot_action correctly identifies shooting actions."""
        assert is_shoot_action(Action.SHOOT_START) is True
        assert is_shoot_action(Action.SHOOT_RELEASE) is True

        assert is_shoot_action(Action.MOVE_LEFT_PRESS) is False
        assert is_shoot_action(Action.AIM_0) is False
        assert is_shoot_action(Action.NO_OP) is False


class TestGetActionCategory:
    """Tests for get_action_category() function."""

    def test_movement_category(self) -> None:
        """Test movement actions return 'movement' category."""
        assert get_action_category(Action.MOVE_LEFT_PRESS) == "movement"
        assert get_action_category(Action.JUMP_PRESS) == "movement"
        assert get_action_category(Action.DIVE_RELEASE) == "movement"

    def test_aim_category(self) -> None:
        """Test aim actions return 'aim' category."""
        assert get_action_category(Action.AIM_0) == "aim"
        assert get_action_category(Action.AIM_8) == "aim"
        assert get_action_category(Action.AIM_15) == "aim"

    def test_shoot_category(self) -> None:
        """Test shoot actions return 'shoot' category."""
        assert get_action_category(Action.SHOOT_START) == "shoot"
        assert get_action_category(Action.SHOOT_RELEASE) == "shoot"

    def test_noop_category(self) -> None:
        """Test NO_OP returns 'noop' category."""
        assert get_action_category(Action.NO_OP) == "noop"


class TestActionToString:
    """Tests for action_to_string() function."""

    def test_returns_action_name(self) -> None:
        """Test action_to_string returns the enum name."""
        assert action_to_string(Action.MOVE_LEFT_PRESS) == "MOVE_LEFT_PRESS"
        assert action_to_string(Action.AIM_0) == "AIM_0"
        assert action_to_string(Action.SHOOT_START) == "SHOOT_START"
        assert action_to_string(Action.NO_OP) == "NO_OP"

    def test_accepts_integer(self) -> None:
        """Test action_to_string accepts integer values."""
        assert action_to_string(0) == "MOVE_LEFT_PRESS"
        assert action_to_string(26) == "NO_OP"


class TestExecuteAction:
    """Tests for execute_action() function with mocked GameClient."""

    @pytest.fixture
    def mock_client(self) -> MagicMock:
        """Create a mock GameClient."""
        client = MagicMock()
        client.send_keyboard_input = AsyncMock()
        client.send_mouse_input = AsyncMock()
        client.room_id = "room-123"
        client.player_id = "player-456"
        client._http_client = MagicMock()
        client._http_client.submit_action = AsyncMock()

        # Mock ClientMode for REST mode
        from bot.client.game_client import ClientMode

        client.mode = ClientMode.REST

        return client

    @pytest.mark.asyncio
    async def test_noop_does_nothing(self, mock_client: MagicMock) -> None:
        """Test NO_OP action does nothing."""
        await execute_action(mock_client, Action.NO_OP)

        mock_client.send_keyboard_input.assert_not_called()
        mock_client.send_mouse_input.assert_not_called()
        mock_client._http_client.submit_action.assert_not_called()

    @pytest.mark.asyncio
    async def test_move_left_press(self, mock_client: MagicMock) -> None:
        """Test MOVE_LEFT_PRESS sends keyboard input."""
        await execute_action(mock_client, Action.MOVE_LEFT_PRESS)

        mock_client.send_keyboard_input.assert_called_once_with("A", True)

    @pytest.mark.asyncio
    async def test_move_left_release(self, mock_client: MagicMock) -> None:
        """Test MOVE_LEFT_RELEASE sends keyboard input."""
        await execute_action(mock_client, Action.MOVE_LEFT_RELEASE)

        mock_client.send_keyboard_input.assert_called_once_with("A", False)

    @pytest.mark.asyncio
    async def test_move_right_press(self, mock_client: MagicMock) -> None:
        """Test MOVE_RIGHT_PRESS sends keyboard input."""
        await execute_action(mock_client, Action.MOVE_RIGHT_PRESS)

        mock_client.send_keyboard_input.assert_called_once_with("D", True)

    @pytest.mark.asyncio
    async def test_move_right_release(self, mock_client: MagicMock) -> None:
        """Test MOVE_RIGHT_RELEASE sends keyboard input."""
        await execute_action(mock_client, Action.MOVE_RIGHT_RELEASE)

        mock_client.send_keyboard_input.assert_called_once_with("D", False)

    @pytest.mark.asyncio
    async def test_jump_press(self, mock_client: MagicMock) -> None:
        """Test JUMP_PRESS sends keyboard input."""
        await execute_action(mock_client, Action.JUMP_PRESS)

        mock_client.send_keyboard_input.assert_called_once_with("W", True)

    @pytest.mark.asyncio
    async def test_jump_release(self, mock_client: MagicMock) -> None:
        """Test JUMP_RELEASE sends keyboard input."""
        await execute_action(mock_client, Action.JUMP_RELEASE)

        mock_client.send_keyboard_input.assert_called_once_with("W", False)

    @pytest.mark.asyncio
    async def test_dive_press(self, mock_client: MagicMock) -> None:
        """Test DIVE_PRESS sends keyboard input."""
        await execute_action(mock_client, Action.DIVE_PRESS)

        mock_client.send_keyboard_input.assert_called_once_with("S", True)

    @pytest.mark.asyncio
    async def test_dive_release(self, mock_client: MagicMock) -> None:
        """Test DIVE_RELEASE sends keyboard input."""
        await execute_action(mock_client, Action.DIVE_RELEASE)

        mock_client.send_keyboard_input.assert_called_once_with("S", False)

    @pytest.mark.asyncio
    async def test_aim_action_rest_mode(self, mock_client: MagicMock) -> None:
        """Test aim action sends direction update in REST mode."""
        await execute_action(mock_client, Action.AIM_4)

        mock_client._http_client.submit_action.assert_called_once()
        call_args = mock_client._http_client.submit_action.call_args
        assert call_args[1]["room_id"] == "room-123"
        assert call_args[1]["player_id"] == "player-456"
        actions = call_args[1]["actions"]
        assert len(actions) == 1
        assert actions[0].type == "direction"
        assert actions[0].direction == pytest.approx(math.pi / 2)

    @pytest.mark.asyncio
    async def test_aim_action_websocket_mode(self, mock_client: MagicMock) -> None:
        """Test aim action sends WebSocket message in WebSocket mode."""
        from bot.client.game_client import ClientMode

        mock_client.mode = ClientMode.WEBSOCKET
        mock_client._websocket = MagicMock()
        mock_client._websocket.send = AsyncMock()

        await execute_action(mock_client, Action.AIM_8)

        mock_client._websocket.send.assert_called_once()
        import json

        sent_message = json.loads(mock_client._websocket.send.call_args[0][0])
        assert sent_message["type"] == "ClientState"
        assert sent_message["payload"]["direction"] == pytest.approx(math.pi)

    @pytest.mark.asyncio
    async def test_shoot_start(self, mock_client: MagicMock) -> None:
        """Test SHOOT_START sends mouse input."""
        await execute_action(mock_client, Action.SHOOT_START)

        mock_client.send_mouse_input.assert_called_once_with("left", True, 0.0, 0.0)

    @pytest.mark.asyncio
    async def test_shoot_release(self, mock_client: MagicMock) -> None:
        """Test SHOOT_RELEASE sends mouse input."""
        await execute_action(mock_client, Action.SHOOT_RELEASE)

        mock_client.send_mouse_input.assert_called_once_with("left", False, 0.0, 0.0)

    @pytest.mark.asyncio
    async def test_accepts_integer(self, mock_client: MagicMock) -> None:
        """Test execute_action accepts integer values."""
        # MOVE_RIGHT_PRESS = 2
        await execute_action(mock_client, 2)

        mock_client.send_keyboard_input.assert_called_once_with("D", True)


class TestGymnasiumIntegration:
    """Tests verifying Gymnasium compatibility."""

    def test_action_space_discrete(self) -> None:
        """Test action space can be used with Gymnasium Discrete space."""
        import gymnasium as gym

        action_space = gym.spaces.Discrete(ACTION_SPACE_SIZE)

        # Verify bounds
        assert action_space.n == 27

        # Verify all actions are within bounds
        for action in Action:
            assert 0 <= action.value < action_space.n

    def test_sample_actions_are_valid(self) -> None:
        """Test sampled actions from Gymnasium space are valid."""
        import gymnasium as gym

        action_space = gym.spaces.Discrete(ACTION_SPACE_SIZE)

        # Sample some actions and verify they're valid
        for _ in range(100):
            sampled = action_space.sample()
            action = Action(sampled)
            assert action in Action
