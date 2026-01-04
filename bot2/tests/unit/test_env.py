"""Unit tests for TowerfallEnv gymnasium environment.

Tests cover:
- Environment instantiation with various configs
- Action space and observation space validation
- Reset returns correct observation shape
- Step returns correct tuple structure
- Reward calculation logic
- Termination detection
- Proper cleanup on close
- Integration with mock server
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import gymnasium as gym
import numpy as np
import pytest

from bot.actions import ACTION_SPACE_SIZE
from bot.gym import TowerfallEnv
from bot.models import (
    CreateGameResponse,
    GameState,
    GetGameStateResponse,
    PlayerState,
    PlayerStatsDTO,
    ResetGameResponse,
)
from bot.observation import ObservationConfig


def make_create_game_response(**overrides: Any) -> CreateGameResponse:
    """Create a CreateGameResponse for testing."""
    data: dict[str, Any] = {
        "success": True,
        "roomId": "room-123",
        "roomCode": "ABC123",
        "roomName": "Test Room",
        "roomPassword": "test-pass",
        "playerId": "player-456",
        "playerToken": "token-789",
        "canvasSizeX": 800,
        "canvasSizeY": 600,
    }
    data.update(overrides)
    return CreateGameResponse.model_validate(data)


def make_get_game_state_response(
    players: dict[str, dict[str, Any]] | None = None,
    **overrides: Any,
) -> GetGameStateResponse:
    """Create a GetGameStateResponse for testing."""
    object_states = {}
    if players:
        object_states.update(players)

    data: dict[str, Any] = {
        "success": True,
        "roomId": "room-123",
        "gameUpdate": {
            "fullUpdate": True,
            "objectStates": object_states,
        },
    }
    data.update(overrides)
    return GetGameStateResponse.model_validate(data)


def make_player_state_dict(
    player_id: str = "player-456",
    **overrides: Any,
) -> dict[str, Any]:
    """Create a player state dict for testing."""
    data: dict[str, Any] = {
        "id": player_id,
        "objectType": "player",
        "name": "TestPlayer",
        "x": 400.0,
        "y": 300.0,
        "dx": 0.0,
        "dy": 0.0,
        "dir": 0.0,
        "rad": 20.0,
        "h": 100,
        "dead": False,
        "sht": False,
        "jc": 2,
        "ac": 3,
    }
    data.update(overrides)
    return data


def make_reset_game_response(**overrides: Any) -> ResetGameResponse:
    """Create a ResetGameResponse for testing."""
    data: dict[str, Any] = {
        "success": True,
        "roomId": "room-123",
        "mapType": "arena1",
        "canvasSizeX": 800,
        "canvasSizeY": 600,
    }
    data.update(overrides)
    return ResetGameResponse.model_validate(data)


def make_player_stats(**overrides: Any) -> PlayerStatsDTO:
    """Create a PlayerStatsDTO for testing."""
    data: dict[str, Any] = {
        "playerId": "player-456",
        "playerName": "TestPlayer",
        "kills": 0,
        "deaths": 0,
    }
    data.update(overrides)
    return PlayerStatsDTO.model_validate(data)


class TestTowerfallEnvInit:
    """Tests for TowerfallEnv initialization."""

    def test_default_configuration(self) -> None:
        """Test environment initializes with default configuration."""
        env = TowerfallEnv()

        assert env.http_url == "http://localhost:4000"
        assert env.player_name == "MLBot"
        assert env.room_name == "Training"
        assert env.map_type == "default"
        assert env.opponent_type == "rule_based"
        assert env.tick_rate_multiplier == 1.0
        assert env.max_episode_steps == 1000
        assert env.render_mode is None

    def test_custom_configuration(self) -> None:
        """Test environment with custom configuration."""
        env = TowerfallEnv(
            http_url="http://example.com:8080",
            player_name="CustomBot",
            room_name="CustomRoom",
            map_type="arena2",
            opponent_type="none",
            tick_rate_multiplier=10.0,
            max_episode_steps=500,
            render_mode="human",
        )

        assert env.http_url == "http://example.com:8080"
        assert env.player_name == "CustomBot"
        assert env.room_name == "CustomRoom"
        assert env.map_type == "arena2"
        assert env.opponent_type == "none"
        assert env.tick_rate_multiplier == 10.0
        assert env.max_episode_steps == 500
        assert env.render_mode == "human"

    def test_custom_observation_config(self) -> None:
        """Test environment with custom observation configuration."""
        custom_config = ObservationConfig(
            max_other_players=1,
            max_tracked_arrows=4,
            include_map=False,
        )
        env = TowerfallEnv(observation_config=custom_config)

        assert env._obs_config.max_other_players == 1
        assert env._obs_config.max_tracked_arrows == 4
        assert env._obs_config.include_map is False


class TestTowerfallEnvSpaces:
    """Tests for action and observation spaces."""

    def test_action_space_is_discrete_27(self) -> None:
        """Test action space is Discrete(27)."""
        env = TowerfallEnv()

        assert isinstance(env.action_space, gym.spaces.Discrete)
        assert env.action_space.n == ACTION_SPACE_SIZE
        assert env.action_space.n == 27

    def test_observation_space_is_box(self) -> None:
        """Test observation space is Box with correct bounds."""
        env = TowerfallEnv()

        assert isinstance(env.observation_space, gym.spaces.Box)
        assert env.observation_space.low.min() == -1.0
        assert env.observation_space.high.max() == 1.0
        assert env.observation_space.dtype == np.float32

    def test_observation_space_shape_matches_config(self) -> None:
        """Test observation space shape matches observation config."""
        config = ObservationConfig()
        env = TowerfallEnv(observation_config=config)

        assert env.observation_space.shape == (config.total_size,)

    def test_observation_space_with_custom_config(self) -> None:
        """Test observation space adapts to custom config."""
        custom_config = ObservationConfig(
            max_other_players=1,
            max_tracked_arrows=4,
            include_map=False,
        )
        env = TowerfallEnv(observation_config=custom_config)

        assert env.observation_space.shape == (custom_config.total_size,)

    def test_action_space_sample_is_valid(self) -> None:
        """Test sampled actions are valid."""
        env = TowerfallEnv()

        for _ in range(100):
            action = env.action_space.sample()
            assert 0 <= action < 27


class TestTowerfallEnvReset:
    """Tests for reset() method."""

    def test_reset_returns_correct_tuple(self) -> None:
        """Test reset returns (observation, info) tuple."""
        env = TowerfallEnv()

        player_state = make_player_state_dict()
        game_state = GameState(
            players={"player-456": PlayerState.model_validate(player_state)},
            canvas_size_x=800,
            canvas_size_y=600,
        )
        info_dict = {
            "room_id": "room-123",
            "room_code": "ABC123",
            "player_id": "player-456",
        }

        with patch.object(
            env, "_async_reset", new=AsyncMock(return_value=(game_state, info_dict))
        ):
            env._client = MagicMock()
            env._client.player_id = "player-456"

            obs, info = env.reset()

            assert isinstance(obs, np.ndarray)
            assert isinstance(info, dict)
            assert obs.dtype == np.float32

    def test_reset_observation_shape(self) -> None:
        """Test reset returns observation with correct shape."""
        env = TowerfallEnv()

        player_state = make_player_state_dict()
        game_state = GameState(
            players={"player-456": PlayerState.model_validate(player_state)},
            canvas_size_x=800,
            canvas_size_y=600,
        )
        info_dict = {
            "room_id": "room-123",
            "room_code": "ABC123",
            "player_id": "player-456",
        }

        with patch.object(
            env, "_async_reset", new=AsyncMock(return_value=(game_state, info_dict))
        ):
            env._client = MagicMock()
            env._client.player_id = "player-456"

            obs, _ = env.reset()

            assert obs.shape == env.observation_space.shape

    def test_reset_resets_episode_step(self) -> None:
        """Test reset resets episode step counter."""
        env = TowerfallEnv()
        env._episode_step = 100

        player_state = make_player_state_dict()
        game_state = GameState(
            players={"player-456": PlayerState.model_validate(player_state)},
            canvas_size_x=800,
            canvas_size_y=600,
        )
        info_dict = {
            "room_id": "room-123",
            "room_code": "ABC123",
            "player_id": "player-456",
        }

        with patch.object(
            env, "_async_reset", new=AsyncMock(return_value=(game_state, info_dict))
        ):
            env._client = MagicMock()
            env._client.player_id = "player-456"

            env.reset()

            assert env._episode_step == 0

    def test_reset_info_contains_room_info(self) -> None:
        """Test reset info contains room information."""
        env = TowerfallEnv()

        player_state = make_player_state_dict()
        game_state = GameState(
            players={"player-456": PlayerState.model_validate(player_state)},
            canvas_size_x=800,
            canvas_size_y=600,
        )
        info_dict = {
            "room_id": "room-123",
            "room_code": "ABC123",
            "player_id": "player-456",
        }

        with patch.object(
            env, "_async_reset", new=AsyncMock(return_value=(game_state, info_dict))
        ):
            env._client = MagicMock()
            env._client.player_id = "player-456"

            _, info = env.reset()

            assert "room_id" in info
            assert "room_code" in info
            assert "player_id" in info


class TestTowerfallEnvStep:
    """Tests for step() method."""

    def test_step_returns_correct_tuple(self) -> None:
        """Test step returns (obs, reward, terminated, truncated, info) tuple."""
        env = TowerfallEnv()
        env._client = MagicMock()
        env._client.player_id = "player-456"
        env._episode_step = 0

        player_state = make_player_state_dict()
        game_state = GameState(
            players={"player-456": PlayerState.model_validate(player_state)},
            canvas_size_x=800,
            canvas_size_y=600,
        )
        stats = {"player-456": make_player_stats()}

        with patch.object(
            env, "_async_step", new=AsyncMock(return_value=(game_state, stats))
        ):
            obs, reward, terminated, truncated, info = env.step(0)

            assert isinstance(obs, np.ndarray)
            assert isinstance(reward, float)
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            assert isinstance(info, dict)

    def test_step_observation_shape(self) -> None:
        """Test step returns observation with correct shape."""
        env = TowerfallEnv()
        env._client = MagicMock()
        env._client.player_id = "player-456"
        env._episode_step = 0

        player_state = make_player_state_dict()
        game_state = GameState(
            players={"player-456": PlayerState.model_validate(player_state)},
            canvas_size_x=800,
            canvas_size_y=600,
        )
        stats = {"player-456": make_player_stats()}

        with patch.object(
            env, "_async_step", new=AsyncMock(return_value=(game_state, stats))
        ):
            obs, _, _, _, _ = env.step(0)

            assert obs.shape == env.observation_space.shape

    def test_step_increments_episode_step(self) -> None:
        """Test step increments episode step counter."""
        env = TowerfallEnv()
        env._client = MagicMock()
        env._client.player_id = "player-456"
        env._episode_step = 5

        player_state = make_player_state_dict()
        game_state = GameState(
            players={"player-456": PlayerState.model_validate(player_state)},
            canvas_size_x=800,
            canvas_size_y=600,
        )
        stats = {"player-456": make_player_stats()}

        with patch.object(
            env, "_async_step", new=AsyncMock(return_value=(game_state, stats))
        ):
            env.step(0)

            assert env._episode_step == 6

    def test_step_truncates_at_max_steps(self) -> None:
        """Test step returns truncated=True at max steps."""
        env = TowerfallEnv(max_episode_steps=100)
        env._client = MagicMock()
        env._client.player_id = "player-456"
        env._episode_step = 99  # Will become 100 after step
        # Sync termination tracker with episode step
        env._termination_tracker._episode_timesteps = 99

        player_state = make_player_state_dict()
        game_state = GameState(
            players={"player-456": PlayerState.model_validate(player_state)},
            canvas_size_x=800,
            canvas_size_y=600,
        )
        stats = {"player-456": make_player_stats()}

        with patch.object(
            env, "_async_step", new=AsyncMock(return_value=(game_state, stats))
        ):
            _, _, _, truncated, _ = env.step(0)

            assert truncated is True

    def test_step_not_initialized_raises(self) -> None:
        """Test step raises if environment not initialized."""
        env = TowerfallEnv()
        env._client = None

        with pytest.raises(RuntimeError, match="not initialized"):
            env.step(0)

    def test_step_info_contains_episode_step(self) -> None:
        """Test step info contains episode step."""
        env = TowerfallEnv()
        env._client = MagicMock()
        env._client.player_id = "player-456"
        env._episode_step = 10

        player_state = make_player_state_dict()
        game_state = GameState(
            players={"player-456": PlayerState.model_validate(player_state)},
            canvas_size_x=800,
            canvas_size_y=600,
        )
        stats = {"player-456": make_player_stats()}

        with patch.object(
            env, "_async_step", new=AsyncMock(return_value=(game_state, stats))
        ):
            _, _, _, _, info = env.step(0)

            assert "episode_step" in info
            assert info["episode_step"] == 11


class TestTowerfallEnvReward:
    """Tests for reward calculation."""

    def test_reward_for_kill(self) -> None:
        """Test reward increases for kills."""
        env = TowerfallEnv()
        env._client = MagicMock()
        env._client.player_id = "player-456"
        # Reset reward function to initial state
        env._reward_fn.reset(None)

        player_state = make_player_state_dict()
        game_state = GameState(
            players={"player-456": PlayerState.model_validate(player_state)},
            canvas_size_x=800,
            canvas_size_y=600,
        )

        # Stats show 1 kill
        stats = {"player-456": make_player_stats(kills=1, deaths=0)}

        reward = env._calculate_reward(game_state, stats)

        # Should be +1.0 for kill, -0.001 for step
        assert reward == pytest.approx(1.0 - 0.001, abs=0.0001)
        assert env._reward_fn.get_episode_stats()["prev_kills"] == 1

    def test_reward_for_death(self) -> None:
        """Test reward decreases for deaths."""
        env = TowerfallEnv()
        env._client = MagicMock()
        env._client.player_id = "player-456"
        # Reset reward function to initial state
        env._reward_fn.reset(None)

        player_state = make_player_state_dict()
        game_state = GameState(
            players={"player-456": PlayerState.model_validate(player_state)},
            canvas_size_x=800,
            canvas_size_y=600,
        )

        # Stats show 1 death
        stats = {"player-456": make_player_stats(kills=0, deaths=1)}

        reward = env._calculate_reward(game_state, stats)

        # Should be -1.0 for death, -0.001 for step
        assert reward == pytest.approx(-1.0 - 0.001, abs=0.0001)
        assert env._reward_fn.get_episode_stats()["prev_deaths"] == 1

    def test_reward_step_penalty(self) -> None:
        """Test small negative reward per step."""
        env = TowerfallEnv()
        env._client = MagicMock()
        env._client.player_id = "player-456"
        # Reset reward function to initial state
        env._reward_fn.reset(None)

        player_state = make_player_state_dict()
        game_state = GameState(
            players={"player-456": PlayerState.model_validate(player_state)},
            canvas_size_x=800,
            canvas_size_y=600,
        )

        # No kills or deaths
        stats = {"player-456": make_player_stats(kills=0, deaths=0)}

        reward = env._calculate_reward(game_state, stats)

        # Should just be step penalty
        assert reward == pytest.approx(-0.001, abs=0.0001)

    def test_reward_tracks_delta(self) -> None:
        """Test reward tracks delta not absolute values."""
        env = TowerfallEnv()
        env._client = MagicMock()
        env._client.player_id = "player-456"
        # Initialize reward function with previous state
        env._reward_fn.reset(make_player_stats(kills=5, deaths=2))

        player_state = make_player_state_dict()
        game_state = GameState(
            players={"player-456": PlayerState.model_validate(player_state)},
            canvas_size_x=800,
            canvas_size_y=600,
        )

        # Stats show 6 kills, 2 deaths (1 new kill, 0 new deaths)
        stats = {"player-456": make_player_stats(kills=6, deaths=2)}

        reward = env._calculate_reward(game_state, stats)

        # Should be +1.0 for new kill, -0.001 for step
        assert reward == pytest.approx(1.0 - 0.001, abs=0.0001)


class TestTowerfallEnvTermination:
    """Tests for termination detection.

    Note: Detailed termination tests are in test_termination.py.
    These tests verify the _check_game_over method uses the server's is_game_over signal.
    """

    def test_game_over_when_server_signals_game_over(self) -> None:
        """Test game over when server sets is_game_over to True."""
        env = TowerfallEnv()

        game_state = GameState(
            players={},
            canvas_size_x=800,
            canvas_size_y=600,
            is_game_over=True,
        )

        is_game_over = env._check_game_over(game_state)

        assert is_game_over is True

    def test_not_game_over_when_server_signals_not_over(self) -> None:
        """Test not game over when server sets is_game_over to False."""
        env = TowerfallEnv()

        game_state = GameState(
            players={},
            canvas_size_x=800,
            canvas_size_y=600,
            is_game_over=False,
        )

        is_game_over = env._check_game_over(game_state)

        assert is_game_over is False

    def test_not_game_over_by_default(self) -> None:
        """Test is_game_over defaults to False."""
        env = TowerfallEnv()

        # No is_game_over specified, should default to False
        game_state = GameState(
            players={},
            canvas_size_x=800,
            canvas_size_y=600,
        )

        is_game_over = env._check_game_over(game_state)

        assert is_game_over is False


class TestTowerfallEnvRender:
    """Tests for render() method."""

    def test_render_none_mode(self) -> None:
        """Test render returns None when render_mode is None."""
        env = TowerfallEnv(render_mode=None)

        result = env.render()

        assert result is None

    def test_render_human_mode(self) -> None:
        """Test render in human mode returns None (prints to console)."""
        env = TowerfallEnv(render_mode="human")
        env._client = MagicMock()
        env._client.room_code = "ABC123"
        env._episode_step = 10

        result = env.render()

        assert result is None

    def test_render_rgb_array_mode(self) -> None:
        """Test render in rgb_array mode returns numpy array."""
        env = TowerfallEnv(render_mode="rgb_array")

        result = env.render()

        assert isinstance(result, np.ndarray)
        assert result.shape == (600, 800, 3)
        assert result.dtype == np.uint8


class TestTowerfallEnvClose:
    """Tests for close() method."""

    def test_close_clears_client(self) -> None:
        """Test close clears client reference."""
        env = TowerfallEnv()
        env._client = MagicMock()
        env._client.close = AsyncMock()

        env.close()

        assert env._client is None

    def test_close_handles_no_client(self) -> None:
        """Test close handles case when client is None."""
        env = TowerfallEnv()
        env._client = None

        # Should not raise
        env.close()


class TestTowerfallEnvRegistration:
    """Tests for gymnasium environment registration."""

    def test_env_registered(self) -> None:
        """Test environment is registered with gymnasium."""
        # Force import to trigger registration
        import bot.gym  # noqa: F401

        # Check environment is registered
        spec = gym.spec("Towerfall-v0")

        assert spec is not None
        assert spec.id == "Towerfall-v0"
        assert spec.max_episode_steps == 1000

    def test_make_creates_env(self) -> None:
        """Test gym.make creates TowerfallEnv."""
        # Force import to trigger registration
        import bot.gym  # noqa: F401

        env = gym.make("Towerfall-v0")

        assert isinstance(env.unwrapped, TowerfallEnv)
        env.close()


class TestTowerfallEnvAsyncBridge:
    """Tests for async/sync bridge functionality."""

    def test_get_loop_creates_loop(self) -> None:
        """Test _get_loop creates event loop if needed."""
        env = TowerfallEnv()
        env._loop = None

        loop = env._get_loop()

        assert loop is not None
        assert env._loop is loop

        # Cleanup
        if env._owns_loop:
            loop.close()

    def test_get_loop_reuses_loop(self) -> None:
        """Test _get_loop reuses existing loop."""
        import asyncio

        env = TowerfallEnv()
        existing_loop = asyncio.new_event_loop()
        env._loop = existing_loop
        env._owns_loop = True

        loop = env._get_loop()

        assert loop is existing_loop

        # Cleanup
        existing_loop.close()


class TestImports:
    """Tests that all imports work correctly."""

    def test_import_from_bot_env(self) -> None:
        """Test importing TowerfallEnv from bot.gym."""
        from bot.gym import TowerfallEnv

        assert TowerfallEnv is not None

    def test_env_is_gymnasium_env(self) -> None:
        """Test TowerfallEnv is a gymnasium.Env subclass."""
        from bot.gym import TowerfallEnv

        assert issubclass(TowerfallEnv, gym.Env)
