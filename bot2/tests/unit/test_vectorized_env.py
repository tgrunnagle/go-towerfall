"""Unit tests for VectorizedTowerfallEnv gymnasium environment.

Tests cover:
- Environment instantiation with various configs
- Action space and observation space validation
- Reset returns correct batched observation shape
- Step returns correct tuple structure with batched arrays
- Parallel execution via asyncio
- Automatic reset on termination
- Proper cleanup on close
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import gymnasium as gym
import numpy as np
import pytest

from bot.actions import ACTION_SPACE_SIZE
from bot.gym import VectorizedTowerfallEnv
from bot.models import (
    GameState,
    PlayerState,
    PlayerStatsDTO,
)
from bot.observation import ObservationConfig


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


def make_game_state(
    player_id: str = "player-456",
    is_game_over: bool = False,
    **overrides: Any,
) -> GameState:
    """Create a GameState for testing."""
    player_state = make_player_state_dict(player_id)
    return GameState(
        players={player_id: PlayerState.model_validate(player_state)},
        canvas_size_x=800,
        canvas_size_y=600,
        is_game_over=is_game_over,
        **overrides,
    )


class TestVectorizedEnvInit:
    """Tests for VectorizedTowerfallEnv initialization."""

    def test_default_configuration(self) -> None:
        """Test environment initializes with default configuration."""
        env = VectorizedTowerfallEnv(num_envs=4)

        assert env.num_envs == 4
        assert env.http_url == "http://localhost:4000"
        assert env.player_name == "MLBot"
        assert env.room_name_prefix == "Training"
        assert env.map_type == "default"
        assert env.opponent_type == "rule_based"
        assert env.tick_rate_multiplier == 1.0
        assert env.max_episode_steps == 1000
        assert env.render_mode is None

    def test_custom_configuration(self) -> None:
        """Test environment with custom configuration."""
        env = VectorizedTowerfallEnv(
            num_envs=8,
            http_url="http://example.com:8080",
            player_name="CustomBot",
            room_name_prefix="CustomRoom",
            map_type="arena2",
            opponent_type="none",
            tick_rate_multiplier=10.0,
            max_episode_steps=500,
            render_mode="human",
        )

        assert env.num_envs == 8
        assert env.http_url == "http://example.com:8080"
        assert env.player_name == "CustomBot"
        assert env.room_name_prefix == "CustomRoom"
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
        env = VectorizedTowerfallEnv(num_envs=2, observation_config=custom_config)

        assert env._obs_config.max_other_players == 1
        assert env._obs_config.max_tracked_arrows == 4
        assert env._obs_config.include_map is False

    def test_unique_session_id(self) -> None:
        """Test that each env instance gets a unique session ID."""
        env1 = VectorizedTowerfallEnv(num_envs=2)
        env2 = VectorizedTowerfallEnv(num_envs=2)

        assert env1._session_id != env2._session_id
        assert len(env1._session_id) == 8
        assert len(env2._session_id) == 8

    def test_creates_per_env_components(self) -> None:
        """Test that per-environment components are created."""
        num_envs = 4
        env = VectorizedTowerfallEnv(num_envs=num_envs)

        assert len(env._obs_builders) == num_envs
        assert len(env._reward_fns) == num_envs
        assert len(env._termination_trackers) == num_envs
        assert len(env._clients) == num_envs
        assert len(env._episode_steps) == num_envs


class TestVectorizedEnvSpaces:
    """Tests for action and observation spaces."""

    def test_single_action_space_is_discrete_27(self) -> None:
        """Test single action space is Discrete(27)."""
        env = VectorizedTowerfallEnv(num_envs=4)

        assert isinstance(env.single_action_space, gym.spaces.Discrete)
        assert env.single_action_space.n == ACTION_SPACE_SIZE
        assert env.single_action_space.n == 27

    def test_single_observation_space_is_box(self) -> None:
        """Test single observation space is Box with correct bounds."""
        env = VectorizedTowerfallEnv(num_envs=4)

        assert isinstance(env.single_observation_space, gym.spaces.Box)
        assert env.single_observation_space.low.min() == -1.0
        assert env.single_observation_space.high.max() == 1.0
        assert env.single_observation_space.dtype == np.float32

    def test_observation_space_shape_matches_config(self) -> None:
        """Test observation space shape matches observation config."""
        config = ObservationConfig()
        env = VectorizedTowerfallEnv(num_envs=4, observation_config=config)

        assert env.single_observation_space.shape == (config.total_size,)

    def test_observation_space_with_custom_config(self) -> None:
        """Test observation space adapts to custom config."""
        custom_config = ObservationConfig(
            max_other_players=1,
            max_tracked_arrows=4,
            include_map=False,
        )
        env = VectorizedTowerfallEnv(num_envs=2, observation_config=custom_config)

        assert env.single_observation_space.shape == (custom_config.total_size,)

    def test_action_space_sample_is_valid(self) -> None:
        """Test sampled actions are valid."""
        env = VectorizedTowerfallEnv(num_envs=4)

        for _ in range(100):
            action = env.single_action_space.sample()
            assert 0 <= action < 27


class TestVectorizedEnvRoomNames:
    """Tests for room name generation."""

    def test_room_name_format(self) -> None:
        """Test room names have correct format."""
        env = VectorizedTowerfallEnv(num_envs=4, room_name_prefix="TestRoom")

        for i in range(4):
            room_name = env._get_room_name(i)
            assert room_name.startswith("TestRoom_")
            assert room_name.endswith(f"_{i}")

    def test_room_names_are_unique(self) -> None:
        """Test each environment gets a unique room name."""
        env = VectorizedTowerfallEnv(num_envs=4)

        room_names = [env._get_room_name(i) for i in range(4)]
        assert len(set(room_names)) == 4

    def test_player_name_format(self) -> None:
        """Test player names have correct format."""
        env = VectorizedTowerfallEnv(num_envs=4, player_name="TestBot")

        for i in range(4):
            player_name = env._get_player_name(i)
            assert player_name == f"TestBot_{i}"


class TestVectorizedEnvReset:
    """Tests for reset() method."""

    def test_reset_returns_correct_tuple(self) -> None:
        """Test reset returns (observations, infos) tuple."""
        num_envs = 4
        env = VectorizedTowerfallEnv(num_envs=num_envs)

        # Mock _reset_single_env to return test data
        async def mock_reset(env_idx: int, map_type: str):
            obs = np.zeros(env.single_observation_space.shape, dtype=np.float32)
            info = {"env_idx": env_idx}
            return obs, info

        with patch.object(env, "_reset_single_env", side_effect=mock_reset):
            observations, infos = env.reset()

            assert isinstance(observations, np.ndarray)
            assert isinstance(infos, dict)
            assert observations.dtype == np.float32

    def test_reset_observation_shape(self) -> None:
        """Test reset returns observations with correct shape."""
        num_envs = 4
        env = VectorizedTowerfallEnv(num_envs=num_envs)
        obs_dim = env.single_observation_space.shape[0]

        async def mock_reset(env_idx: int, map_type: str):
            obs = np.zeros(env.single_observation_space.shape, dtype=np.float32)
            return obs, {"env_idx": env_idx}

        with patch.object(env, "_reset_single_env", side_effect=mock_reset):
            observations, _ = env.reset()

            assert observations.shape == (num_envs, obs_dim)

    def test_reset_infos_contain_env_infos(self) -> None:
        """Test reset infos contain per-environment info."""
        num_envs = 4
        env = VectorizedTowerfallEnv(num_envs=num_envs)

        async def mock_reset(env_idx: int, map_type: str):
            obs = np.zeros(env.single_observation_space.shape, dtype=np.float32)
            return obs, {"env_idx": env_idx, "room_id": f"room-{env_idx}"}

        with patch.object(env, "_reset_single_env", side_effect=mock_reset):
            _, infos = env.reset()

            assert "env_infos" in infos
            assert len(infos["env_infos"]) == num_envs
            for i, info in enumerate(infos["env_infos"]):
                assert info["env_idx"] == i

    def test_reset_respects_map_type_option(self) -> None:
        """Test reset passes map_type option to environments."""
        env = VectorizedTowerfallEnv(num_envs=2, map_type="arena1")
        captured_map_types = []

        async def mock_reset(env_idx: int, map_type: str):
            captured_map_types.append(map_type)
            obs = np.zeros(env.single_observation_space.shape, dtype=np.float32)
            return obs, {"env_idx": env_idx}

        with patch.object(env, "_reset_single_env", side_effect=mock_reset):
            env.reset(options={"map_type": "arena2"})

            assert all(mt == "arena2" for mt in captured_map_types)


class TestVectorizedEnvStep:
    """Tests for step() method."""

    def test_step_returns_correct_tuple(self) -> None:
        """Test step returns correct tuple structure."""
        num_envs = 4
        env = VectorizedTowerfallEnv(num_envs=num_envs)

        async def mock_step(env_idx: int, action: int):
            obs = np.zeros(env.single_observation_space.shape, dtype=np.float32)
            return obs, 0.0, False, False, {"env_idx": env_idx}

        with patch.object(env, "_step_single_env", side_effect=mock_step):
            actions = np.array([0, 1, 2, 3])
            obs, rewards, terminated, truncated, infos = env.step(actions)

            assert isinstance(obs, np.ndarray)
            assert isinstance(rewards, np.ndarray)
            assert isinstance(terminated, np.ndarray)
            assert isinstance(truncated, np.ndarray)
            assert isinstance(infos, dict)

    def test_step_observation_shape(self) -> None:
        """Test step returns observations with correct shape."""
        num_envs = 4
        env = VectorizedTowerfallEnv(num_envs=num_envs)
        obs_dim = env.single_observation_space.shape[0]

        async def mock_step(env_idx: int, action: int):
            obs = np.zeros(env.single_observation_space.shape, dtype=np.float32)
            return obs, 0.0, False, False, {"env_idx": env_idx}

        with patch.object(env, "_step_single_env", side_effect=mock_step):
            actions = np.array([0, 1, 2, 3])
            obs, _, _, _, _ = env.step(actions)

            assert obs.shape == (num_envs, obs_dim)

    def test_step_rewards_shape(self) -> None:
        """Test step returns rewards with correct shape."""
        num_envs = 4
        env = VectorizedTowerfallEnv(num_envs=num_envs)

        async def mock_step(env_idx: int, action: int):
            obs = np.zeros(env.single_observation_space.shape, dtype=np.float32)
            return obs, float(env_idx) * 0.1, False, False, {}

        with patch.object(env, "_step_single_env", side_effect=mock_step):
            actions = np.array([0, 1, 2, 3])
            _, rewards, _, _, _ = env.step(actions)

            assert rewards.shape == (num_envs,)
            assert rewards.dtype == np.float32

    def test_step_terminated_truncated_shape(self) -> None:
        """Test step returns terminated/truncated with correct shape."""
        num_envs = 4
        env = VectorizedTowerfallEnv(num_envs=num_envs)

        async def mock_step(env_idx: int, action: int):
            obs = np.zeros(env.single_observation_space.shape, dtype=np.float32)
            return obs, 0.0, env_idx == 0, env_idx == 1, {}

        with patch.object(env, "_step_single_env", side_effect=mock_step):
            actions = np.array([0, 1, 2, 3])
            _, _, terminated, truncated, _ = env.step(actions)

            assert terminated.shape == (num_envs,)
            assert truncated.shape == (num_envs,)
            assert terminated.dtype == np.bool_
            assert truncated.dtype == np.bool_
            assert terminated[0] is np.True_
            assert truncated[1] is np.True_

    def test_step_validates_action_shape(self) -> None:
        """Test step raises on invalid action shape."""
        env = VectorizedTowerfallEnv(num_envs=4)

        with pytest.raises(ValueError, match="Expected actions with shape"):
            env.step(np.array([0, 1]))  # Wrong shape

    def test_step_infos_contain_env_infos(self) -> None:
        """Test step infos contain per-environment info."""
        num_envs = 4
        env = VectorizedTowerfallEnv(num_envs=num_envs)

        async def mock_step(env_idx: int, action: int):
            obs = np.zeros(env.single_observation_space.shape, dtype=np.float32)
            return obs, 0.0, False, False, {"env_idx": env_idx, "action": action}

        with patch.object(env, "_step_single_env", side_effect=mock_step):
            actions = np.array([0, 1, 2, 3])
            _, _, _, _, infos = env.step(actions)

            assert "env_infos" in infos
            assert len(infos["env_infos"]) == num_envs
            for i, info in enumerate(infos["env_infos"]):
                assert info["env_idx"] == i
                assert info["action"] == i


class TestVectorizedEnvAutoReset:
    """Tests for automatic reset on termination."""

    @pytest.mark.asyncio
    async def test_step_auto_resets_terminated_env(self) -> None:
        """Test that step auto-resets terminated environments."""
        num_envs = 2
        env = VectorizedTowerfallEnv(num_envs=num_envs)

        # Set up mock clients with matching player IDs
        for i in range(num_envs):
            player_id = f"player-{i}"
            mock_client = MagicMock()
            mock_client.player_id = player_id
            mock_client.room_id = f"room-{i}"
            mock_client.room_code = f"ABC{i}"
            env._clients[i] = mock_client

        # Track calls
        reset_calls = []

        async def mock_reset(env_idx: int, map_type: str):
            reset_calls.append(env_idx)
            obs = np.zeros(env.single_observation_space.shape, dtype=np.float32)
            return obs, {"env_idx": env_idx}

        # Mock game states with correct player IDs
        game_state_over = make_game_state(player_id="player-0", is_game_over=True)
        game_state_normal = make_game_state(player_id="player-1", is_game_over=False)

        with patch.object(env, "_reset_single_env", side_effect=mock_reset):
            # Mock step for first env (terminates)
            env._clients[0].get_game_state = AsyncMock(  # type: ignore[union-attr]
                return_value=game_state_over
            )
            env._clients[0].get_stats = AsyncMock(  # type: ignore[union-attr]
                return_value={"player-0": make_player_stats(playerId="player-0")}
            )

            # Mock step for second env (continues)
            env._clients[1].get_game_state = AsyncMock(  # type: ignore[union-attr]
                return_value=game_state_normal
            )
            env._clients[1].get_stats = AsyncMock(  # type: ignore[union-attr]
                return_value={"player-1": make_player_stats(playerId="player-1")}
            )

            # Enable game over signal in termination config
            env._termination_config.use_game_over_signal = True

            # Step with patched execute_action
            with patch("bot.gym.vectorized_env.execute_action", new=AsyncMock()):
                _, _, terminated, _, infos = await env._async_step(np.array([0, 0]))

            # First env should have been auto-reset
            assert 0 in reset_calls
            # Second env should not have been reset
            assert 1 not in reset_calls

    def test_terminal_observation_in_info(self) -> None:
        """Test that terminal observation is stored in info on termination."""
        num_envs = 2
        env = VectorizedTowerfallEnv(num_envs=num_envs)

        terminal_obs = np.ones(env.single_observation_space.shape, dtype=np.float32)
        reset_obs = np.zeros(env.single_observation_space.shape, dtype=np.float32)

        async def mock_step(env_idx: int, action: int):
            # First env terminates
            if env_idx == 0:
                return (
                    reset_obs,  # New obs after auto-reset
                    1.0,
                    True,  # terminated
                    False,
                    {
                        "terminal_observation": terminal_obs,
                        "episode": {"r": 5, "l": 100},
                    },
                )
            else:
                return reset_obs, 0.0, False, False, {}

        with patch.object(env, "_step_single_env", side_effect=mock_step):
            actions = np.array([0, 0])
            _, _, _, _, infos = env.step(actions)

            # Check terminal observation is in info
            assert "_terminal_observation" in infos
            assert infos["_terminal_observation"][0] is not None
            np.testing.assert_array_equal(
                infos["_terminal_observation"][0], terminal_obs
            )


class TestVectorizedEnvRender:
    """Tests for render() method."""

    def test_render_none_mode(self) -> None:
        """Test render returns None when render_mode is None."""
        env = VectorizedTowerfallEnv(num_envs=2, render_mode=None)

        result = env.render()

        assert result is None

    def test_render_human_mode(self) -> None:
        """Test render in human mode returns None (prints to console)."""
        env = VectorizedTowerfallEnv(num_envs=2, render_mode="human")
        env._clients[0] = MagicMock()
        env._clients[0].room_code = "ABC123"
        env._episode_steps[0] = 10

        result = env.render()

        assert result is None

    def test_render_rgb_array_mode(self) -> None:
        """Test render in rgb_array mode returns tuple of numpy arrays."""
        num_envs = 2
        env = VectorizedTowerfallEnv(num_envs=num_envs, render_mode="rgb_array")

        result = env.render()

        assert isinstance(result, tuple)
        assert len(result) == num_envs
        for frame in result:
            assert isinstance(frame, np.ndarray)
            assert frame.shape == (600, 800, 3)
            assert frame.dtype == np.uint8


class TestVectorizedEnvClose:
    """Tests for close() method."""

    def test_close_clears_clients(self) -> None:
        """Test close clears all client references."""
        num_envs = 4
        env = VectorizedTowerfallEnv(num_envs=num_envs)

        # Set up mock clients
        for i in range(num_envs):
            mock_client = MagicMock()
            mock_client.close = AsyncMock()
            env._clients[i] = mock_client

        # Create an event loop so close_extras will actually run
        env._get_loop()

        env.close()

        assert all(client is None for client in env._clients)

    def test_close_handles_no_clients(self) -> None:
        """Test close handles case when clients are None."""
        env = VectorizedTowerfallEnv(num_envs=4)

        # Should not raise
        env.close()


class TestVectorizedEnvIsVectorEnv:
    """Tests that VectorizedTowerfallEnv is a proper VectorEnv."""

    def test_is_vector_env_subclass(self) -> None:
        """Test VectorizedTowerfallEnv is a gymnasium.vector.VectorEnv subclass."""
        assert issubclass(VectorizedTowerfallEnv, gym.vector.VectorEnv)

    def test_has_num_envs(self) -> None:
        """Test environment has num_envs attribute."""
        env = VectorizedTowerfallEnv(num_envs=8)
        assert env.num_envs == 8

    def test_has_single_spaces(self) -> None:
        """Test environment has single_observation_space and single_action_space."""
        env = VectorizedTowerfallEnv(num_envs=4)

        assert hasattr(env, "single_observation_space")
        assert hasattr(env, "single_action_space")
        assert isinstance(env.single_observation_space, gym.spaces.Box)
        assert isinstance(env.single_action_space, gym.spaces.Discrete)


class TestImports:
    """Tests that all imports work correctly."""

    def test_import_from_bot_gym(self) -> None:
        """Test importing VectorizedTowerfallEnv from bot.gym."""
        from bot.gym import VectorizedTowerfallEnv

        assert VectorizedTowerfallEnv is not None

    def test_env_is_vector_env(self) -> None:
        """Test VectorizedTowerfallEnv is a gymnasium.vector.VectorEnv subclass."""
        from bot.gym import VectorizedTowerfallEnv

        assert issubclass(VectorizedTowerfallEnv, gym.vector.VectorEnv)
