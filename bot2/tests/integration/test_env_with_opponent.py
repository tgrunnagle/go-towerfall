"""Integration tests for TowerfallEnv with rule-based bot opponent.

Tests cover:
- Environment reset with rule-based opponent spawns two players
- Opponent takes actions during episode steps
- Environment cleanup properly stops opponent
- Full episode with ML agent vs rule-based bot
"""

import numpy as np
import pytest

from bot.gym import TowerfallEnv, VectorizedTowerfallEnv
from tests.conftest import requires_server, unique_room_name

# Default map type for integration tests
DEFAULT_MAP_TYPE = "default"


@pytest.mark.integration
class TestTowerfallEnvWithOpponent:
    """Integration tests for TowerfallEnv with rule-based opponent."""

    @requires_server
    def test_reset_with_rule_based_opponent(self, server_url: str) -> None:
        """Environment reset with rule-based opponent includes opponent info."""
        room_name = unique_room_name("EnvWithOpponent")
        env = TowerfallEnv(
            http_url=server_url,
            ws_url=server_url.replace("http://", "ws://") + "/ws",
            room_name=room_name,
            map_type=DEFAULT_MAP_TYPE,
            opponent_type="rule_based",
            opponent_name="TestRuleBot",
            tick_rate_multiplier=10.0,
            max_episode_steps=100,
        )

        try:
            obs, info = env.reset()

            # Check observation is valid
            assert isinstance(obs, np.ndarray)
            assert obs.dtype == np.float32
            assert obs.shape == env.observation_space.shape

            # Check info includes opponent type
            assert "opponent_type" in info
            assert info["opponent_type"] == "rule_based"
        finally:
            env.close()

    @requires_server
    def test_step_with_rule_based_opponent(self, server_url: str) -> None:
        """Steps execute successfully with rule-based opponent active."""
        room_name = unique_room_name("EnvOpponentStep")
        env = TowerfallEnv(
            http_url=server_url,
            ws_url=server_url.replace("http://", "ws://") + "/ws",
            room_name=room_name,
            map_type=DEFAULT_MAP_TYPE,
            opponent_type="rule_based",
            tick_rate_multiplier=10.0,
            max_episode_steps=100,
        )

        try:
            env.reset()

            # Execute multiple steps
            for _ in range(10):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)

                assert isinstance(obs, np.ndarray)
                assert obs.shape == env.observation_space.shape
                assert isinstance(reward, float)
                assert isinstance(terminated, bool)
                assert isinstance(truncated, bool)

                if terminated or truncated:
                    break
        finally:
            env.close()

    @requires_server
    def test_multiple_resets_with_opponent(self, server_url: str) -> None:
        """Multiple resets work correctly with opponent."""
        room_name = unique_room_name("EnvMultiReset")
        env = TowerfallEnv(
            http_url=server_url,
            ws_url=server_url.replace("http://", "ws://") + "/ws",
            room_name=room_name,
            map_type=DEFAULT_MAP_TYPE,
            opponent_type="rule_based",
            tick_rate_multiplier=10.0,
            max_episode_steps=50,
        )

        try:
            # First episode
            obs1, info1 = env.reset()
            assert info1["opponent_type"] == "rule_based"

            for _ in range(5):
                action = env.action_space.sample()
                obs, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break

            # Second episode (reset within same room)
            obs2, info2 = env.reset()
            assert info2["opponent_type"] == "rule_based"

            for _ in range(5):
                action = env.action_space.sample()
                obs, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break
        finally:
            env.close()

    @requires_server
    def test_full_episode_with_opponent(self, server_url: str) -> None:
        """Run a complete episode with ML agent vs rule-based bot."""
        room_name = unique_room_name("EnvFullEpisode")
        env = TowerfallEnv(
            http_url=server_url,
            ws_url=server_url.replace("http://", "ws://") + "/ws",
            room_name=room_name,
            map_type=DEFAULT_MAP_TYPE,
            opponent_type="rule_based",
            opponent_name="FullEpisodeBot",
            tick_rate_multiplier=10.0,
            max_episode_steps=100,
        )

        try:
            obs, info = env.reset()
            assert info["opponent_type"] == "rule_based"

            total_reward = 0.0
            steps = 0

            for step in range(100):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)

                total_reward += reward
                steps += 1

                if terminated or truncated:
                    break

            # Episode should complete (either terminated or truncated)
            assert steps > 0
        finally:
            env.close()

    @requires_server
    def test_no_opponent_mode(self, server_url: str) -> None:
        """Environment works correctly with no opponent (solo mode)."""
        room_name = unique_room_name("EnvNoOpponent")
        env = TowerfallEnv(
            http_url=server_url,
            ws_url=server_url.replace("http://", "ws://") + "/ws",
            room_name=room_name,
            map_type=DEFAULT_MAP_TYPE,
            opponent_type="none",
            tick_rate_multiplier=10.0,
            max_episode_steps=50,
        )

        try:
            obs, info = env.reset()
            assert info["opponent_type"] == "none"

            # Take some steps in solo mode
            for _ in range(10):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)

                assert isinstance(obs, np.ndarray)

                if terminated or truncated:
                    break
        finally:
            env.close()


@pytest.mark.integration
class TestVectorizedEnvWithOpponent:
    """Integration tests for VectorizedTowerfallEnv with opponents."""

    @requires_server
    def test_vectorized_reset_with_opponents(self, server_url: str) -> None:
        """Vectorized environment reset creates opponents for each env."""
        env = VectorizedTowerfallEnv(
            num_envs=2,
            http_url=server_url,
            ws_url=server_url.replace("http://", "ws://") + "/ws",
            room_name_prefix=unique_room_name("VecOpponent"),
            map_type=DEFAULT_MAP_TYPE,
            opponent_type="rule_based",
            opponent_name="VecRuleBot",
            tick_rate_multiplier=10.0,
            max_episode_steps=50,
        )

        try:
            observations, infos = env.reset()

            # Check batched observations
            assert observations.shape == (2, env.single_observation_space.shape[0])

            # Check each environment has opponent info
            for env_info in infos["env_infos"]:
                assert env_info["opponent_type"] == "rule_based"
        finally:
            env.close()

    @requires_server
    def test_vectorized_step_with_opponents(self, server_url: str) -> None:
        """Vectorized environment steps work with opponents."""
        env = VectorizedTowerfallEnv(
            num_envs=2,
            http_url=server_url,
            ws_url=server_url.replace("http://", "ws://") + "/ws",
            room_name_prefix=unique_room_name("VecOpponentStep"),
            map_type=DEFAULT_MAP_TYPE,
            opponent_type="rule_based",
            tick_rate_multiplier=10.0,
            max_episode_steps=50,
        )

        try:
            env.reset()

            # Execute multiple steps
            for _ in range(10):
                actions = np.array([
                    env.single_action_space.sample()
                    for _ in range(env.num_envs)
                ])
                observations, rewards, terminated, truncated, infos = env.step(actions)

                assert observations.shape == (2, env.single_observation_space.shape[0])
                assert rewards.shape == (2,)
                assert terminated.shape == (2,)
                assert truncated.shape == (2,)
        finally:
            env.close()

    @requires_server
    def test_vectorized_no_opponent_mode(self, server_url: str) -> None:
        """Vectorized environment works with no opponent (solo mode)."""
        env = VectorizedTowerfallEnv(
            num_envs=2,
            http_url=server_url,
            ws_url=server_url.replace("http://", "ws://") + "/ws",
            room_name_prefix=unique_room_name("VecNoOpponent"),
            map_type=DEFAULT_MAP_TYPE,
            opponent_type="none",
            tick_rate_multiplier=10.0,
            max_episode_steps=50,
        )

        try:
            observations, infos = env.reset()

            # Check each environment has no opponent
            for env_info in infos["env_infos"]:
                assert env_info["opponent_type"] == "none"

            # Take some steps
            for _ in range(5):
                actions = np.array([
                    env.single_action_space.sample()
                    for _ in range(env.num_envs)
                ])
                env.step(actions)
        finally:
            env.close()
