"""Integration tests for VectorizedTowerfallEnv with real backend.

Tests cover:
- 2 parallel environments reset/step correctly
- 4 parallel environments reset/step correctly
- Stepping all environments in parallel
- Auto-reset when individual environments terminate
- Sending different actions to each environment
"""

import numpy as np
import pytest

from bot.gym import VectorizedTowerfallEnv
from tests.conftest import requires_server, unique_room_name

# Default map type for integration tests (must match server's available maps)
DEFAULT_MAP_TYPE = "default"


@pytest.mark.integration
class TestVectorizedEnvIntegration:
    """Integration tests for VectorizedTowerfallEnv with real server."""

    @requires_server
    def test_two_parallel_envs_reset_step(self, server_url: str) -> None:
        """2 parallel environments reset/step correctly."""
        room_prefix = unique_room_name("Vec2Envs")
        env = VectorizedTowerfallEnv(
            num_envs=2,
            http_url=server_url,
            room_name_prefix=room_prefix,
            map_type=DEFAULT_MAP_TYPE,
            opponent_type="none",
            tick_rate_multiplier=10.0,
            max_episode_steps=100,
        )

        try:
            obs, infos = env.reset()

            assert isinstance(obs, np.ndarray)
            assert obs.shape[0] == 2  # 2 environments
            assert obs.dtype == np.float32

            # Take a step
            actions = np.array([0, 1])  # Different actions for each env
            obs, rewards, terminateds, truncateds, infos = env.step(actions)

            assert obs.shape[0] == 2
            assert len(rewards) == 2
            assert len(terminateds) == 2
            assert len(truncateds) == 2
        finally:
            env.close()

    @requires_server
    def test_four_parallel_envs_reset_step(self, server_url: str) -> None:
        """4 parallel environments reset/step correctly."""
        room_prefix = unique_room_name("Vec4Envs")
        env = VectorizedTowerfallEnv(
            num_envs=4,
            http_url=server_url,
            room_name_prefix=room_prefix,
            map_type=DEFAULT_MAP_TYPE,
            opponent_type="none",
            tick_rate_multiplier=10.0,
            max_episode_steps=100,
        )

        try:
            obs, infos = env.reset()

            assert isinstance(obs, np.ndarray)
            assert obs.shape[0] == 4  # 4 environments
            assert obs.dtype == np.float32

            # Take a step
            actions = np.array([0, 1, 2, 3])
            obs, rewards, terminateds, truncateds, infos = env.step(actions)

            assert obs.shape[0] == 4
            assert len(rewards) == 4
            assert len(terminateds) == 4
            assert len(truncateds) == 4
        finally:
            env.close()

    @requires_server
    def test_stepping_all_envs_parallel(self, server_url: str) -> None:
        """Stepping all environments in parallel."""
        room_prefix = unique_room_name("VecParallel")
        env = VectorizedTowerfallEnv(
            num_envs=2,
            http_url=server_url,
            room_name_prefix=room_prefix,
            map_type=DEFAULT_MAP_TYPE,
            opponent_type="none",
            tick_rate_multiplier=10.0,
            max_episode_steps=100,
        )

        try:
            env.reset()

            # Take multiple parallel steps
            for step in range(10):
                actions = np.array([env.single_action_space.sample() for _ in range(2)])
                obs, rewards, terminateds, truncateds, infos = env.step(actions)

                assert obs.shape[0] == 2
                # All observations should be valid
                for i in range(2):
                    assert not np.any(np.isnan(obs[i]))
        finally:
            env.close()

    @requires_server
    def test_auto_reset_on_termination(self, server_url: str) -> None:
        """Auto-reset when individual environments terminate."""
        room_prefix = unique_room_name("VecAutoReset")
        env = VectorizedTowerfallEnv(
            num_envs=2,
            http_url=server_url,
            room_name_prefix=room_prefix,
            map_type=DEFAULT_MAP_TYPE,
            opponent_type="none",
            tick_rate_multiplier=10.0,
            max_episode_steps=50,  # Short episodes to trigger truncation
        )

        try:
            env.reset()

            # Run until at least one environment truncates
            for _ in range(60):  # More than max_episode_steps
                actions = np.array([env.single_action_space.sample() for _ in range(2)])
                obs, rewards, terminateds, truncateds, infos = env.step(actions)

                # Even after truncation, we should get valid observations
                # due to auto-reset
                assert obs.shape[0] == 2
                for i in range(2):
                    assert not np.any(np.isnan(obs[i]))
        finally:
            env.close()

    @requires_server
    def test_different_actions_each_env(self, server_url: str) -> None:
        """Sending different actions to each environment."""
        room_prefix = unique_room_name("VecDiffActions")
        env = VectorizedTowerfallEnv(
            num_envs=4,
            http_url=server_url,
            room_name_prefix=room_prefix,
            map_type=DEFAULT_MAP_TYPE,
            opponent_type="none",
            tick_rate_multiplier=10.0,
            max_episode_steps=100,
        )

        try:
            env.reset()

            # Send unique actions to each environment
            # 0=noop, 1=left, 2=right, 3=up
            actions = np.array([0, 1, 2, 3])

            for _ in range(5):
                obs, rewards, terminateds, truncateds, infos = env.step(actions)

                # Each environment should process independently
                assert obs.shape[0] == 4
                assert len(rewards) == 4

                # Rotate actions for variety
                actions = np.roll(actions, 1)
        finally:
            env.close()
