"""Integration tests for TowerfallEnv with real backend.

Tests cover:
- Environment reset in solo mode (opponent_type="none")
- Environment reset vs rule-based bot (opponent_type="rule_based")
- Step returns valid observation, reward, done flags
- Execute multiple steps in sequence
- All sampled actions execute successfully
- Observations are normalized to [-1, 1] range
"""

import numpy as np
import pytest
from gymnasium.spaces import Discrete

from bot.actions import ACTION_SPACE_SIZE
from bot.gym import TowerfallEnv
from tests.conftest import requires_server, unique_room_name

# Default map type for integration tests (must match server's available maps)
DEFAULT_MAP_TYPE = "default"


@pytest.mark.integration
class TestTowerfallEnvIntegration:
    """Integration tests for TowerfallEnv with real server."""

    @requires_server
    def test_reset_solo_mode(self, server_url: str) -> None:
        """Environment reset in solo mode (opponent_type='none')."""
        room_name = unique_room_name("EnvSolo")
        env = TowerfallEnv(
            http_url=server_url,
            room_name=room_name,
            map_type=DEFAULT_MAP_TYPE,
            opponent_type="none",
            tick_rate_multiplier=10.0,
            max_episode_steps=100,
        )

        try:
            obs, info = env.reset()

            assert isinstance(obs, np.ndarray)
            assert obs.dtype == np.float32
            assert obs.shape == env.observation_space.shape
            assert isinstance(info, dict)
            assert "room_id" in info
            assert "player_id" in info
        finally:
            env.close()

    @requires_server
    def test_reset_vs_rule_based_bot(self, server_url: str) -> None:
        """Environment reset vs rule-based bot (opponent_type='rule_based')."""
        room_name = unique_room_name("EnvRuleBased")
        env = TowerfallEnv(
            http_url=server_url,
            room_name=room_name,
            map_type=DEFAULT_MAP_TYPE,
            opponent_type="rule_based",
            tick_rate_multiplier=10.0,
            max_episode_steps=100,
        )

        try:
            obs, info = env.reset()

            assert isinstance(obs, np.ndarray)
            assert obs.dtype == np.float32
            assert obs.shape == env.observation_space.shape
        finally:
            env.close()

    @requires_server
    def test_step_returns_valid_tuple(self, server_url: str) -> None:
        """Step returns valid observation, reward, done flags."""
        room_name = unique_room_name("EnvStep")
        env = TowerfallEnv(
            http_url=server_url,
            room_name=room_name,
            map_type=DEFAULT_MAP_TYPE,
            opponent_type="none",
            tick_rate_multiplier=10.0,
            max_episode_steps=100,
        )

        try:
            env.reset()

            # Take a step with action 0 (noop)
            obs, reward, terminated, truncated, info = env.step(0)

            assert isinstance(obs, np.ndarray)
            assert obs.dtype == np.float32
            assert obs.shape == env.observation_space.shape
            assert isinstance(reward, float)
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            assert isinstance(info, dict)
            assert "episode_step" in info
        finally:
            env.close()

    @requires_server
    def test_execute_multiple_steps(self, server_url: str) -> None:
        """Execute multiple steps in sequence."""
        room_name = unique_room_name("EnvMultiStep")
        env = TowerfallEnv(
            http_url=server_url,
            room_name=room_name,
            map_type=DEFAULT_MAP_TYPE,
            opponent_type="none",
            tick_rate_multiplier=10.0,
            max_episode_steps=100,
        )

        try:
            env.reset()

            # Execute 10 steps
            for step in range(10):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)

                assert obs.shape == env.observation_space.shape
                assert info["episode_step"] == step + 1

                if terminated or truncated:
                    break
        finally:
            env.close()

    @requires_server
    def test_all_sampled_actions_execute(self, server_url: str) -> None:
        """All sampled actions execute successfully."""
        room_name = unique_room_name("EnvAllActions")
        env = TowerfallEnv(
            http_url=server_url,
            room_name=room_name,
            map_type=DEFAULT_MAP_TYPE,
            opponent_type="none",
            tick_rate_multiplier=10.0,
            max_episode_steps=100,
        )

        try:
            env.reset()

            # Verify action space is as expected
            assert isinstance(env.action_space, Discrete)

            # Try each action at least once
            for action in range(ACTION_SPACE_SIZE):
                obs, reward, terminated, truncated, info = env.step(action)

                # All actions should complete without error
                assert obs.shape == env.observation_space.shape

                if terminated or truncated:
                    env.reset()
        finally:
            env.close()

    @requires_server
    def test_observations_normalized(self, server_url: str) -> None:
        """Observations are normalized to [-1, 1] range."""
        room_name = unique_room_name("EnvNormalized")
        env = TowerfallEnv(
            http_url=server_url,
            room_name=room_name,
            map_type=DEFAULT_MAP_TYPE,
            opponent_type="none",
            tick_rate_multiplier=10.0,
            max_episode_steps=100,
        )

        try:
            obs, _ = env.reset()

            # Check observation bounds
            assert obs.min() >= -1.0, f"Observation min {obs.min()} < -1.0"
            assert obs.max() <= 1.0, f"Observation max {obs.max()} > 1.0"

            # Take some steps and check bounds
            for _ in range(5):
                action = env.action_space.sample()
                obs, _, terminated, truncated, _ = env.step(action)

                assert obs.min() >= -1.0, f"Observation min {obs.min()} < -1.0"
                assert obs.max() <= 1.0, f"Observation max {obs.max()} > 1.0"

                if terminated or truncated:
                    break
        finally:
            env.close()
