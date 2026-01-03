"""Integration tests for full episode rollouts.

Tests cover:
- Complete episode in solo mode (100 steps max)
- Complete episode vs rule-based bot
- Multiple episodes with reset between them
- Multiple episodes on vectorized env
"""

import numpy as np
import pytest

from bot.gym import TowerfallEnv, VectorizedTowerfallEnv
from tests.conftest import requires_server, unique_room_name


@pytest.mark.integration
class TestFullEpisodeRollout:
    """Integration tests for full episode rollouts."""

    @requires_server
    def test_complete_episode_solo_mode(self, server_url: str) -> None:
        """Complete episode in solo mode (100 steps max)."""
        room_name = unique_room_name("EpisodeSolo")
        env = TowerfallEnv(
            http_url=server_url,
            room_name=room_name,
            opponent_type="none",
            tick_rate_multiplier=10.0,
            max_episode_steps=100,
        )

        try:
            obs, _ = env.reset()
            total_reward = 0.0
            steps = 0

            while True:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                steps += 1

                if terminated or truncated:
                    break

            # Should complete within max_episode_steps
            assert steps <= 100
            assert "episode_step" in info
            # Total reward should be a valid number
            assert not np.isnan(total_reward)
        finally:
            env.close()

    @requires_server
    def test_complete_episode_vs_rule_based(self, server_url: str) -> None:
        """Complete episode vs rule-based bot."""
        room_name = unique_room_name("EpisodeRuleBased")
        env = TowerfallEnv(
            http_url=server_url,
            room_name=room_name,
            opponent_type="rule_based",
            tick_rate_multiplier=10.0,
            max_episode_steps=100,
        )

        try:
            obs, _ = env.reset()
            total_reward = 0.0
            steps = 0

            while True:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                steps += 1

                if terminated or truncated:
                    break

            assert steps <= 100
            assert not np.isnan(total_reward)
        finally:
            env.close()

    @requires_server
    def test_multiple_episodes_with_reset(self, server_url: str) -> None:
        """Multiple episodes with reset between them."""
        room_name = unique_room_name("MultiEpisode")
        env = TowerfallEnv(
            http_url=server_url,
            room_name=room_name,
            opponent_type="none",
            tick_rate_multiplier=10.0,
            max_episode_steps=50,
        )

        try:
            num_episodes = 3

            for episode in range(num_episodes):
                obs, _ = env.reset()
                episode_reward = 0.0
                steps = 0

                while True:
                    action = env.action_space.sample()
                    obs, reward, terminated, truncated, info = env.step(action)
                    episode_reward += reward
                    steps += 1

                    if terminated or truncated:
                        break

                # Each episode should complete
                assert steps <= 50
                assert not np.isnan(episode_reward)
        finally:
            env.close()

    @requires_server
    def test_multiple_episodes_vectorized(self, server_url: str) -> None:
        """Multiple episodes on vectorized env."""
        room_prefix = unique_room_name("VecMultiEpisode")
        num_envs = 2
        env = VectorizedTowerfallEnv(
            num_envs=num_envs,
            http_url=server_url,
            room_name_prefix=room_prefix,
            opponent_type="none",
            tick_rate_multiplier=10.0,
            max_episode_steps=50,
        )

        try:
            obs, _ = env.reset()
            total_steps = 0
            episode_counts = [0] * num_envs

            # Run enough steps to see multiple episodes complete
            for _ in range(150):  # 3 episodes worth of steps
                actions = np.array(
                    [env.single_action_space.sample() for _ in range(num_envs)]
                )
                obs, rewards, terminateds, truncateds, infos = env.step(actions)
                total_steps += 1

                # Count completed episodes
                for i in range(num_envs):
                    if terminateds[i] or truncateds[i]:
                        episode_counts[i] += 1

            # Each environment should have completed at least 2 episodes
            for i in range(num_envs):
                assert episode_counts[i] >= 2, (
                    f"Env {i} only completed {episode_counts[i]} episodes"
                )
        finally:
            env.close()
