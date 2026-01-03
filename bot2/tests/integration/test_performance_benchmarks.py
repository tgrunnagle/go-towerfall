"""Integration tests for performance benchmarks.

Tests cover:
- Single env steps/second (print results, assert > 5 steps/s)
- Vectorized env (4 envs) steps/second (assert > 10 env-steps/s)
- Reset latency measurement
"""

import time

import numpy as np
import pytest

from bot.gym import TowerfallEnv, VectorizedTowerfallEnv
from tests.conftest import requires_server, unique_room_name


@pytest.mark.integration
class TestPerformanceBenchmarks:
    """Performance benchmark tests."""

    @requires_server
    def test_single_env_steps_per_second(self, server_url: str) -> None:
        """Single env steps/second benchmark."""
        room_name = unique_room_name("BenchSingle")
        env = TowerfallEnv(
            http_url=server_url,
            room_name=room_name,
            opponent_type="none",
            tick_rate_multiplier=10.0,
            max_episode_steps=1000,
        )

        try:
            env.reset()

            num_steps = 100
            start_time = time.perf_counter()

            for _ in range(num_steps):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    env.reset()

            elapsed_time = time.perf_counter() - start_time
            steps_per_second = num_steps / elapsed_time

            print(f"\nSingle env performance: {steps_per_second:.2f} steps/second")
            print(f"  Total steps: {num_steps}")
            print(f"  Elapsed time: {elapsed_time:.2f}s")

            # Assert minimum performance threshold
            assert steps_per_second > 5.0, (
                f"Single env too slow: {steps_per_second:.2f} steps/s (expected > 5)"
            )
        finally:
            env.close()

    @requires_server
    def test_vectorized_env_steps_per_second(self, server_url: str) -> None:
        """Vectorized env (4 envs) steps/second benchmark."""
        room_prefix = unique_room_name("BenchVec")
        num_envs = 4
        env = VectorizedTowerfallEnv(
            num_envs=num_envs,
            http_url=server_url,
            room_name_prefix=room_prefix,
            opponent_type="none",
            tick_rate_multiplier=10.0,
            max_episode_steps=1000,
        )

        try:
            env.reset()

            num_steps = 100
            start_time = time.perf_counter()

            for _ in range(num_steps):
                actions = np.array(
                    [env.single_action_space.sample() for _ in range(num_envs)]
                )
                obs, rewards, terminateds, truncateds, infos = env.step(actions)

            elapsed_time = time.perf_counter() - start_time
            steps_per_second = num_steps / elapsed_time
            env_steps_per_second = (num_steps * num_envs) / elapsed_time

            print(f"\nVectorized env performance ({num_envs} envs):")
            print(f"  {steps_per_second:.2f} batch steps/second")
            print(f"  {env_steps_per_second:.2f} env-steps/second")
            print(f"  Total batch steps: {num_steps}")
            print(f"  Total env steps: {num_steps * num_envs}")
            print(f"  Elapsed time: {elapsed_time:.2f}s")

            # Assert minimum performance threshold
            assert env_steps_per_second > 10.0, (
                f"Vectorized env too slow: {env_steps_per_second:.2f} env-steps/s "
                "(expected > 10)"
            )
        finally:
            env.close()

    @requires_server
    def test_reset_latency(self, server_url: str) -> None:
        """Reset latency measurement."""
        room_name = unique_room_name("BenchReset")
        env = TowerfallEnv(
            http_url=server_url,
            room_name=room_name,
            opponent_type="none",
            tick_rate_multiplier=10.0,
            max_episode_steps=100,
        )

        try:
            # Measure multiple resets
            num_resets = 10
            reset_times: list[float] = []

            for _ in range(num_resets):
                start_time = time.perf_counter()
                env.reset()
                elapsed_time = time.perf_counter() - start_time
                reset_times.append(elapsed_time)

            avg_reset_time = sum(reset_times) / len(reset_times)
            min_reset_time = min(reset_times)
            max_reset_time = max(reset_times)

            print(f"\nReset latency ({num_resets} resets):")
            print(f"  Average: {avg_reset_time * 1000:.2f}ms")
            print(f"  Min: {min_reset_time * 1000:.2f}ms")
            print(f"  Max: {max_reset_time * 1000:.2f}ms")

            # Reset should complete in reasonable time (< 2 seconds)
            assert avg_reset_time < 2.0, (
                f"Reset too slow: {avg_reset_time:.2f}s average (expected < 2s)"
            )
        finally:
            env.close()
