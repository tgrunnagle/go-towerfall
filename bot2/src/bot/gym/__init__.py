"""Environment module for TowerFall RL training.

This module provides a gymnasium environment for training RL agents
to play TowerFall through the go-towerfall game server.

Usage:
    from bot.gym import TowerfallEnv, RewardConfig

    # Direct instantiation with default rewards
    env = TowerfallEnv(
        http_url="http://localhost:4000",
        player_name="TrainingBot",
        map_type="arena1",
        tick_rate_multiplier=10.0,
    )

    # Custom reward configuration
    reward_config = RewardConfig(
        kill_reward=2.0,
        death_penalty=-1.5,
        timestep_penalty=-0.002,
    )
    env = TowerfallEnv(reward_config=reward_config)

    # Or via gymnasium registry
    import gymnasium as gym
    env = gym.make("Towerfall-v0")

    # Standard RL loop
    observation, info = env.reset()
    for _ in range(1000):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            observation, info = env.reset()
    env.close()
"""

from bot.gym.reward import RewardConfig, RewardFunction, StandardRewardFunction
from bot.gym.termination import TerminationConfig, TerminationTracker
from bot.gym.towerfall_env import TowerfallEnv

__all__ = [
    "TowerfallEnv",
    "RewardConfig",
    "RewardFunction",
    "StandardRewardFunction",
    "TerminationConfig",
    "TerminationTracker",
]

# Register with gymnasium
from gymnasium.envs.registration import register

register(
    id="Towerfall-v0",
    entry_point="bot.gym:TowerfallEnv",
    max_episode_steps=1000,
)
