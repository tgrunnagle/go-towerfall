"""Agent module for PPO reinforcement learning.

This module contains the neural network architecture and utilities
for training PPO (Proximal Policy Optimization) agents.
"""

from bot.agent.network import ActorCriticNetwork

__all__ = ["ActorCriticNetwork"]
