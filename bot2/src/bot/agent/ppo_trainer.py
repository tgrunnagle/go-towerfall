"""PPO training loop implementation.

This module implements the Proximal Policy Optimization (PPO) algorithm
for training actor-critic neural networks on reinforcement learning tasks.
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from bot.agent.network import ActorCriticNetwork
from bot.agent.rollout_buffer import RolloutBuffer


class VectorizedEnvironment(Protocol):
    """Protocol for vectorized gym-like environment.

    This protocol defines the interface expected by PPOTrainer for
    collecting rollouts from parallel environments.
    """

    num_envs: int

    def reset(self) -> tuple[np.ndarray, dict]:
        """Reset all environments.

        Returns:
            Tuple of (observations, info_dict) where observations has
            shape (num_envs, obs_dim)
        """
        ...

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        """Step all environments.

        Args:
            actions: Array of actions with shape (num_envs,)

        Returns:
            Tuple of (observations, rewards, terminated, truncated, info)
            where each array has shape (num_envs,) or (num_envs, obs_dim)
        """
        ...


@dataclass
class PPOConfig:
    """PPO hyperparameters.

    Attributes:
        num_steps: Steps per environment per rollout
        gamma: Discount factor for rewards
        gae_lambda: GAE lambda for advantage estimation
        num_epochs: Training epochs per update
        minibatch_size: Samples per minibatch
        clip_range: PPO clipping epsilon for policy
        clip_range_vf: Value function clip range (None = no clipping)
        value_coef: Coefficient for value loss
        entropy_coef: Coefficient for entropy bonus
        learning_rate: Optimizer learning rate
        max_grad_norm: Maximum gradient norm for clipping
        normalize_advantages: Whether to normalize advantages per minibatch
    """

    # Rollout
    num_steps: int = 2048

    # GAE
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # PPO
    num_epochs: int = 10
    minibatch_size: int = 64
    clip_range: float = 0.2
    clip_range_vf: float | None = None

    # Loss coefficients
    value_coef: float = 0.5
    entropy_coef: float = 0.01

    # Optimization
    learning_rate: float = 3e-4
    max_grad_norm: float = 0.5

    # Normalization
    normalize_advantages: bool = True


class PPOTrainer:
    """PPO training loop implementation.

    Orchestrates the PPO training process including:
    - Rollout collection from vectorized environments
    - Advantage estimation using GAE
    - Policy and value network updates with clipped objectives

    Attributes:
        config: PPO hyperparameters
        device: Torch device for computation
        network: Actor-critic neural network
        optimizer: Adam optimizer for network parameters
        total_timesteps: Cumulative timesteps collected
        num_updates: Number of update steps performed
    """

    def __init__(
        self,
        network: ActorCriticNetwork,
        config: PPOConfig | None = None,
        device: torch.device | None = None,
    ):
        """Initialize the PPO trainer.

        Args:
            network: Actor-critic network to train
            config: PPO hyperparameters (uses defaults if None)
            device: Torch device (auto-detects if None)
        """
        self.config = config or PPOConfig()
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.network = network.to(self.device)
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=self.config.learning_rate,
        )

        # Training statistics
        self.total_timesteps = 0
        self.num_updates = 0

    def collect_rollout(
        self,
        env: VectorizedEnvironment,
        obs: torch.Tensor,
    ) -> tuple[RolloutBuffer, torch.Tensor]:
        """Collect rollout experience from vectorized environment.

        Runs the current policy in the environment for num_steps timesteps,
        storing observations, actions, rewards, and values in a buffer.

        Args:
            env: Vectorized gym-like environment
            obs: Current observation tensor of shape (num_envs, obs_size)

        Returns:
            Tuple of (filled rollout buffer, next observation tensor)
        """
        buffer = RolloutBuffer.create(
            self.config.num_steps,
            env.num_envs,
            self.network.observation_size,
            self.device,
        )

        for step in range(self.config.num_steps):
            obs_tensor = obs.to(self.device)

            with torch.no_grad():
                action, log_prob, _, value = self.network.get_action_and_value(
                    obs_tensor
                )

            # Store experience
            buffer.observations[step] = obs_tensor
            buffer.actions[step] = action
            buffer.log_probs[step] = log_prob
            buffer.values[step] = value

            # Take action in environment
            next_obs, reward, terminated, truncated, _info = env.step(
                action.cpu().numpy()
            )
            done = np.logical_or(terminated, truncated)

            buffer.rewards[step] = torch.as_tensor(
                reward, dtype=torch.float32, device=self.device
            )
            buffer.dones[step] = torch.as_tensor(
                done, dtype=torch.float32, device=self.device
            )

            # Environment handles auto-reset, so next_obs is always valid
            obs = torch.as_tensor(next_obs, dtype=torch.float32)

        # Compute advantages with bootstrapped value from final state
        with torch.no_grad():
            last_value = self.network.get_value(obs.to(self.device))

        # Use dones from last step for bootstrapping
        last_done = buffer.dones[-1]

        buffer.compute_advantages(
            last_value,
            last_done,
            self.config.gamma,
            self.config.gae_lambda,
        )

        self.total_timesteps += self.config.num_steps * env.num_envs
        return buffer, obs

    def update(self, buffer: RolloutBuffer) -> dict[str, float]:
        """Perform PPO update on collected rollout.

        Runs multiple epochs of minibatch gradient descent using the
        PPO clipped surrogate objective.

        Args:
            buffer: Rollout buffer with computed advantages

        Returns:
            Dictionary of training metrics including:
            - policy_loss: Mean policy loss
            - value_loss: Mean value loss
            - entropy: Mean policy entropy
            - clip_fraction: Fraction of samples clipped
            - approx_kl: Approximate KL divergence
            - total_timesteps: Cumulative timesteps
            - num_updates: Total update count
        """
        # Track metrics across all epochs and minibatches
        policy_losses: list[float] = []
        value_losses: list[float] = []
        entropy_losses: list[float] = []
        clip_fractions: list[float] = []
        approx_kls: list[float] = []

        for _epoch in range(self.config.num_epochs):
            for batch in buffer.get_minibatches(self.config.minibatch_size):
                # Get current policy outputs for the batch
                _, new_log_prob, entropy, new_value = self.network.get_action_and_value(
                    batch["observations"],
                    batch["actions"],
                )

                # Optionally normalize advantages per minibatch
                advantages = batch["advantages"]
                if self.config.normalize_advantages and advantages.numel() > 1:
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-8
                    )

                # Compute policy loss with PPO clipping
                log_ratio = new_log_prob - batch["log_probs"]
                ratio = torch.exp(log_ratio)

                # Compute metrics for monitoring
                with torch.no_grad():
                    # Approximate KL divergence: E[(ratio - 1) - log(ratio)]
                    approx_kl = ((ratio - 1) - log_ratio).mean()
                    # Fraction of samples where clipping was applied
                    clip_fraction = (
                        ((ratio - 1.0).abs() > self.config.clip_range).float().mean()
                    )

                # PPO clipped surrogate objective
                policy_loss_1 = -advantages * ratio
                policy_loss_2 = -advantages * torch.clamp(
                    ratio,
                    1.0 - self.config.clip_range,
                    1.0 + self.config.clip_range,
                )
                policy_loss = torch.max(policy_loss_1, policy_loss_2).mean()

                # Value loss (optionally clipped)
                if self.config.clip_range_vf is not None:
                    value_clipped = batch["values"] + torch.clamp(
                        new_value - batch["values"],
                        -self.config.clip_range_vf,
                        self.config.clip_range_vf,
                    )
                    value_loss_1 = (new_value - batch["returns"]) ** 2
                    value_loss_2 = (value_clipped - batch["returns"]) ** 2
                    value_loss = 0.5 * torch.max(value_loss_1, value_loss_2).mean()
                else:
                    value_loss = 0.5 * ((new_value - batch["returns"]) ** 2).mean()

                # Entropy bonus (negative because we maximize entropy)
                entropy_loss = -entropy.mean()

                # Combined loss
                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    + self.config.entropy_coef * entropy_loss
                )

                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.network.parameters(),
                    self.config.max_grad_norm,
                )
                self.optimizer.step()

                # Record metrics
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(-entropy_loss.item())  # Convert back to positive
                clip_fractions.append(clip_fraction.item())
                approx_kls.append(approx_kl.item())

        self.num_updates += 1

        return {
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
            "entropy": float(np.mean(entropy_losses)),
            "clip_fraction": float(np.mean(clip_fractions)),
            "approx_kl": float(np.mean(approx_kls)),
            "total_timesteps": self.total_timesteps,
            "num_updates": self.num_updates,
        }

    def train_step(
        self,
        env: VectorizedEnvironment,
        obs: torch.Tensor,
    ) -> tuple[dict[str, float], torch.Tensor]:
        """Perform one full training step (rollout + update).

        This is the main entry point for training iteration, combining
        rollout collection and policy update.

        Args:
            env: Vectorized gym-like environment
            obs: Current observation tensor

        Returns:
            Tuple of (metrics dictionary, next observation tensor)
        """
        buffer, next_obs = self.collect_rollout(env, obs)
        metrics = self.update(buffer)
        return metrics, next_obs

    def save(self, path: str) -> None:
        """Save trainer state to file.

        Saves the network weights, optimizer state, and training statistics.

        Args:
            path: Path to save checkpoint file
        """
        torch.save(
            {
                "network_state_dict": self.network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "total_timesteps": self.total_timesteps,
                "num_updates": self.num_updates,
                "config": self.config,
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load trainer state from file.

        Restores network weights, optimizer state, and training statistics.

        Args:
            path: Path to checkpoint file
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.total_timesteps = checkpoint["total_timesteps"]
        self.num_updates = checkpoint["num_updates"]
        # Note: config is not restored to allow changing hyperparameters
