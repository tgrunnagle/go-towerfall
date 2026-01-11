"""Rollout buffer for storing PPO experience data.

This module provides a buffer for storing trajectory data during rollout
collection, with support for Generalized Advantage Estimation (GAE).
"""

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class RolloutBuffer:
    """Buffer for storing rollout experience.

    Stores experience tuples during rollout collection for PPO training.
    Pre-allocates tensors for efficient storage and provides methods for
    computing advantages using GAE and generating minibatches.

    Attributes:
        observations: Observation tensors of shape (num_steps, num_envs, obs_size)
        actions: Action tensors of shape (num_steps, num_envs)
        log_probs: Log probabilities of shape (num_steps, num_envs)
        rewards: Reward tensors of shape (num_steps, num_envs)
        values: Value estimates of shape (num_steps, num_envs)
        dones: Done flags of shape (num_steps, num_envs)
        advantages: Computed GAE advantages (set by compute_advantages)
        returns: Computed returns (set by compute_advantages)
    """

    observations: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    rewards: torch.Tensor
    values: torch.Tensor
    dones: torch.Tensor
    advantages: torch.Tensor | None = None
    returns: torch.Tensor | None = None

    @classmethod
    def create(
        cls,
        num_steps: int,
        num_envs: int,
        observation_size: int,
        device: torch.device,
    ) -> "RolloutBuffer":
        """Create an empty buffer with pre-allocated tensors.

        Args:
            num_steps: Number of steps per rollout
            num_envs: Number of parallel environments
            observation_size: Dimension of observation vectors
            device: Device to allocate tensors on

        Returns:
            RolloutBuffer with zeroed tensors
        """
        return cls(
            observations=torch.zeros(
                num_steps, num_envs, observation_size, device=device
            ),
            actions=torch.zeros(num_steps, num_envs, dtype=torch.long, device=device),
            log_probs=torch.zeros(num_steps, num_envs, device=device),
            rewards=torch.zeros(num_steps, num_envs, device=device),
            values=torch.zeros(num_steps, num_envs, device=device),
            dones=torch.zeros(num_steps, num_envs, device=device),
        )

    def compute_advantages(
        self,
        last_value: torch.Tensor,
        last_done: torch.Tensor,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        """Compute GAE advantages and returns.

        Implements Generalized Advantage Estimation to compute advantages
        that balance bias and variance in policy gradient estimation.

        GAE formula: A_t = sum_{l=0}^{inf} (gamma * lambda)^l * delta_{t+l}
        where delta_t = r_t + gamma * V(s_{t+1}) * (1 - done) - V(s_t)

        Args:
            last_value: Bootstrap value for final state, shape (num_envs,)
            last_done: Whether final state was terminal, shape (num_envs,)
            gamma: Discount factor for future rewards (default: 0.99)
            gae_lambda: GAE lambda parameter for bias-variance tradeoff (default: 0.95)
        """
        num_steps = self.rewards.shape[0]
        num_envs = self.rewards.shape[1]
        device = self.rewards.device

        advantages = torch.zeros(num_steps, num_envs, device=device)
        last_gae = torch.zeros(num_envs, device=device)

        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                next_non_terminal = 1.0 - last_done.float()
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_value = self.values[t + 1]

            delta = (
                self.rewards[t]
                + gamma * next_value * next_non_terminal
                - self.values[t]
            )
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        self.advantages = advantages
        self.returns = advantages + self.values

    def flatten(self) -> dict[str, torch.Tensor]:
        """Flatten the buffer for minibatch sampling.

        Reshapes all tensors from (num_steps, num_envs, ...) to
        (num_steps * num_envs, ...) for batch processing.

        Returns:
            Dictionary of flattened tensors

        Raises:
            RuntimeError: If advantages have not been computed
        """
        if self.advantages is None or self.returns is None:
            raise RuntimeError(
                "Advantages must be computed before flattening. "
                "Call compute_advantages() first."
            )

        batch_size = self.observations.shape[0] * self.observations.shape[1]
        obs_size = self.observations.shape[2]

        return {
            "observations": self.observations.reshape(batch_size, obs_size),
            "actions": self.actions.reshape(batch_size),
            "log_probs": self.log_probs.reshape(batch_size),
            "advantages": self.advantages.reshape(batch_size),
            "returns": self.returns.reshape(batch_size),
            "values": self.values.reshape(batch_size),
        }

    def get_minibatches(
        self,
        minibatch_size: int,
        shuffle: bool = True,
    ) -> list[dict[str, torch.Tensor]]:
        """Split buffer into minibatches for training.

        Args:
            minibatch_size: Number of samples per minibatch
            shuffle: Whether to shuffle indices before splitting

        Returns:
            List of dictionaries containing minibatch tensors

        Raises:
            RuntimeError: If advantages have not been computed
        """
        flat = self.flatten()
        batch_size = flat["observations"].shape[0]

        indices = np.arange(batch_size)
        if shuffle:
            np.random.shuffle(indices)

        batches = []
        for start in range(0, batch_size, minibatch_size):
            end = min(start + minibatch_size, batch_size)
            batch_indices = indices[start:end]

            # Convert numpy indices to tensor for indexing
            idx = torch.as_tensor(batch_indices, dtype=torch.long, device=flat["observations"].device)

            batches.append({
                "observations": flat["observations"][idx],
                "actions": flat["actions"][idx],
                "log_probs": flat["log_probs"][idx],
                "advantages": flat["advantages"][idx],
                "returns": flat["returns"][idx],
                "values": flat["values"][idx],
            })

        return batches

    @property
    def total_steps(self) -> int:
        """Total number of steps in buffer (num_steps * num_envs)."""
        return self.observations.shape[0] * self.observations.shape[1]
