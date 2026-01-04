"""PPO Actor-Critic neural network architecture.

This module implements the neural network for Proximal Policy Optimization (PPO)
with a shared feature extractor and separate actor/critic heads.
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical

from bot.actions import ACTION_SPACE_SIZE
from bot.observation.observation_space import DEFAULT_CONFIG


class ActorCriticNetwork(nn.Module):
    """PPO Actor-Critic neural network with shared feature extractor.

    The network consists of:
    - Shared feature extractor: MLP that processes observations
    - Actor head: Outputs action logits for policy distribution
    - Critic head: Outputs scalar state value estimate

    Architecture:
        Input (obs_size) -> Linear(256) -> ReLU -> Linear(256) -> ReLU -> features
        features -> Linear(128) -> ReLU -> Linear(action_size) -> action_logits (actor)
        features -> Linear(128) -> ReLU -> Linear(1) -> value (critic)

    Attributes:
        observation_size: Dimension of input observation vector
        action_size: Number of discrete actions
        hidden_size: Size of shared feature layers
        actor_hidden: Size of actor head hidden layer
        critic_hidden: Size of critic head hidden layer
    """

    def __init__(
        self,
        observation_size: int | None = None,
        action_size: int | None = None,
        hidden_size: int = 256,
        actor_hidden: int = 128,
        critic_hidden: int = 128,
    ):
        """Initialize the actor-critic network.

        Args:
            observation_size: Dimension of observation vector. Defaults to
                DEFAULT_CONFIG.total_size (414 with map encoding).
            action_size: Number of discrete actions. Defaults to ACTION_SPACE_SIZE (27).
            hidden_size: Size of shared feature extractor layers.
            actor_hidden: Size of actor head hidden layer.
            critic_hidden: Size of critic head hidden layer.
        """
        super().__init__()

        # Use defaults from existing modules
        self.observation_size = (
            observation_size
            if observation_size is not None
            else DEFAULT_CONFIG.total_size
        )
        self.action_size = action_size if action_size is not None else ACTION_SPACE_SIZE
        self.hidden_size = hidden_size
        self.actor_hidden = actor_hidden
        self.critic_hidden = critic_hidden

        # Shared feature extractor
        self.features = nn.Sequential(
            nn.Linear(self.observation_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, actor_hidden),
            nn.ReLU(),
            nn.Linear(actor_hidden, self.action_size),
        )

        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, critic_hidden),
            nn.ReLU(),
            nn.Linear(critic_hidden, 1),
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize network weights using orthogonal initialization.

        Uses orthogonal initialization with ReLU gain for hidden layers
        and smaller initialization for output layers to encourage
        exploration and stable training.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain("relu"))
                nn.init.zeros_(module.bias)

        # Smaller initialization for output layers
        # Actor output: small gain encourages initial exploration
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.zeros_(self.actor[-1].bias)

        # Critic output: standard gain for value estimation
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)
        nn.init.zeros_(self.critic[-1].bias)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning action logits and state value.

        Args:
            obs: Observation tensor of shape (batch, observation_size) or (observation_size,).

        Returns:
            Tuple of:
                - action_logits: Tensor of shape (batch, action_size) or (action_size,)
                - value: Tensor of shape (batch,) or scalar
        """
        features = self.features(obs)
        action_logits = self.actor(features)
        value = self.critic(features).squeeze(-1)
        return action_logits, value

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action, log probability, entropy, and value.

        This method is the primary interface for both rollout collection
        and policy evaluation during training.

        Args:
            obs: Observation tensor of shape (batch, observation_size).
            action: Optional action tensor. If None, sample from policy.
            deterministic: If True and action is None, use argmax instead of sampling.

        Returns:
            Tuple of:
                - action: Selected action tensor of shape (batch,)
                - log_prob: Log probability of the action, shape (batch,)
                - entropy: Entropy of the action distribution, shape (batch,)
                - value: Estimated state value, shape (batch,)
        """
        action_logits, value = self.forward(obs)
        dist = Categorical(logits=action_logits)

        if action is None:
            if deterministic:
                action = action_logits.argmax(dim=-1)
            else:
                action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy, value

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Get state value only.

        This method is more efficient than get_action_and_value when only
        the value is needed, such as for bootstrapping at the end of a rollout.

        Args:
            obs: Observation tensor of shape (batch, observation_size).

        Returns:
            State value tensor of shape (batch,).
        """
        features = self.features(obs)
        return self.critic(features).squeeze(-1)

    def get_action_distribution(self, obs: torch.Tensor) -> Categorical:
        """Get the action distribution for given observations.

        Useful for inspecting action probabilities or doing custom sampling.

        Args:
            obs: Observation tensor of shape (batch, observation_size).

        Returns:
            Categorical distribution over actions.
        """
        action_logits, _ = self.forward(obs)
        return Categorical(logits=action_logits)
