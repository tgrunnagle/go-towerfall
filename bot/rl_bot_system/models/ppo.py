"""
Proximal Policy Optimization (PPO) implementation for RL bot training.

This module implements PPO with actor-critic architecture, clipped surrogate objective,
and generalized advantage estimation for stable policy learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from collections import deque
import torch.distributions as distributions


class ActorNetwork(nn.Module):
    """
    Actor network for PPO that outputs action probabilities.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_sizes: list = [512, 256],
        output_size: int = 9,
        dropout: float = 0.1
    ):
        """
        Initialize actor network.
        
        Args:
            input_size: Size of input state representation
            hidden_sizes: List of hidden layer sizes
            output_size: Number of discrete actions
            dropout: Dropout rate for regularization
        """
        super(ActorNetwork, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
        
        self.feature_layers = nn.Sequential(*layers)
        self.action_head = nn.Linear(prev_size, output_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through actor network.
        
        Args:
            x: Input state tensor
            
        Returns:
            Action logits
        """
        features = self.feature_layers(x)
        action_logits = self.action_head(features)
        return action_logits
    
    def get_action_probs(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get action probabilities.
        
        Args:
            x: Input state tensor
            
        Returns:
            Action probabilities
        """
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)


class CriticNetwork(nn.Module):
    """
    Critic network for PPO that estimates state values.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_sizes: list = [512, 256],
        dropout: float = 0.1
    ):
        """
        Initialize critic network.
        
        Args:
            input_size: Size of input state representation
            hidden_sizes: List of hidden layer sizes
            dropout: Dropout rate for regularization
        """
        super(CriticNetwork, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
        
        self.feature_layers = nn.Sequential(*layers)
        self.value_head = nn.Linear(prev_size, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through critic network.
        
        Args:
            x: Input state tensor
            
        Returns:
            State value estimate
        """
        features = self.feature_layers(x)
        value = self.value_head(features)
        return value


class PPOBuffer:
    """
    Buffer for storing PPO rollout data.
    """
    
    def __init__(self, capacity: int = 2048):
        """
        Initialize PPO buffer.
        
        Args:
            capacity: Maximum number of transitions to store
        """
        self.capacity = capacity
        self.clear()
    
    def clear(self):
        """Clear the buffer."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.advantages = []
        self.returns = []
    
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        value: float,
        log_prob: float,
        done: bool
    ):
        """
        Add a transition to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            value: Value estimate
            log_prob: Log probability of action
            done: Whether episode ended
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def compute_advantages(self, gamma: float = 0.99, gae_lambda: float = 0.95):
        """
        Compute advantages using Generalized Advantage Estimation (GAE).
        
        Args:
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
        """
        advantages = []
        returns = []
        
        # Convert to numpy arrays for easier computation
        rewards = np.array(self.rewards)
        values = np.array(self.values)
        dones = np.array(self.dones)
        
        # Compute returns and advantages
        gae = 0
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[i + 1]
            
            delta = rewards[i] + gamma * next_value * (1 - dones[i]) - values[i]
            gae = delta + gamma * gae_lambda * (1 - dones[i]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[i])
        
        self.advantages = advantages
        self.returns = returns
    
    def get_batch(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        Get a batch of data for training.
        
        Args:
            batch_size: Size of batch to return
            
        Returns:
            Tuple of tensors for training
        """
        indices = np.random.choice(len(self.states), batch_size, replace=False)
        
        # Convert to numpy arrays first for efficiency
        states = torch.FloatTensor(np.array([self.states[i] for i in indices]))
        actions = torch.LongTensor(np.array([self.actions[i] for i in indices]))
        old_log_probs = torch.FloatTensor(np.array([self.log_probs[i] for i in indices]))
        advantages = torch.FloatTensor(np.array([self.advantages[i] for i in indices]))
        returns = torch.FloatTensor(np.array([self.returns[i] for i in indices]))
        
        return states, actions, old_log_probs, advantages, returns
    
    def __len__(self) -> int:
        return len(self.states)


class PPOAgent:
    """
    PPO agent with actor-critic architecture.
    
    Implements PPO algorithm with clipped surrogate objective and GAE.
    """
    
    def __init__(
        self,
        state_size: int,
        action_size: int,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 4,
        batch_size: int = 64,
        buffer_size: int = 2048,
        device: str = 'cpu'
    ):
        """
        Initialize PPO agent.
        
        Args:
            state_size: Size of state representation
            action_size: Number of discrete actions
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_epsilon: PPO clipping parameter
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Maximum gradient norm for clipping
            ppo_epochs: Number of PPO update epochs per rollout
            batch_size: Training batch size
            buffer_size: Rollout buffer size
            device: Device to run on ('cpu' or 'cuda')
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.device = torch.device(device)
        
        # Networks
        self.actor = ActorNetwork(state_size, output_size=action_size).to(self.device)
        self.critic = CriticNetwork(state_size).to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=learning_rate
        )
        
        # Buffer
        self.buffer = PPOBuffer(buffer_size)
        
        # Training tracking
        self.training_step = 0
        self.episode_count = 0
    
    def act(self, state: np.ndarray, training: bool = True) -> Tuple[int, float, float]:
        """
        Select action using current policy.
        
        Args:
            state: Current state
            training: Whether in training mode
            
        Returns:
            Tuple of (action, log_prob, value)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Get action probabilities and value
            action_logits = self.actor(state_tensor)
            value = self.critic(state_tensor)
            
            # Sample action
            action_probs = F.softmax(action_logits, dim=-1)
            dist = distributions.Categorical(action_probs)
            
            if training:
                action = dist.sample()
            else:
                action = action_probs.argmax()
            
            log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item(), value.item()
    
    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        value: float,
        log_prob: float,
        done: bool
    ):
        """
        Store experience in buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            value: Value estimate
            log_prob: Log probability of action
            done: Whether episode ended
        """
        self.buffer.push(state, action, reward, value, log_prob, done)
    
    def update(self) -> Dict[str, float]:
        """
        Update the policy using PPO.
        
        Returns:
            Dictionary of training metrics
        """
        if len(self.buffer) < self.buffer_size:
            return {}
        
        # Compute advantages
        self.buffer.compute_advantages(self.gamma, self.gae_lambda)
        
        # Normalize advantages
        advantages = torch.FloatTensor(self.buffer.advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        self.buffer.advantages = advantages.tolist()
        
        # Training metrics
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        
        # PPO update epochs
        for epoch in range(self.ppo_epochs):
            # Get batch
            states, actions, old_log_probs, batch_advantages, returns = self.buffer.get_batch(self.batch_size)
            
            states = states.to(self.device)
            actions = actions.to(self.device)
            old_log_probs = old_log_probs.to(self.device)
            batch_advantages = batch_advantages.to(self.device)
            returns = returns.to(self.device)
            
            # Forward pass
            action_logits = self.actor(states)
            values = self.critic(states).squeeze()
            
            # Action probabilities and entropy
            action_probs = F.softmax(action_logits, dim=-1)
            dist = distributions.Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            # Policy loss (clipped surrogate objective)
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(values, returns)
            
            # Total loss
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.actor.parameters()) + list(self.critic.parameters()),
                self.max_grad_norm
            )
            self.optimizer.step()
            
            # Track metrics
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy_loss += entropy.item()
        
        # Clear buffer
        self.buffer.clear()
        self.training_step += 1
        
        return {
            'policy_loss': total_policy_loss / self.ppo_epochs,
            'value_loss': total_value_loss / self.ppo_epochs,
            'entropy': total_entropy_loss / self.ppo_epochs,
            'training_step': self.training_step
        }
    
    def get_network_architecture(self) -> Dict[str, Any]:
        """Get network architecture configuration."""
        return {
            'actor': {
                'input_size': self.state_size,
                'hidden_sizes': [512, 256],
                'output_size': self.action_size,
                'dropout': 0.1
            },
            'critic': {
                'input_size': self.state_size,
                'hidden_sizes': [512, 256],
                'dropout': 0.1
            }
        }
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        """Get current hyperparameters."""
        return {
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'gae_lambda': self.gae_lambda,
            'clip_epsilon': self.clip_epsilon,
            'value_coef': self.value_coef,
            'entropy_coef': self.entropy_coef,
            'max_grad_norm': self.max_grad_norm,
            'ppo_epochs': self.ppo_epochs,
            'batch_size': self.batch_size,
            'buffer_size': self.buffer_size
        }
    
    def save_model(self, filepath: str):
        """Save model state."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_step': self.training_step,
            'episode_count': self.episode_count
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load model state."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_step = checkpoint['training_step']
        self.episode_count = checkpoint['episode_count']