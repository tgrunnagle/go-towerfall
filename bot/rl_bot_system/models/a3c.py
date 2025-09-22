"""
Advantage Actor-Critic (A2C/A3C) implementation for RL bot training.

This module implements A2C (synchronous version of A3C) with actor-critic architecture
and n-step returns for stable policy learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from collections import deque
import torch.distributions as distributions


class A2CNetwork(nn.Module):
    """
    Combined actor-critic network for A2C.
    
    Shares feature extraction layers between actor and critic for efficiency.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_sizes: list = [512, 256],
        action_size: int = 9,
        dropout: float = 0.1
    ):
        """
        Initialize A2C network.
        
        Args:
            input_size: Size of input state representation
            hidden_sizes: List of hidden layer sizes
            action_size: Number of discrete actions
            dropout: Dropout rate for regularization
        """
        super(A2CNetwork, self).__init__()
        
        self.input_size = input_size
        self.action_size = action_size
        
        # Shared feature extraction layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
        
        self.shared_layers = nn.Sequential(*layers)
        
        # Actor head (policy)
        self.actor_head = nn.Linear(prev_size, action_size)
        
        # Critic head (value function)
        self.critic_head = nn.Linear(prev_size, 1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            x: Input state tensor
            
        Returns:
            Tuple of (action_logits, value)
        """
        shared_features = self.shared_layers(x)
        
        action_logits = self.actor_head(shared_features)
        value = self.critic_head(shared_features)
        
        return action_logits, value
    
    def get_action_probs(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get action probabilities.
        
        Args:
            x: Input state tensor
            
        Returns:
            Action probabilities
        """
        action_logits, _ = self.forward(x)
        return F.softmax(action_logits, dim=-1)
    
    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get state value estimate.
        
        Args:
            x: Input state tensor
            
        Returns:
            State value
        """
        _, value = self.forward(x)
        return value


class A2CBuffer:
    """
    Buffer for storing A2C rollout data with n-step returns.
    """
    
    def __init__(self, n_steps: int = 5):
        """
        Initialize A2C buffer.
        
        Args:
            n_steps: Number of steps for n-step returns
        """
        self.n_steps = n_steps
        self.clear()
    
    def clear(self):
        """Clear the buffer."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
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
    
    def compute_returns_and_advantages(
        self, 
        next_value: float = 0.0, 
        gamma: float = 0.99
    ) -> Tuple[List[float], List[float]]:
        """
        Compute n-step returns and advantages.
        
        Args:
            next_value: Value of the next state (for bootstrapping)
            gamma: Discount factor
            
        Returns:
            Tuple of (returns, advantages)
        """
        returns = []
        advantages = []
        
        # Convert to numpy for easier computation
        rewards = np.array(self.rewards)
        values = np.array(self.values + [next_value])
        dones = np.array(self.dones)
        
        # Compute n-step returns
        for i in range(len(rewards)):
            n_step_return = 0
            for j in range(min(self.n_steps, len(rewards) - i)):
                if dones[i + j]:
                    n_step_return += (gamma ** j) * rewards[i + j]
                    break
                else:
                    n_step_return += (gamma ** j) * rewards[i + j]
            
            # Add bootstrapped value if we didn't hit a terminal state
            if not any(dones[i:i + self.n_steps]):
                bootstrap_idx = min(i + self.n_steps, len(values) - 1)
                n_step_return += (gamma ** self.n_steps) * values[bootstrap_idx]
            
            returns.append(n_step_return)
            advantages.append(n_step_return - values[i])
        
        return returns, advantages
    
    def get_data(self) -> Tuple[torch.Tensor, ...]:
        """
        Get all data as tensors.
        
        Returns:
            Tuple of tensors for training
        """
        states = torch.FloatTensor(self.states)
        actions = torch.LongTensor(self.actions)
        log_probs = torch.FloatTensor(self.log_probs)
        
        return states, actions, log_probs
    
    def __len__(self) -> int:
        return len(self.states)


class A2CAgent:
    """
    A2C agent with shared actor-critic network.
    
    Implements A2C algorithm with n-step returns and entropy regularization.
    """
    
    def __init__(
        self,
        state_size: int,
        action_size: int,
        learning_rate: float = 7e-4,
        gamma: float = 0.99,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        n_steps: int = 5,
        device: str = 'cpu'
    ):
        """
        Initialize A2C agent.
        
        Args:
            state_size: Size of state representation
            action_size: Number of discrete actions
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Maximum gradient norm for clipping
            n_steps: Number of steps for n-step returns
            device: Device to run on ('cpu' or 'cuda')
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_steps = n_steps
        self.device = torch.device(device)
        
        # Network
        self.network = A2CNetwork(
            state_size, 
            action_size=action_size
        ).to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.RMSprop(
            self.network.parameters(), 
            lr=learning_rate,
            eps=1e-5,
            alpha=0.99
        )
        
        # Buffer
        self.buffer = A2CBuffer(n_steps)
        
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
            action_logits, value = self.network(state_tensor)
            
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
    
    def update(self, next_state: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Update the policy using A2C.
        
        Args:
            next_state: Next state for bootstrapping (if episode didn't end)
            
        Returns:
            Dictionary of training metrics
        """
        if len(self.buffer) == 0:
            return {}
        
        # Get next value for bootstrapping
        next_value = 0.0
        if next_state is not None:
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                _, next_value_tensor = self.network(next_state_tensor)
                next_value = next_value_tensor.item()
        
        # Compute returns and advantages
        returns, advantages = self.buffer.compute_returns_and_advantages(
            next_value, self.gamma
        )
        
        # Get buffer data
        states, actions, old_log_probs = self.buffer.get_data()
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        old_log_probs = old_log_probs.to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Forward pass
        action_logits, values = self.network(states)
        values = values.squeeze()
        
        # Action probabilities and entropy
        action_probs = F.softmax(action_logits, dim=-1)
        dist = distributions.Categorical(action_probs)
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        
        # Policy loss
        policy_loss = -(new_log_probs * advantages).mean()
        
        # Value loss
        value_loss = F.mse_loss(values, returns)
        
        # Total loss
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        # Clear buffer
        self.buffer.clear()
        self.training_step += 1
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'training_step': self.training_step
        }
    
    def get_network_architecture(self) -> Dict[str, Any]:
        """Get network architecture configuration."""
        return {
            'input_size': self.state_size,
            'hidden_sizes': [512, 256],
            'action_size': self.action_size,
            'dropout': 0.1
        }
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        """Get current hyperparameters."""
        return {
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'value_coef': self.value_coef,
            'entropy_coef': self.entropy_coef,
            'max_grad_norm': self.max_grad_norm,
            'n_steps': self.n_steps
        }
    
    def save_model(self, filepath: str):
        """Save model state."""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_step': self.training_step,
            'episode_count': self.episode_count
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load model state."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_step = checkpoint['training_step']
        self.episode_count = checkpoint['episode_count']