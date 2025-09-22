"""
Deep Q-Network (DQN) implementation for RL bot training.

This module implements DQN with experience replay, target networks, and double DQN
for stable training in the game environment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple, Optional
from collections import deque
import random


class DQNNetwork(nn.Module):
    """
    Deep Q-Network architecture for game state processing.
    
    Supports different input types (coordinate, grid, feature vector) and
    produces Q-values for discrete actions.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_sizes: list = [512, 256, 128],
        output_size: int = 9,  # Default discrete action space size
        dueling: bool = True,
        dropout: float = 0.1
    ):
        """
        Initialize DQN network.
        
        Args:
            input_size: Size of input state representation
            hidden_sizes: List of hidden layer sizes
            output_size: Number of discrete actions
            dueling: Whether to use dueling DQN architecture
            dropout: Dropout rate for regularization
        """
        super(DQNNetwork, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.dueling = dueling
        
        # Build feature extraction layers
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
        
        if dueling:
            # Dueling DQN: separate value and advantage streams
            self.value_stream = nn.Linear(prev_size, 1)
            self.advantage_stream = nn.Linear(prev_size, output_size)
        else:
            # Standard DQN
            self.q_values = nn.Linear(prev_size, output_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input state tensor
            
        Returns:
            Q-values for each action
        """
        features = self.feature_layers(x)
        
        if self.dueling:
            # Dueling DQN computation
            value = self.value_stream(features)
            advantage = self.advantage_stream(features)
            
            # Q(s,a) = V(s) + A(s,a) - mean(A(s,a'))
            q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        else:
            q_values = self.q_values(features)
        
        return q_values


class ExperienceReplay:
    """
    Experience replay buffer for DQN training.
    
    Stores transitions and provides random sampling for training.
    """
    
    def __init__(self, capacity: int = 100000):
        """
        Initialize experience replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def push(
        self, 
        state: np.ndarray, 
        action: int, 
        reward: float, 
        next_state: np.ndarray, 
        done: bool
    ):
        """
        Add a transition to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        Sample a batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of tensors (states, actions, rewards, next_states, dones)
        """
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        batch = random.sample(self.buffer, batch_size)
        
        # Convert to numpy arrays first for efficiency
        states = torch.FloatTensor(np.array([t[0] for t in batch]))
        actions = torch.LongTensor(np.array([t[1] for t in batch]))
        rewards = torch.FloatTensor(np.array([t[2] for t in batch]))
        next_states = torch.FloatTensor(np.array([t[3] for t in batch]))
        dones = torch.tensor(np.array([bool(t[4]) for t in batch]), dtype=torch.bool)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    """
    DQN agent with experience replay and target network.
    
    Implements DQN algorithm with optional double DQN and dueling architecture.
    """
    
    def __init__(
        self,
        state_size: int,
        action_size: int,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        batch_size: int = 32,
        memory_size: int = 100000,
        target_update_freq: int = 1000,
        double_dqn: bool = True,
        dueling: bool = True,
        device: str = 'cpu'
    ):
        """
        Initialize DQN agent.
        
        Args:
            state_size: Size of state representation
            action_size: Number of discrete actions
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Exploration decay rate
            batch_size: Training batch size
            memory_size: Experience replay buffer size
            target_update_freq: Frequency of target network updates
            double_dqn: Whether to use double DQN
            dueling: Whether to use dueling architecture
            device: Device to run on ('cpu' or 'cuda')
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.double_dqn = double_dqn
        self.device = torch.device(device)
        
        # Networks
        self.q_network = DQNNetwork(
            state_size, 
            output_size=action_size, 
            dueling=dueling
        ).to(self.device)
        
        self.target_network = DQNNetwork(
            state_size, 
            output_size=action_size, 
            dueling=dueling
        ).to(self.device)
        
        # Initialize target network with same weights
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer and loss
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # Experience replay
        self.memory = ExperienceReplay(memory_size)
        
        # Training tracking
        self.training_step = 0
        self.episode_count = 0
        
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            training: Whether in training mode (affects exploration)
            
        Returns:
            Selected action
        """
        if training and np.random.random() <= self.epsilon:
            return np.random.choice(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        return q_values.argmax().item()
    
    def remember(
        self, 
        state: np.ndarray, 
        action: int, 
        reward: float, 
        next_state: np.ndarray, 
        done: bool
    ):
        """
        Store experience in replay buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        self.memory.push(state, action, reward, next_state, done)
    
    def replay(self) -> Optional[float]:
        """
        Train the network on a batch of experiences.
        
        Returns:
            Training loss if training occurred, None otherwise
        """
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values
        with torch.no_grad():
            if self.double_dqn:
                # Double DQN: use main network to select actions, target network to evaluate
                next_actions = self.q_network(next_states).argmax(1).unsqueeze(1)
                next_q_values = self.target_network(next_states).gather(1, next_actions)
            else:
                # Standard DQN: use target network for both selection and evaluation
                next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
            
            target_q_values = rewards.unsqueeze(1) + (self.gamma * next_q_values * ~dones.unsqueeze(1))
        
        # Compute loss
        loss = self.criterion(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update target network
        self.training_step += 1
        if self.training_step % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()
    
    def get_network_architecture(self) -> Dict[str, Any]:
        """Get network architecture configuration."""
        return {
            'input_size': self.state_size,
            'hidden_sizes': [512, 256, 128],  # Default from DQNNetwork
            'output_size': self.action_size,
            'dueling': True,  # Default from DQNNetwork
            'dropout': 0.1   # Default from DQNNetwork
        }
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        """Get current hyperparameters."""
        return {
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'epsilon_min': self.epsilon_min,
            'epsilon_decay': self.epsilon_decay,
            'batch_size': self.batch_size,
            'memory_size': self.memory.capacity,
            'target_update_freq': self.target_update_freq,
            'double_dqn': self.double_dqn
        }
    
    def save_model(self, filepath: str):
        """Save model state."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step,
            'episode_count': self.episode_count
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load model state."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.training_step = checkpoint['training_step']
        self.episode_count = checkpoint['episode_count']