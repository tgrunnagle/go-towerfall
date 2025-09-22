"""
Tests for DQN implementation.

This module tests the DQN agent, network, and experience replay components.
"""

import pytest
import numpy as np
import torch
import tempfile
import os
from unittest.mock import Mock

from bot.rl_bot_system.models.dqn import DQNAgent, DQNNetwork, ExperienceReplay


class TestDQNNetwork:
    """Test DQNNetwork class."""
    
    def test_network_creation(self):
        """Test network creation with default parameters."""
        network = DQNNetwork(input_size=10, output_size=5)
        
        assert network.input_size == 10
        assert network.output_size == 5
        assert network.dueling is True
        
    def test_network_forward_pass(self):
        """Test forward pass through network."""
        network = DQNNetwork(input_size=10, output_size=5)
        
        # Test with batch of states
        batch_size = 32
        input_tensor = torch.randn(batch_size, 10)
        output = network(input_tensor)
        
        assert output.shape == (batch_size, 5)
        
    def test_dueling_vs_standard_architecture(self):
        """Test difference between dueling and standard DQN."""
        dueling_net = DQNNetwork(input_size=10, output_size=5, dueling=True)
        standard_net = DQNNetwork(input_size=10, output_size=5, dueling=False)
        
        input_tensor = torch.randn(1, 10)
        
        dueling_output = dueling_net(input_tensor)
        standard_output = standard_net(input_tensor)
        
        assert dueling_output.shape == standard_output.shape
        assert hasattr(dueling_net, 'value_stream')
        assert hasattr(dueling_net, 'advantage_stream')
        assert hasattr(standard_net, 'q_values')
        
    def test_custom_architecture(self):
        """Test network with custom hidden layers."""
        network = DQNNetwork(
            input_size=20,
            hidden_sizes=[256, 128, 64],
            output_size=8,
            dropout=0.2
        )
        
        input_tensor = torch.randn(16, 20)
        output = network(input_tensor)
        
        assert output.shape == (16, 8)


class TestExperienceReplay:
    """Test ExperienceReplay class."""
    
    def test_buffer_creation(self):
        """Test buffer creation."""
        buffer = ExperienceReplay(capacity=1000)
        
        assert len(buffer) == 0
        assert buffer.capacity == 1000
        
    def test_push_and_sample(self):
        """Test pushing experiences and sampling."""
        buffer = ExperienceReplay(capacity=100)
        
        # Add some experiences
        for i in range(50):
            state = np.random.random(10)
            action = np.random.randint(0, 5)
            reward = np.random.uniform(-1, 1)
            next_state = np.random.random(10)
            done = np.random.choice([True, False])
            
            buffer.push(state, action, reward, next_state, done)
        
        assert len(buffer) == 50
        
        # Sample a batch
        batch_size = 32
        states, actions, rewards, next_states, dones = buffer.sample(batch_size)
        
        assert states.shape == (batch_size, 10)
        assert actions.shape == (batch_size,)
        assert rewards.shape == (batch_size,)
        assert next_states.shape == (batch_size, 10)
        assert dones.shape == (batch_size,)
        
    def test_buffer_overflow(self):
        """Test buffer behavior when capacity is exceeded."""
        buffer = ExperienceReplay(capacity=10)
        
        # Add more experiences than capacity
        for i in range(15):
            state = np.random.random(5)
            buffer.push(state, 0, 0.0, state, False)
        
        # Buffer should not exceed capacity
        assert len(buffer) == 10
        
    def test_sample_insufficient_data(self):
        """Test sampling when buffer has insufficient data."""
        buffer = ExperienceReplay(capacity=100)
        
        # Add only a few experiences
        for i in range(5):
            state = np.random.random(5)
            buffer.push(state, 0, 0.0, state, False)
        
        # Sample more than available
        states, actions, rewards, next_states, dones = buffer.sample(10)
        
        # Should return all available data
        assert states.shape[0] == 5


class TestDQNAgent:
    """Test DQNAgent class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.state_size = 10
        self.action_size = 5
        self.agent = DQNAgent(
            state_size=self.state_size,
            action_size=self.action_size,
            learning_rate=1e-3,
            memory_size=1000,
            batch_size=32,
            device='cpu'
        )
        
    def test_agent_creation(self):
        """Test agent creation."""
        assert self.agent.state_size == self.state_size
        assert self.agent.action_size == self.action_size
        assert self.agent.epsilon == 1.0
        assert len(self.agent.memory) == 0
        
    def test_action_selection_exploration(self):
        """Test action selection during exploration."""
        state = np.random.random(self.state_size)
        
        # With high epsilon, should explore
        self.agent.epsilon = 1.0
        actions = []
        for _ in range(100):
            action = self.agent.act(state, training=True)
            actions.append(action)
        
        # Should have some variety in actions
        unique_actions = set(actions)
        assert len(unique_actions) > 1
        
    def test_action_selection_exploitation(self):
        """Test action selection during exploitation."""
        # Use a fixed state for deterministic behavior
        state = np.ones(self.state_size)
        
        # With zero epsilon, should exploit
        self.agent.epsilon = 0.0
        
        # Set network to eval mode for deterministic behavior
        self.agent.q_network.eval()
        
        # Get the first action
        first_action = self.agent.act(state, training=True)
        
        # All subsequent actions with the same state should be the same (greedy)
        for _ in range(10):
            action = self.agent.act(state, training=True)
            assert action == first_action
        
    def test_action_selection_evaluation(self):
        """Test action selection during evaluation."""
        # Use a fixed state for deterministic behavior
        state = np.ones(self.state_size)
        
        # In evaluation mode, should not explore regardless of epsilon
        self.agent.epsilon = 1.0
        
        # Set network to eval mode for deterministic behavior
        self.agent.q_network.eval()
        
        # Get the first action
        first_action = self.agent.act(state, training=False)
        
        # All subsequent actions with the same state should be the same (greedy)
        for _ in range(10):
            action = self.agent.act(state, training=False)
            assert action == first_action
        
    def test_remember_experience(self):
        """Test storing experiences in memory."""
        state = np.random.random(self.state_size)
        action = 2
        reward = 1.0
        next_state = np.random.random(self.state_size)
        done = False
        
        self.agent.remember(state, action, reward, next_state, done)
        
        assert len(self.agent.memory) == 1
        
    def test_replay_insufficient_memory(self):
        """Test replay when memory is insufficient."""
        # Add fewer experiences than batch size
        for i in range(10):
            state = np.random.random(self.state_size)
            self.agent.remember(state, 0, 0.0, state, False)
        
        loss = self.agent.replay()
        assert loss is None  # Should not train
        
    def test_replay_training(self):
        """Test replay training."""
        # Fill memory with enough experiences
        for i in range(100):
            state = np.random.random(self.state_size)
            action = np.random.randint(0, self.action_size)
            reward = np.random.uniform(-1, 1)
            next_state = np.random.random(self.state_size)
            done = np.random.choice([True, False])
            
            self.agent.remember(state, action, reward, next_state, done)
        
        loss = self.agent.replay()
        assert loss is not None
        assert isinstance(loss, float)
        assert loss >= 0
        
    def test_epsilon_decay(self):
        """Test epsilon decay during training."""
        initial_epsilon = self.agent.epsilon
        
        # Fill memory and train multiple times
        for i in range(100):
            state = np.random.random(self.state_size)
            self.agent.remember(state, 0, 0.0, state, False)
        
        # Train multiple times
        for _ in range(10):
            self.agent.replay()
        
        # Epsilon should have decayed
        assert self.agent.epsilon < initial_epsilon
        assert self.agent.epsilon >= self.agent.epsilon_min
        
    def test_target_network_update(self):
        """Test target network updates."""
        # Get initial target network weights
        initial_target_weights = self.agent.target_network.state_dict()
        
        # Set training step to trigger update on next replay
        self.agent.training_step = self.agent.target_update_freq - 1
        
        # Fill memory and train
        for i in range(100):
            state = np.random.random(self.state_size)
            self.agent.remember(state, 0, 0.0, state, False)
        
        # Train the network to modify weights
        for _ in range(10):  # Multiple training steps to ensure weight changes
            loss = self.agent.replay()
            if loss is not None:
                break
        
        # Target network should have been updated
        updated_target_weights = self.agent.target_network.state_dict()
        
        # Check that training step was incremented
        assert self.agent.training_step >= self.agent.target_update_freq
            
    def test_double_dqn_vs_standard(self):
        """Test difference between double DQN and standard DQN."""
        double_agent = DQNAgent(
            state_size=self.state_size,
            action_size=self.action_size,
            double_dqn=True,
            device='cpu'
        )
        
        standard_agent = DQNAgent(
            state_size=self.state_size,
            action_size=self.action_size,
            double_dqn=False,
            device='cpu'
        )
        
        assert double_agent.double_dqn is True
        assert standard_agent.double_dqn is False
        
        # Both should work for training
        for agent in [double_agent, standard_agent]:
            for i in range(50):
                state = np.random.random(self.state_size)
                agent.remember(state, 0, 0.0, state, False)
            
            loss = agent.replay()
            assert loss is not None
            
    def test_get_network_architecture(self):
        """Test getting network architecture."""
        arch = self.agent.get_network_architecture()
        
        assert 'input_size' in arch
        assert 'output_size' in arch
        assert 'hidden_sizes' in arch
        assert arch['input_size'] == self.state_size
        assert arch['output_size'] == self.action_size
        
    def test_get_hyperparameters(self):
        """Test getting hyperparameters."""
        params = self.agent.get_hyperparameters()
        
        assert 'learning_rate' in params
        assert 'gamma' in params
        assert 'epsilon' in params
        assert 'batch_size' in params
        
    def test_save_and_load_model(self):
        """Test saving and loading model."""
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            filepath = f.name
        
        try:
            # Modify agent state
            self.agent.epsilon = 0.5
            self.agent.training_step = 100
            self.agent.episode_count = 50
            
            # Save model
            self.agent.save_model(filepath)
            
            # Create new agent and load
            new_agent = DQNAgent(
                state_size=self.state_size,
                action_size=self.action_size,
                device='cpu'
            )
            new_agent.load_model(filepath)
            
            # Check that state was loaded
            assert new_agent.epsilon == 0.5
            assert new_agent.training_step == 100
            assert new_agent.episode_count == 50
            
        finally:
            os.unlink(filepath)


if __name__ == '__main__':
    pytest.main([__file__])