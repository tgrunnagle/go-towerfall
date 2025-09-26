"""
RL Training Engine for successive bot training.

This module implements the core training loop that orchestrates RL model training,
episode management, progress tracking, and behavior cloning initialization.
"""

import asyncio
import logging
import numpy as np
import torch
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import time

from rl_bot_system.models import DQNAgent, PPOAgent, A2CAgent
from rl_bot_system.environment import GameEnvironment, TrainingMode
from rl_bot_system.training.model_manager import ModelManager, RLModel
from rl_bot_system.training.cohort_training import CohortTrainingSystem, CohortConfig
from rl_bot_system.rules_based.rules_based_bot import RulesBasedBot
from game_client import GameClient

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for RL training."""
    algorithm: str  # 'DQN', 'PPO', 'A2C'
    max_episodes: int = 1000
    max_steps_per_episode: int = 1000
    evaluation_frequency: int = 100
    evaluation_episodes: int = 10
    save_frequency: int = 50
    
    # Algorithm-specific hyperparameters
    hyperparameters: Dict[str, Any] = None
    
    # Behavior cloning settings
    use_behavior_cloning: bool = True
    bc_episodes: int = 100
    bc_learning_rate: float = 1e-3
    
    # Training environment settings
    training_mode: TrainingMode = TrainingMode.TRAINING
    game_speed_multiplier: float = 10.0
    
    # Cohort training settings
    cohort_config: Optional[CohortConfig] = None
    
    def __post_init__(self):
        if self.hyperparameters is None:
            self.hyperparameters = self._get_default_hyperparameters()
    
    def _get_default_hyperparameters(self) -> Dict[str, Any]:
        """Get default hyperparameters for each algorithm."""
        defaults = {
            'DQN': {
                'learning_rate': 1e-4,
                'gamma': 0.99,
                'epsilon': 1.0,
                'epsilon_min': 0.01,
                'epsilon_decay': 0.995,
                'batch_size': 32,
                'memory_size': 100000,
                'target_update_freq': 1000,
                'double_dqn': True,
                'dueling': True
            },
            'PPO': {
                'learning_rate': 3e-4,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_epsilon': 0.2,
                'value_coef': 0.5,
                'entropy_coef': 0.01,
                'max_grad_norm': 0.5,
                'ppo_epochs': 4,
                'batch_size': 64,
                'buffer_size': 2048
            },
            'A2C': {
                'learning_rate': 7e-4,
                'gamma': 0.99,
                'value_coef': 0.5,
                'entropy_coef': 0.01,
                'max_grad_norm': 0.5,
                'n_steps': 5
            }
        }
        return defaults.get(self.algorithm, {})


@dataclass
class TrainingMetrics:
    """Metrics tracked during training."""
    episode: int
    total_reward: float
    episode_length: int
    loss: Optional[float] = None
    epsilon: Optional[float] = None  # For DQN
    policy_loss: Optional[float] = None  # For PPO/A2C
    value_loss: Optional[float] = None  # For PPO/A2C
    entropy: Optional[float] = None  # For PPO/A2C
    win_rate: Optional[float] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class EvaluationResult:
    """Results from model evaluation."""
    generation: int
    total_episodes: int
    wins: int
    losses: int
    draws: int
    average_reward: float
    win_rate: float
    average_episode_length: float
    evaluation_time: datetime = None
    
    def __post_init__(self):
        if self.evaluation_time is None:
            self.evaluation_time = datetime.now()


class TrainingEngine:
    """
    Main training engine for RL bot training.
    
    Orchestrates the training process including:
    - Model initialization and management
    - Training loop execution
    - Progress tracking and logging
    - Behavior cloning initialization
    - Model evaluation and comparison
    """
    
    def __init__(
        self,
        model_manager: ModelManager,
        cohort_system: Optional[CohortTrainingSystem] = None,
        device: str = 'cpu'
    ):
        """
        Initialize training engine.
        
        Args:
            model_manager: ModelManager for model lifecycle
            cohort_system: CohortTrainingSystem for opponent selection
            device: Device to run training on ('cpu' or 'cuda')
        """
        self.model_manager = model_manager
        self.cohort_system = cohort_system
        self.device = device
        self.logger = logging.getLogger(__name__)
        
        # Training state
        self.current_agent = None
        self.current_config = None
        self.training_metrics: List[TrainingMetrics] = []
        self.evaluation_results: List[EvaluationResult] = []
        
        # Progress tracking
        self.episodes_completed = 0
        self.total_training_time = 0.0
        self.best_performance = -float('inf')
        
    def train_next_generation(
        self,
        generation: int,
        config: TrainingConfig,
        environment: GameEnvironment,
        previous_model: Optional[RLModel] = None
    ) -> RLModel:
        """
        Train the next generation of RL model.
        
        Args:
            generation: Generation number for this model
            config: Training configuration
            environment: Game environment for training
            previous_model: Previous generation model for knowledge transfer
            
        Returns:
            Trained RLModel with metadata
        """
        self.logger.info(f"Starting training for generation {generation} using {config.algorithm}")
        self.current_config = config
        
        # Initialize agent
        self.current_agent = self._create_agent(config, environment)
        
        # Behavior cloning initialization if requested
        if config.use_behavior_cloning and previous_model is None:
            self.logger.info("Performing behavior cloning initialization from rules-based bot")
            self._behavior_cloning_initialization(config, environment)
        
        # Knowledge transfer from previous model if available
        if previous_model is not None:
            self.logger.info(f"Transferring knowledge from generation {previous_model.generation}")
            self._transfer_knowledge(previous_model)
        
        # Main training loop
        start_time = time.time()
        self._training_loop(config, environment)
        training_time = time.time() - start_time
        
        # Final evaluation
        final_evaluation = self._evaluate_model(environment, config.evaluation_episodes)
        
        # Save model
        metadata = {
            'algorithm': config.algorithm,
            'network_architecture': self.current_agent.get_network_architecture(),
            'hyperparameters': self.current_agent.get_hyperparameters(),
            'training_episodes': self.episodes_completed,
            'performance_metrics': {
                'final_evaluation': asdict(final_evaluation),
                'training_time': training_time,
                'best_reward': self.best_performance,
                'total_episodes': len(self.training_metrics)
            },
            'parent_generation': previous_model.generation if previous_model else None
        }
        
        # Get the actual model for saving
        if hasattr(self.current_agent, 'q_network'):
            model_to_save = self.current_agent.q_network  # DQN
        elif hasattr(self.current_agent, 'actor'):
            model_to_save = self.current_agent.actor  # PPO (save actor)
        else:
            model_to_save = self.current_agent.network  # A2C
        
        saved_model = self.model_manager.save_model(model_to_save, generation, metadata)
        
        self.logger.info(f"Generation {generation} training completed. "
                        f"Final win rate: {final_evaluation.win_rate:.2%}")
        
        return saved_model
    
    def _create_agent(self, config: TrainingConfig, environment: GameEnvironment):
        """Create RL agent based on configuration."""
        state_size = environment.observation_space.shape[0]
        action_size = environment.action_space.n
        
        if config.algorithm == 'DQN':
            return DQNAgent(
                state_size=state_size,
                action_size=action_size,
                device=self.device,
                **config.hyperparameters
            )
        elif config.algorithm == 'PPO':
            return PPOAgent(
                state_size=state_size,
                action_size=action_size,
                device=self.device,
                **config.hyperparameters
            )
        elif config.algorithm == 'A2C':
            return A2CAgent(
                state_size=state_size,
                action_size=action_size,
                device=self.device,
                **config.hyperparameters
            )
        else:
            raise ValueError(f"Unsupported algorithm: {config.algorithm}")
    
    def _behavior_cloning_initialization(self, config: TrainingConfig, environment: GameEnvironment):
        """
        Initialize agent using behavior cloning from rules-based bot.
        
        Args:
            config: Training configuration
            environment: Game environment
        """
        self.logger.info("Starting behavior cloning initialization")
        
        # Create rules-based bot for demonstration
        rules_bot = RulesBasedBot()
        
        # Collect demonstration data
        states = []
        actions = []
        
        for episode in range(config.bc_episodes):
            state, _ = environment.reset()
            episode_done = False
            
            while not episode_done:
                # Get action from rules-based bot
                # Note: This is a simplified interface - actual implementation
                # would need to convert between state representations
                rules_action = rules_bot.select_action(state, environment.action_space.sample())
                
                states.append(state)
                actions.append(rules_action)
                
                # Take step in environment
                next_state, reward, terminated, truncated, info = environment.step(rules_action)
                episode_done = terminated or truncated
                state = next_state
        
        # Train agent on demonstration data
        if len(states) > 0:
            self._train_behavior_cloning(states, actions, config.bc_learning_rate)
        
        self.logger.info(f"Behavior cloning completed with {len(states)} demonstrations")
    
    def _train_behavior_cloning(self, states: List, actions: List, learning_rate: float):
        """
        Train agent using behavior cloning on demonstration data.
        
        Args:
            states: List of demonstration states
            actions: List of demonstration actions
            learning_rate: Learning rate for behavior cloning
        """
        # Convert to tensors (convert to numpy array first for efficiency)
        states_tensor = torch.FloatTensor(np.array(states)).to(self.device)
        actions_tensor = torch.LongTensor(np.array(actions)).to(self.device)
        
        # Create temporary optimizer for behavior cloning
        if hasattr(self.current_agent, 'q_network'):
            # DQN - train Q-network to predict actions
            network = self.current_agent.q_network
        elif hasattr(self.current_agent, 'actor'):
            # PPO - train actor network
            network = self.current_agent.actor
        else:
            # A2C - train the shared network
            network = self.current_agent.network
        
        optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss()
        
        # Training loop
        batch_size = min(32, len(states))  # Ensure batch size doesn't exceed data size
        num_batches = max(1, len(states) // batch_size)  # Ensure at least 1 batch
        
        for epoch in range(10):  # Multiple epochs over the data
            total_loss = 0
            
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, len(states))  # Ensure we don't exceed bounds
                
                batch_states = states_tensor[start_idx:end_idx]
                batch_actions = actions_tensor[start_idx:end_idx]
                
                # Forward pass
                if hasattr(self.current_agent, 'q_network'):
                    logits = network(batch_states)
                elif hasattr(self.current_agent, 'actor'):
                    logits = network(batch_states)
                else:
                    logits, _ = network(batch_states)  # A2C returns both logits and values
                
                # Compute loss
                loss = criterion(logits, batch_actions)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / num_batches
            self.logger.debug(f"Behavior cloning epoch {epoch + 1}, loss: {avg_loss:.4f}")
    
    def _transfer_knowledge(self, previous_model: RLModel):
        """
        Transfer knowledge from previous generation model.
        
        Args:
            previous_model: Previous generation model to transfer from
        """
        try:
            # Load previous model
            previous_agent_class = self._get_agent_class(previous_model.algorithm)
            previous_agent = previous_agent_class(
                state_size=self.current_agent.state_size,
                action_size=self.current_agent.action_size,
                device=self.device,
                **previous_model.hyperparameters
            )
            previous_agent.load_model(previous_model.model_path)
            
            # Transfer weights (simplified - copy network weights)
            if hasattr(self.current_agent, 'q_network') and hasattr(previous_agent, 'q_network'):
                # DQN to DQN transfer
                self.current_agent.q_network.load_state_dict(previous_agent.q_network.state_dict())
            elif hasattr(self.current_agent, 'actor') and hasattr(previous_agent, 'actor'):
                # PPO to PPO transfer
                self.current_agent.actor.load_state_dict(previous_agent.actor.state_dict())
                self.current_agent.critic.load_state_dict(previous_agent.critic.state_dict())
            elif hasattr(self.current_agent, 'network') and hasattr(previous_agent, 'network'):
                # A2C to A2C transfer
                self.current_agent.network.load_state_dict(previous_agent.network.state_dict())
            
            self.logger.info("Knowledge transfer completed successfully")
            
        except Exception as e:
            self.logger.warning(f"Knowledge transfer failed: {e}. Starting from scratch.")
    
    def _get_agent_class(self, algorithm: str):
        """Get agent class for given algorithm."""
        if algorithm == 'DQN':
            return DQNAgent
        elif algorithm == 'PPO':
            return PPOAgent
        elif algorithm == 'A2C':
            return A2CAgent
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    
    def _training_loop(self, config: TrainingConfig, environment: GameEnvironment):
        """
        Main training loop.
        
        Args:
            config: Training configuration
            environment: Game environment
        """
        self.episodes_completed = 0
        self.training_metrics = []
        
        for episode in range(config.max_episodes):
            episode_start_time = time.time()
            
            # Reset environment
            state, _ = environment.reset()
            episode_reward = 0
            episode_length = 0
            episode_done = False
            
            # Episode loop
            while not episode_done and episode_length < config.max_steps_per_episode:
                # Select action
                if hasattr(self.current_agent, 'act'):
                    if config.algorithm == 'DQN':
                        action = self.current_agent.act(state, training=True)
                        log_prob = None
                        value = None
                    else:  # PPO or A2C
                        action, log_prob, value = self.current_agent.act(state, training=True)
                else:
                    action = environment.action_space.sample()  # Fallback
                    log_prob = None
                    value = None
                
                # Take step
                next_state, reward, terminated, truncated, info = environment.step(action)
                episode_done = terminated or truncated
                
                # Store experience
                if config.algorithm == 'DQN':
                    self.current_agent.remember(state, action, reward, next_state, episode_done)
                    
                    # Train DQN
                    if len(self.current_agent.memory) > self.current_agent.batch_size:
                        loss = self.current_agent.replay()
                else:  # PPO or A2C
                    if log_prob is not None and value is not None:
                        self.current_agent.remember(state, action, reward, value, log_prob, episode_done)
                
                episode_reward += reward
                episode_length += 1
                state = next_state
            
            # Update policy for PPO/A2C
            training_info = {}
            if config.algorithm in ['PPO', 'A2C']:
                if config.algorithm == 'PPO' and len(self.current_agent.buffer) >= self.current_agent.buffer_size:
                    training_info = self.current_agent.update()
                elif config.algorithm == 'A2C':
                    training_info = self.current_agent.update(next_state if not episode_done else None)
            
            # Track metrics
            metrics = TrainingMetrics(
                episode=episode,
                total_reward=episode_reward,
                episode_length=episode_length,
                loss=training_info.get('policy_loss'),
                epsilon=getattr(self.current_agent, 'epsilon', None),
                policy_loss=training_info.get('policy_loss'),
                value_loss=training_info.get('value_loss'),
                entropy=training_info.get('entropy')
            )
            self.training_metrics.append(metrics)
            
            # Update best performance
            if episode_reward > self.best_performance:
                self.best_performance = episode_reward
            
            # Logging
            if episode % 10 == 0:
                episode_time = time.time() - episode_start_time
                self.logger.info(
                    f"Episode {episode}: reward={episode_reward:.2f}, "
                    f"length={episode_length}, time={episode_time:.2f}s"
                )
            
            # Evaluation
            if episode % config.evaluation_frequency == 0 and episode > 0:
                eval_result = self._evaluate_model(environment, config.evaluation_episodes)
                self.evaluation_results.append(eval_result)
                self.logger.info(f"Evaluation at episode {episode}: win_rate={eval_result.win_rate:.2%}")
            
            self.episodes_completed += 1
    
    def _evaluate_model(self, environment: GameEnvironment, num_episodes: int) -> EvaluationResult:
        """
        Evaluate current model performance.
        
        Args:
            environment: Game environment for evaluation
            num_episodes: Number of episodes to evaluate
            
        Returns:
            EvaluationResult with performance metrics
        """
        wins = 0
        losses = 0
        draws = 0
        total_reward = 0
        total_length = 0
        
        # Set environment to evaluation mode
        original_mode = environment.training_mode
        environment.set_training_mode(TrainingMode.EVALUATION)
        
        try:
            for episode in range(num_episodes):
                state, _ = environment.reset()
                episode_reward = 0
                episode_length = 0
                episode_done = False
                
                while not episode_done:
                    # Select action (no exploration)
                    if hasattr(self.current_agent, 'act'):
                        if self.current_config.algorithm == 'DQN':
                            action = self.current_agent.act(state, training=False)
                        else:
                            action, _, _ = self.current_agent.act(state, training=False)
                    else:
                        action = environment.action_space.sample()
                    
                    next_state, reward, terminated, truncated, info = environment.step(action)
                    episode_done = terminated or truncated
                    
                    episode_reward += reward
                    episode_length += 1
                    state = next_state
                
                total_reward += episode_reward
                total_length += episode_length
                
                # Determine outcome (simplified - based on reward)
                if episode_reward > 0:
                    wins += 1
                elif episode_reward < 0:
                    losses += 1
                else:
                    draws += 1
        
        finally:
            # Restore original training mode
            environment.set_training_mode(original_mode)
        
        return EvaluationResult(
            generation=0,  # Will be set by caller
            total_episodes=num_episodes,
            wins=wins,
            losses=losses,
            draws=draws,
            average_reward=total_reward / num_episodes,
            win_rate=wins / num_episodes,
            average_episode_length=total_length / num_episodes
        )
    
    def get_training_progress(self) -> Dict[str, Any]:
        """Get current training progress and metrics."""
        return {
            'episodes_completed': self.episodes_completed,
            'total_training_time': self.total_training_time,
            'best_performance': self.best_performance,
            'recent_metrics': [asdict(m) for m in self.training_metrics[-10:]],
            'evaluation_results': [asdict(r) for r in self.evaluation_results]
        }
    
    def save_training_progress(self, filepath: str):
        """Save training progress to file."""
        progress_data = {
            'training_metrics': [asdict(m) for m in self.training_metrics],
            'evaluation_results': [asdict(r) for r in self.evaluation_results],
            'episodes_completed': self.episodes_completed,
            'total_training_time': self.total_training_time,
            'best_performance': self.best_performance
        }
        
        with open(filepath, 'w') as f:
            json.dump(progress_data, f, indent=2, default=str)
    
    def load_training_progress(self, filepath: str):
        """Load training progress from file."""
        with open(filepath, 'r') as f:
            progress_data = json.load(f)
        
        self.episodes_completed = progress_data.get('episodes_completed', 0)
        self.total_training_time = progress_data.get('total_training_time', 0.0)
        self.best_performance = progress_data.get('best_performance', -float('inf'))
        
        # Reconstruct metrics objects
        self.training_metrics = []
        for m_data in progress_data.get('training_metrics', []):
            m_data['timestamp'] = datetime.fromisoformat(m_data['timestamp'])
            self.training_metrics.append(TrainingMetrics(**m_data))
        
        self.evaluation_results = []
        for r_data in progress_data.get('evaluation_results', []):
            r_data['evaluation_time'] = datetime.fromisoformat(r_data['evaluation_time'])
            self.evaluation_results.append(EvaluationResult(**r_data))