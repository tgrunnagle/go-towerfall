"""
Base configuration classes for the RL bot system.
"""
import os
import yaml
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from pathlib import Path


@dataclass
class GameConfig:
    """Configuration for game connection and environment."""
    ws_url: str = "ws://localhost:4000/ws"
    http_url: str = "http://localhost:4000"
    default_room_code: Optional[str] = None
    default_player_name: str = "RL-Bot"
    connection_timeout: float = 30.0
    reconnect_attempts: int = 3
    reconnect_delay: float = 5.0


@dataclass
class StateRepresentationConfig:
    """Configuration for state representation options."""
    representation_type: str = "feature_vector"  # "raw_coordinates", "grid_based", "feature_vector", "hybrid"
    grid_resolution: int = 32  # For grid-based representation
    history_length: int = 4  # Number of previous frames to include
    normalize_coordinates: bool = True
    include_velocity: bool = True
    include_health: bool = True
    include_ammunition: bool = True
    feature_vector_size: int = 64


@dataclass
class ActionSpaceConfig:
    """Configuration for action space options."""
    action_type: str = "discrete"  # "discrete", "continuous", "hybrid"
    discrete_actions: List[str] = field(default_factory=lambda: [
        "no_action", "move_left", "move_right", "jump", "shoot", 
        "move_left_shoot", "move_right_shoot", "jump_shoot"
    ])
    continuous_movement_range: List[float] = field(default_factory=lambda: [-1.0, 1.0])
    continuous_aim_range: List[float] = field(default_factory=lambda: [0.0, 1.0])
    action_repeat: int = 1  # Number of frames to repeat each action


@dataclass
class RewardConfig:
    """Configuration for reward function options."""
    primary_function: str = "health_differential"  # "win_loss", "survival", "health_differential", "damage_dealt"
    secondary_functions: List[str] = field(default_factory=lambda: ["aim_accuracy", "survival"])
    weights: List[float] = field(default_factory=lambda: [0.7, 0.2, 0.1])
    horizon_weights: Dict[str, float] = field(default_factory=lambda: {
        "short_term": 0.5,
        "medium_term": 0.3,
        "long_term": 0.2
    })
    normalization: str = "z_score"  # "none", "min_max", "z_score"
    reward_clipping: Optional[List[float]] = field(default_factory=lambda: [-10.0, 10.0])
    exploration_bonus: float = 0.01


@dataclass
class TrainingConfig:
    """Configuration for RL training parameters."""
    algorithm: str = "PPO"  # "DQN", "PPO", "A3C", "SAC"
    total_timesteps: int = 1000000
    learning_rate: float = 3e-4
    batch_size: int = 64
    buffer_size: int = 100000
    gamma: float = 0.99  # Discount factor
    tau: float = 0.005  # Soft update coefficient
    exploration_fraction: float = 0.1
    exploration_initial_eps: float = 1.0
    exploration_final_eps: float = 0.02
    train_freq: int = 4
    gradient_steps: int = 1
    target_update_interval: int = 10000
    
    # PPO specific
    n_steps: int = 2048
    n_epochs: int = 10
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    
    # Training management
    save_freq: int = 10000
    eval_freq: int = 5000
    eval_episodes: int = 10
    log_interval: int = 100
    device: str = "auto"  # "auto", "cpu", "cuda"


@dataclass
class CohortTrainingConfig:
    """Configuration for cohort-based training against previous generations."""
    cohort_size: int = 3  # Number of previous generations to include
    enemy_count_range: List[int] = field(default_factory=lambda: [1, 3])  # Min/max enemies per episode
    opponent_selection: str = "weighted"  # "random", "weighted", "round_robin"
    difficulty_progression: bool = True
    rules_bot_weight: float = 0.3  # Weight for including rules-based bot in cohort
    generation_weights: List[float] = field(default_factory=lambda: [0.4, 0.3, 0.2])  # Weights for recent generations


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""
    evaluation_episodes: int = 100
    evaluation_opponents: List[str] = field(default_factory=lambda: ["rules_bot", "previous_generation", "random"])
    metrics_to_track: List[str] = field(default_factory=lambda: [
        "win_rate", "average_reward", "episode_length", "strategic_diversity"
    ])
    statistical_significance: float = 0.05
    min_improvement_threshold: float = 0.05  # Minimum improvement to promote a model


@dataclass
class ModelConfig:
    """Configuration for model architecture and management."""
    network_architecture: str = "mlp"  # "mlp", "cnn", "lstm", "transformer"
    hidden_layers: List[int] = field(default_factory=lambda: [256, 256])
    activation: str = "relu"  # "relu", "tanh", "elu"
    dropout_rate: float = 0.0
    batch_norm: bool = False
    
    # Model management
    max_generations: int = 50
    keep_best_n: int = 10  # Number of best models to keep
    knowledge_transfer_method: str = "weight_initialization"  # "weight_initialization", "behavior_cloning", "distillation"


@dataclass
class SpeedControlConfig:
    """Configuration for game speed control during training."""
    training_speed_multiplier: float = 10.0
    evaluation_speed_multiplier: float = 1.0
    headless_mode: bool = True
    max_parallel_episodes: int = 4
    batch_episode_size: int = 16


@dataclass
class LoggingConfig:
    """Configuration for logging and monitoring."""
    log_level: str = "INFO"
    log_to_file: bool = True
    log_file_path: str = "logs/rl_bot_system.log"
    use_tensorboard: bool = True
    tensorboard_log_dir: str = "logs/tensorboard"
    use_wandb: bool = False
    wandb_project: str = "rl-bot-system"
    wandb_entity: Optional[str] = None


@dataclass
class RLBotSystemConfig:
    """Main configuration class that combines all sub-configurations."""
    game: GameConfig = field(default_factory=GameConfig)
    state_representation: StateRepresentationConfig = field(default_factory=StateRepresentationConfig)
    action_space: ActionSpaceConfig = field(default_factory=ActionSpaceConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    cohort_training: CohortTrainingConfig = field(default_factory=CohortTrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    speed_control: SpeedControlConfig = field(default_factory=SpeedControlConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Paths
    data_dir: str = "data"
    models_dir: str = "data/models"
    replays_dir: str = "data/replays"
    metrics_dir: str = "data/metrics"
    logs_dir: str = "logs"
    
    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> 'RLBotSystemConfig':
        """Load configuration from YAML file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'RLBotSystemConfig':
        """Create configuration from dictionary."""
        # Create nested configurations
        game_config = GameConfig(**config_dict.get('game', {}))
        state_config = StateRepresentationConfig(**config_dict.get('state_representation', {}))
        action_config = ActionSpaceConfig(**config_dict.get('action_space', {}))
        reward_config = RewardConfig(**config_dict.get('reward', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        cohort_config = CohortTrainingConfig(**config_dict.get('cohort_training', {}))
        evaluation_config = EvaluationConfig(**config_dict.get('evaluation', {}))
        model_config = ModelConfig(**config_dict.get('model', {}))
        speed_config = SpeedControlConfig(**config_dict.get('speed_control', {}))
        logging_config = LoggingConfig(**config_dict.get('logging', {}))
        
        # Extract top-level configuration
        top_level = {k: v for k, v in config_dict.items() if k not in [
            'game', 'state_representation', 'action_space', 'reward', 'training',
            'cohort_training', 'evaluation', 'model', 'speed_control', 'logging'
        ]}
        
        return cls(
            game=game_config,
            state_representation=state_config,
            action_space=action_config,
            reward=reward_config,
            training=training_config,
            cohort_training=cohort_config,
            evaluation=evaluation_config,
            model=model_config,
            speed_control=speed_config,
            logging=logging_config,
            **top_level
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'game': self.game.__dict__,
            'state_representation': self.state_representation.__dict__,
            'action_space': self.action_space.__dict__,
            'reward': self.reward.__dict__,
            'training': self.training.__dict__,
            'cohort_training': self.cohort_training.__dict__,
            'evaluation': self.evaluation.__dict__,
            'model': self.model.__dict__,
            'speed_control': self.speed_control.__dict__,
            'logging': self.logging.__dict__,
            'data_dir': self.data_dir,
            'models_dir': self.models_dir,
            'replays_dir': self.replays_dir,
            'metrics_dir': self.metrics_dir,
            'logs_dir': self.logs_dir
        }
    
    def save_yaml(self, config_path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        # Validate paths exist or can be created
        for path_attr in ['data_dir', 'models_dir', 'replays_dir', 'metrics_dir', 'logs_dir']:
            path = Path(getattr(self, path_attr))
            path.mkdir(parents=True, exist_ok=True)
        
        # Validate algorithm choice
        valid_algorithms = ['DQN', 'PPO', 'A3C', 'SAC']
        if self.training.algorithm not in valid_algorithms:
            raise ValueError(f"Invalid algorithm: {self.training.algorithm}. Must be one of {valid_algorithms}")
        
        # Validate state representation
        valid_representations = ['raw_coordinates', 'grid_based', 'feature_vector', 'hybrid']
        if self.state_representation.representation_type not in valid_representations:
            raise ValueError(f"Invalid state representation: {self.state_representation.representation_type}")
        
        # Validate action space
        valid_action_types = ['discrete', 'continuous', 'hybrid']
        if self.action_space.action_type not in valid_action_types:
            raise ValueError(f"Invalid action type: {self.action_space.action_type}")
        
        # Validate reward weights sum to 1
        if abs(sum(self.reward.weights) - 1.0) > 1e-6:
            raise ValueError("Reward weights must sum to 1.0")
        
        # Validate horizon weights sum to 1
        if abs(sum(self.reward.horizon_weights.values()) - 1.0) > 1e-6:
            raise ValueError("Horizon weights must sum to 1.0")