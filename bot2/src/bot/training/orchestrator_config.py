"""Configuration for the training orchestrator.

This module defines the OrchestratorConfig dataclass that holds all configuration
options for the TrainingOrchestrator, including environment settings, training
hyperparameters, checkpointing, and logging.
"""

from dataclasses import dataclass, field

from bot.agent.ppo_trainer import PPOConfig
from bot.training.server_manager import TrainingGameConfig


@dataclass
class OrchestratorConfig:
    """Configuration for the training orchestrator.

    This dataclass consolidates all settings needed to configure the training
    orchestrator, including environment configuration, PPO hyperparameters,
    checkpointing settings, and logging options.

    Attributes:
        num_envs: Number of parallel environments for training
        game_server_url: URL of the game server HTTP API
        game_config: Configuration for training game instances
        ppo_config: PPO hyperparameter configuration
        total_timesteps: Total training timesteps before completion
        checkpoint_interval: Timesteps between saving checkpoints
        checkpoint_dir: Directory to save training checkpoints
        registry_path: Path to the model registry storage
        opponent_model_id: Model ID for opponent (None = rule-based bot)
        log_interval: Timesteps between logging progress
        eval_interval: Timesteps between evaluation runs
        eval_episodes: Number of episodes per evaluation
        seed: Random seed for reproducibility (None = random)

    Example:
        config = OrchestratorConfig(
            num_envs=4,
            total_timesteps=500_000,
            ppo_config=PPOConfig(learning_rate=3e-4),
        )
    """

    # Environment settings
    num_envs: int = 4
    game_server_url: str = "http://localhost:4000"
    game_config: TrainingGameConfig = field(
        default_factory=lambda: TrainingGameConfig(room_name="Training")
    )

    # Training settings
    ppo_config: PPOConfig = field(default_factory=lambda: PPOConfig())
    total_timesteps: int = 1_000_000

    # Checkpointing
    checkpoint_interval: int = 10_000
    checkpoint_dir: str = "./checkpoints"

    # Model registry
    registry_path: str = "./model_registry"

    # Opponent settings
    opponent_model_id: str | None = None

    # Logging
    log_interval: int = 2048

    # Evaluation
    eval_interval: int = 50_000
    eval_episodes: int = 10

    # Seeds
    seed: int | None = None

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.num_envs < 1:
            raise ValueError("num_envs must be at least 1")
        if self.total_timesteps < 1:
            raise ValueError("total_timesteps must be at least 1")
        if self.checkpoint_interval < 1:
            raise ValueError("checkpoint_interval must be at least 1")
        if self.log_interval < 1:
            raise ValueError("log_interval must be at least 1")
        if self.eval_interval < 1:
            raise ValueError("eval_interval must be at least 1")
        if self.eval_episodes < 1:
            raise ValueError("eval_episodes must be at least 1")

    @property
    def steps_per_rollout(self) -> int:
        """Total environment steps collected per rollout.

        Returns:
            num_steps * num_envs
        """
        return self.ppo_config.num_steps * self.num_envs
