"""Configuration for the training orchestrator.

This module defines the OrchestratorConfig dataclass that holds all configuration
options for the TrainingOrchestrator, including environment settings, training
hyperparameters, checkpointing, and logging.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

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

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to a nested dictionary.

        Converts all dataclass fields to dictionaries, including nested
        PPOConfig and TrainingGameConfig objects.

        Returns:
            Dictionary representation of the configuration.
        """
        return {
            "num_envs": self.num_envs,
            "game_server_url": self.game_server_url,
            "game_config": asdict(self.game_config),
            "ppo_config": asdict(self.ppo_config),
            "total_timesteps": self.total_timesteps,
            "checkpoint_interval": self.checkpoint_interval,
            "checkpoint_dir": self.checkpoint_dir,
            "registry_path": self.registry_path,
            "opponent_model_id": self.opponent_model_id,
            "log_interval": self.log_interval,
            "eval_interval": self.eval_interval,
            "eval_episodes": self.eval_episodes,
            "seed": self.seed,
        }

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file.

        Args:
            path: Path to the output YAML file.

        Example:
            config = OrchestratorConfig(num_envs=8)
            config.to_yaml("training_config.yaml")
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OrchestratorConfig":
        """Create configuration from a dictionary.

        Handles nested PPOConfig and TrainingGameConfig objects.

        Args:
            data: Dictionary containing configuration values.

        Returns:
            OrchestratorConfig instance.

        Example:
            config = OrchestratorConfig.from_dict({
                "num_envs": 8,
                "total_timesteps": 500_000,
            })
        """
        # Extract nested configs and convert them
        ppo_config_data = data.pop("ppo_config", None)
        game_config_data = data.pop("game_config", None)

        # Build nested config objects
        ppo_config = PPOConfig(**ppo_config_data) if ppo_config_data else PPOConfig()
        game_config = (
            TrainingGameConfig(**game_config_data)
            if game_config_data
            else TrainingGameConfig(room_name="Training")
        )

        return cls(
            ppo_config=ppo_config,
            game_config=game_config,
            **data,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "OrchestratorConfig":
        """Load configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            OrchestratorConfig instance.

        Raises:
            FileNotFoundError: If the file does not exist.
            yaml.YAMLError: If the file is not valid YAML.

        Example:
            config = OrchestratorConfig.from_yaml("training_config.yaml")
        """
        path = Path(path)
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)
