"""Configuration for the training orchestrator.

This module defines the OrchestratorConfig dataclass that holds all configuration
options for the TrainingOrchestrator, including environment settings, training
hyperparameters, checkpointing, logging, and metrics.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from bot.agent.ppo_trainer import PPOConfig
from bot.training.server_manager import TrainingGameConfig


@dataclass
class MetricsLoggerConfig:
    """Configuration for training metrics logging.

    Controls what metrics are logged during training and where they are persisted.

    Attributes:
        enabled: Whether metrics logging is enabled
        log_dir: Directory for metrics logs (relative to checkpoint_dir if not absolute)
        enable_tensorboard: Enable TensorBoard logging for visualization
        enable_file: Enable file-based logging (JSON/CSV)
        file_format: Format for file logging ("json" or "csv")
        window_size: Rolling window size for aggregate statistics

    Example:
        config = MetricsLoggerConfig(
            log_dir="logs/run_001",
            enable_tensorboard=True,
            file_format="json",
        )
    """

    enabled: bool = True
    log_dir: str = "./metrics"
    enable_tensorboard: bool = True
    enable_file: bool = True
    file_format: Literal["json", "csv"] = "json"
    window_size: int = 100

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.window_size < 1:
            raise ValueError("window_size must be at least 1")
        if self.file_format not in ("json", "csv"):
            raise ValueError("file_format must be 'json' or 'csv'")


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
        metrics_config: Configuration for training metrics logging
        total_timesteps: Total training timesteps before completion
        checkpoint_interval: Timesteps between saving checkpoints
        checkpoint_dir: Directory to save training checkpoints
        registry_path: Path to the model registry storage
        opponent_model_id: Model ID for opponent (None = rule-based bot)
        log_interval: Timesteps between logging progress
        eval_interval: Timesteps between evaluation runs
        eval_episodes: Number of episodes per evaluation
        max_eval_episode_steps: Maximum steps per evaluation episode
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
    metrics_config: MetricsLoggerConfig = field(
        default_factory=lambda: MetricsLoggerConfig()
    )
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
    max_eval_episode_steps: int = 10_000

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
        if self.max_eval_episode_steps < 1:
            raise ValueError("max_eval_episode_steps must be at least 1")

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
        PPOConfig, TrainingGameConfig, and MetricsLoggerConfig objects.

        Returns:
            Dictionary representation of the configuration.
        """
        return {
            "num_envs": self.num_envs,
            "game_server_url": self.game_server_url,
            "game_config": asdict(self.game_config),
            "ppo_config": asdict(self.ppo_config),
            "metrics_config": asdict(self.metrics_config),
            "total_timesteps": self.total_timesteps,
            "checkpoint_interval": self.checkpoint_interval,
            "checkpoint_dir": self.checkpoint_dir,
            "registry_path": self.registry_path,
            "opponent_model_id": self.opponent_model_id,
            "log_interval": self.log_interval,
            "eval_interval": self.eval_interval,
            "eval_episodes": self.eval_episodes,
            "max_eval_episode_steps": self.max_eval_episode_steps,
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

        Handles nested PPOConfig, TrainingGameConfig, and MetricsLoggerConfig objects.

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
        ppo_config_data = data.get("ppo_config", None)
        game_config_data = data.get("game_config", None)
        metrics_config_data = data.get("metrics_config", None)

        # Build nested config objects
        ppo_config = PPOConfig(**ppo_config_data) if ppo_config_data else PPOConfig()
        game_config = (
            TrainingGameConfig(**game_config_data)
            if game_config_data
            else TrainingGameConfig(room_name="Training")
        )
        metrics_config = (
            MetricsLoggerConfig(**metrics_config_data)
            if metrics_config_data
            else MetricsLoggerConfig()
        )

        # Filter out nested config keys to avoid duplicate arguments
        remaining_data = {
            k: v
            for k, v in data.items()
            if k not in ("ppo_config", "game_config", "metrics_config")
        }

        return cls(
            ppo_config=ppo_config,
            game_config=game_config,
            metrics_config=metrics_config,
            **remaining_data,
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
