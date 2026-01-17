"""Configuration for successive self-play training.

This module defines configuration dataclasses for the successive training
pipeline that progressively trains agents against stronger opponents.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from bot.training.orchestrator_config import OrchestratorConfig


@dataclass
class PromotionCriteria:
    """Criteria for determining when an agent is "better" than its opponent.

    An agent is promoted when ALL specified criteria are met. The primary
    comparison metric is kills/deaths ratio (K/D), which must exceed the
    opponent's baseline by a configurable margin.

    Attributes:
        min_kd_ratio: Minimum K/D ratio the agent must achieve
        kd_improvement: Minimum percentage improvement over opponent K/D
        min_win_rate: Optional minimum win rate requirement (None = disabled)
        min_eval_episodes: Minimum episodes for reliable evaluation
        confidence_threshold: Statistical confidence level for promotion
        consecutive_passes: Times criteria must be passed consecutively
    """

    # Kill/Death ratio thresholds
    min_kd_ratio: float = 1.0
    kd_improvement: float = 0.1

    # Win rate (optional)
    min_win_rate: float | None = None

    # Consistency requirements
    min_eval_episodes: int = 50
    confidence_threshold: float = 0.95

    # Stability requirements
    consecutive_passes: int = 3

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.min_kd_ratio < 0:
            raise ValueError("min_kd_ratio must be non-negative")
        if self.kd_improvement < 0:
            raise ValueError("kd_improvement must be non-negative")
        if self.min_win_rate is not None and not 0 <= self.min_win_rate <= 1:
            raise ValueError("min_win_rate must be between 0 and 1")
        if self.min_eval_episodes < 1:
            raise ValueError("min_eval_episodes must be at least 1")
        if not 0 < self.confidence_threshold < 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if self.consecutive_passes < 1:
            raise ValueError("consecutive_passes must be at least 1")


@dataclass
class SuccessiveTrainingConfig:
    """Configuration for successive self-play training.

    Defines how the training pipeline progresses through generations,
    evaluates agent performance, and determines when to promote agents.

    Attributes:
        base_config: Base orchestrator config used for each generation
        max_generations: Maximum number of generations to train
        initial_opponent: Starting opponent type ("rule_based" or "none")
        timesteps_per_generation: Training timesteps per generation
        promotion_criteria: Criteria for agent promotion decisions
        evaluation_interval: Timesteps between evaluation runs
        evaluation_episodes: Number of episodes per evaluation
        max_stagnant_evaluations: Stop if no progress for N evaluations
        min_improvement_rate: Minimum K/D improvement per evaluation
        output_dir: Base directory for all outputs
        base_seed: Base random seed (incremented per generation)

    Example:
        config = SuccessiveTrainingConfig(
            base_config=OrchestratorConfig(num_envs=4),
            max_generations=5,
            timesteps_per_generation=500_000,
        )
    """

    # Orchestrator config (used for each generation)
    base_config: OrchestratorConfig = field(default_factory=OrchestratorConfig)

    # Generation settings
    max_generations: int = 10
    initial_opponent: str = "rule_based"

    # Per-generation training
    timesteps_per_generation: int = 500_000

    # Promotion criteria
    promotion_criteria: PromotionCriteria = field(default_factory=PromotionCriteria)

    # Evaluation settings
    evaluation_interval: int = 50_000
    evaluation_episodes: int = 100

    # Early stopping
    max_stagnant_evaluations: int = 10
    min_improvement_rate: float = 0.01

    # Output settings
    output_dir: str = "./successive_training"

    # Seeds for reproducibility
    base_seed: int | None = None

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.max_generations < 1:
            raise ValueError("max_generations must be at least 1")
        if self.initial_opponent not in ("rule_based", "none"):
            raise ValueError("initial_opponent must be 'rule_based' or 'none'")
        if self.timesteps_per_generation < 1:
            raise ValueError("timesteps_per_generation must be at least 1")
        if self.evaluation_interval < 1:
            raise ValueError("evaluation_interval must be at least 1")
        if self.evaluation_episodes < 1:
            raise ValueError("evaluation_episodes must be at least 1")
        if self.max_stagnant_evaluations < 1:
            raise ValueError("max_stagnant_evaluations must be at least 1")
        if self.min_improvement_rate < 0:
            raise ValueError("min_improvement_rate must be non-negative")

    def create_generation_config(
        self,
        generation: int,
        opponent_model_id: str | None,
    ) -> OrchestratorConfig:
        """Create orchestrator config for a specific generation.

        Args:
            generation: Generation number (0-indexed)
            opponent_model_id: Model ID for opponent (None for rule-based)

        Returns:
            OrchestratorConfig configured for this generation
        """
        # Copy base config values
        generation_dir = Path(self.output_dir) / f"generation_{generation:03d}"

        # Calculate seed for this generation
        seed = None
        if self.base_seed is not None:
            seed = self.base_seed + generation

        return OrchestratorConfig(
            num_envs=self.base_config.num_envs,
            game_server_url=self.base_config.game_server_url,
            game_config=self.base_config.game_config,
            ppo_config=self.base_config.ppo_config,
            metrics_config=self.base_config.metrics_config,
            total_timesteps=self.timesteps_per_generation,
            checkpoint_interval=self.base_config.checkpoint_interval,
            checkpoint_dir=str(generation_dir / "checkpoints"),
            registry_path=str(Path(self.output_dir) / "model_registry"),
            opponent_model_id=opponent_model_id,
            log_interval=self.base_config.log_interval,
            eval_interval=self.evaluation_interval,
            eval_episodes=self.evaluation_episodes,
            max_eval_episode_steps=self.base_config.max_eval_episode_steps,
            seed=seed,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to a dictionary.

        Returns:
            Dictionary representation of the configuration.
        """
        return {
            "base_config": self.base_config.to_dict(),
            "max_generations": self.max_generations,
            "initial_opponent": self.initial_opponent,
            "timesteps_per_generation": self.timesteps_per_generation,
            "promotion_criteria": {
                "min_kd_ratio": self.promotion_criteria.min_kd_ratio,
                "kd_improvement": self.promotion_criteria.kd_improvement,
                "min_win_rate": self.promotion_criteria.min_win_rate,
                "min_eval_episodes": self.promotion_criteria.min_eval_episodes,
                "confidence_threshold": self.promotion_criteria.confidence_threshold,
                "consecutive_passes": self.promotion_criteria.consecutive_passes,
            },
            "evaluation_interval": self.evaluation_interval,
            "evaluation_episodes": self.evaluation_episodes,
            "max_stagnant_evaluations": self.max_stagnant_evaluations,
            "min_improvement_rate": self.min_improvement_rate,
            "output_dir": self.output_dir,
            "base_seed": self.base_seed,
        }

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file.

        Args:
            path: Path to the output YAML file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SuccessiveTrainingConfig":
        """Create configuration from a dictionary.

        Args:
            data: Dictionary containing configuration values.

        Returns:
            SuccessiveTrainingConfig instance.
        """
        # Extract nested configs
        base_config_data = data.get("base_config", {})
        promotion_data = data.get("promotion_criteria", {})

        # Build nested config objects
        base_config = OrchestratorConfig.from_dict(base_config_data)
        promotion_criteria = PromotionCriteria(**promotion_data)

        # Filter out nested config keys
        remaining_data = {
            k: v
            for k, v in data.items()
            if k not in ("base_config", "promotion_criteria")
        }

        return cls(
            base_config=base_config,
            promotion_criteria=promotion_criteria,
            **remaining_data,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SuccessiveTrainingConfig":
        """Load configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            SuccessiveTrainingConfig instance.

        Raises:
            FileNotFoundError: If the file does not exist.
            yaml.YAMLError: If the file is not valid YAML.
        """
        path = Path(path)
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)
