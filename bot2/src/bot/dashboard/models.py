"""Pydantic models for dashboard metrics.

This module defines data models for generation-level metrics aggregation
used in the training comparison dashboard.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class GenerationMetrics(BaseModel):
    """Aggregated metrics for a single model generation.

    These metrics summarize training performance for one generation in the
    successive training pipeline, enabling comparison across generations.

    Attributes:
        generation_id: Generation number in the successive training pipeline
        model_version: Model ID from the registry (e.g., "ppo_gen_003")
        opponent_type: Description of opponent trained against
        total_episodes: Number of training episodes completed
        total_kills: Total kills across all episodes
        total_deaths: Total deaths across all episodes
        kill_death_ratio: Ratio of total kills to total deaths (K/D)
        win_rate: Percentage of episodes won (0.0 to 1.0)
        avg_episode_reward: Mean reward across all episodes
        avg_episode_length: Mean episode length in timesteps
        training_steps: Total PPO training updates performed
        training_duration_seconds: Wall-clock training time in seconds
        timestamp: When this generation's training completed
    """

    model_config = ConfigDict(frozen=True)

    generation_id: int = Field(ge=0, description="Generation number")
    model_version: str = Field(description="Model ID from registry")
    opponent_type: str = Field(description="Opponent trained against")
    total_episodes: int = Field(ge=0, description="Total training episodes")
    total_kills: int = Field(ge=0, description="Total kills")
    total_deaths: int = Field(ge=0, description="Total deaths")
    kill_death_ratio: float = Field(ge=0, description="Kill/death ratio")
    win_rate: float = Field(ge=0, le=1, description="Win rate (0.0 to 1.0)")
    avg_episode_reward: float = Field(description="Mean episode reward")
    avg_episode_length: float = Field(ge=0, description="Mean episode length")
    training_steps: int = Field(ge=0, description="Total PPO updates")
    training_duration_seconds: float = Field(
        ge=0, description="Training time in seconds"
    )
    timestamp: datetime = Field(description="Training completion timestamp")


class DashboardConfig(BaseModel):
    """Configuration for dashboard generation.

    Attributes:
        registry_path: Path to model registry directory
        metrics_dir: Path to training metrics logs (optional)
        output_dir: Directory for generated dashboard files
        output_format: Output format(s) to generate
        generations: Optional range of generations to include
        title: Dashboard title
    """

    model_config = ConfigDict(frozen=True)

    registry_path: str = Field(
        default="./model_registry",
        description="Path to model registry",
    )
    metrics_dir: str | None = Field(
        default=None,
        description="Path to training metrics logs",
    )
    output_dir: str = Field(
        default="./reports",
        description="Output directory for dashboard",
    )
    output_format: str = Field(
        default="html",
        description="Output format: 'html', 'png', or 'both'",
    )
    generations: tuple[int, int] | None = Field(
        default=None,
        description="Optional (start, end) generation range to include",
    )
    title: str = Field(
        default="Training Generation Comparison",
        description="Dashboard title",
    )
