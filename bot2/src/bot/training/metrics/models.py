"""Pydantic models for training metrics.

This module defines data classes for capturing and validating training metrics
during reinforcement learning, including episode metrics, training step metrics,
and aggregated statistics.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class EpisodeMetrics(BaseModel):
    """Metrics captured at the end of each episode.

    These metrics track individual episode performance including rewards,
    episode length, and game-specific statistics like kills and deaths.

    Attributes:
        episode_id: Unique identifier for this episode
        total_reward: Cumulative reward received during the episode
        length: Number of timesteps in the episode
        kills: Number of kills during the episode
        deaths: Number of deaths during the episode
        win: Whether the episode resulted in a win
        timestamp: When the episode completed
    """

    model_config = ConfigDict(frozen=True)

    episode_id: int = Field(ge=0, description="Unique episode identifier")
    total_reward: float = Field(description="Cumulative episode reward")
    length: int = Field(ge=0, description="Episode length in timesteps")
    kills: int = Field(ge=0, default=0, description="Number of kills")
    deaths: int = Field(ge=0, default=0, description="Number of deaths")
    win: bool = Field(default=False, description="Whether episode was a win")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Episode completion timestamp",
    )


class TrainingStepMetrics(BaseModel):
    """Metrics captured during each PPO training update.

    These metrics track the optimization process including losses,
    entropy, and KL divergence.

    Attributes:
        step: Training step number (PPO update count)
        policy_loss: Mean policy loss from PPO update
        value_loss: Mean value function loss
        entropy: Mean policy entropy
        kl_divergence: Approximate KL divergence between old and new policy
        clip_fraction: Fraction of samples where PPO clipping was applied
        learning_rate: Current learning rate
        total_timesteps: Total timesteps collected so far
    """

    model_config = ConfigDict(frozen=True)

    step: int = Field(ge=0, description="Training step number")
    policy_loss: float = Field(description="Mean policy loss")
    value_loss: float = Field(description="Mean value function loss")
    entropy: float = Field(description="Mean policy entropy")
    kl_divergence: float = Field(ge=0, description="Approximate KL divergence")
    clip_fraction: float = Field(ge=0, le=1, description="Fraction of samples clipped")
    learning_rate: float = Field(gt=0, description="Current learning rate")
    total_timesteps: int = Field(ge=0, description="Total timesteps collected")


class AggregateMetrics(BaseModel):
    """Aggregated statistics computed over a rolling window of episodes.

    These metrics provide summary statistics for monitoring training progress
    and determining when a model has improved over its predecessor.

    Attributes:
        mean_reward: Mean episode reward over the window
        std_reward: Standard deviation of episode rewards
        mean_length: Mean episode length over the window
        win_rate: Fraction of episodes won (0.0 to 1.0)
        mean_kills: Mean kills per episode
        mean_deaths: Mean deaths per episode
        kd_ratio: Aggregate kill/death ratio (total kills / total deaths)
        episodes_count: Number of episodes in the aggregate
        timestamp: When the aggregate was computed
    """

    model_config = ConfigDict(frozen=True)

    mean_reward: float = Field(description="Mean episode reward")
    std_reward: float = Field(ge=0, description="Standard deviation of rewards")
    mean_length: float = Field(ge=0, description="Mean episode length")
    win_rate: float = Field(ge=0, le=1, description="Fraction of episodes won")
    mean_kills: float = Field(ge=0, description="Mean kills per episode")
    mean_deaths: float = Field(ge=0, description="Mean deaths per episode")
    kd_ratio: float = Field(ge=0, description="Aggregate kill/death ratio")
    episodes_count: int = Field(ge=0, description="Number of episodes in aggregate")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Aggregate computation timestamp",
    )
