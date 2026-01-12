"""Model metadata Pydantic models for the model registry.

This module defines the data models for tracking model versions, training metrics,
and metadata in the model registry system.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class TrainingMetrics(BaseModel):
    """Metrics captured during training.

    These metrics track the performance of a trained model and are used
    to compare models across generations in the successive training pipeline.

    Attributes:
        total_episodes: Number of episodes completed during training
        total_timesteps: Total environment steps collected
        average_reward: Mean episode reward during training
        average_episode_length: Mean episode length in timesteps
        win_rate: Percentage of episodes won (0.0 to 1.0)
        average_kills: Mean kills per episode
        average_deaths: Mean deaths per episode
        kills_deaths_ratio: Ratio of kills to deaths (the "better" metric)
    """

    model_config = ConfigDict(frozen=True)

    total_episodes: int = Field(ge=0, description="Number of episodes completed")
    total_timesteps: int = Field(ge=0, description="Total environment steps collected")
    average_reward: float = Field(description="Mean episode reward")
    average_episode_length: float = Field(ge=0, description="Mean episode length")
    win_rate: float = Field(
        ge=0.0, le=1.0, description="Percentage of episodes won (0.0 to 1.0)"
    )
    average_kills: float = Field(ge=0, description="Mean kills per episode")
    average_deaths: float = Field(ge=0, description="Mean deaths per episode")
    kills_deaths_ratio: float = Field(
        ge=0, description="Ratio of kills to deaths (K/D ratio)"
    )


class NetworkArchitecture(BaseModel):
    """Neural network architecture parameters.

    These parameters define the structure of the saved model and are used
    to validate compatibility when loading models.

    Attributes:
        observation_size: Dimension of the observation vector
        action_size: Number of discrete actions
        hidden_size: Size of shared feature extractor layers
        actor_hidden: Size of actor head hidden layer
        critic_hidden: Size of critic head hidden layer
    """

    model_config = ConfigDict(frozen=True)

    observation_size: int = Field(gt=0, description="Observation vector dimension")
    action_size: int = Field(gt=0, description="Number of discrete actions")
    hidden_size: int = Field(gt=0, default=256, description="Shared hidden layer size")
    actor_hidden: int = Field(gt=0, default=128, description="Actor hidden layer size")
    critic_hidden: int = Field(
        gt=0, default=128, description="Critic hidden layer size"
    )


class ModelMetadata(BaseModel):
    """Metadata for a registered model.

    This model contains all the information needed to track, retrieve,
    and compare trained models in the registry.

    Attributes:
        model_id: Unique identifier for this model (e.g., "ppo_gen_003")
        generation: Generation number in successive training (0 for first model)
        created_at: Timestamp when the model was registered
        training_duration_seconds: Total training time in seconds
        opponent_model_id: ID of opponent model trained against (None for gen 0)
        training_metrics: Performance metrics from training
        hyperparameters: PPO hyperparameters used during training
        architecture: Neural network architecture parameters
        checkpoint_path: Relative path to the serialized model file
        notes: Optional human-readable notes about this model
    """

    model_config = ConfigDict(frozen=True)

    model_id: str = Field(description="Unique model identifier (e.g., ppo_gen_003)")
    generation: int = Field(
        ge=0, description="Generation number in successive training"
    )
    created_at: datetime = Field(description="Timestamp when model was registered")
    training_duration_seconds: float = Field(
        ge=0, description="Total training time in seconds"
    )
    opponent_model_id: str | None = Field(
        default=None, description="ID of opponent model trained against"
    )
    training_metrics: TrainingMetrics = Field(description="Performance metrics")
    hyperparameters: dict = Field(
        default_factory=dict, description="PPO hyperparameters used"
    )
    architecture: NetworkArchitecture = Field(description="Neural network architecture")
    checkpoint_path: str = Field(description="Relative path to serialized model file")
    notes: str | None = Field(default=None, description="Optional notes about model")
