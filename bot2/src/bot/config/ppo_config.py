"""PPO hyperparameter configuration with Pydantic validation.

This module provides type-safe configuration classes for PPO training,
with support for YAML file loading and sensible defaults based on
PPO best practices.
"""

from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class PPOCoreConfig(BaseModel):
    """Core PPO algorithm hyperparameters.

    These parameters control the fundamental PPO optimization process,
    including clipping, learning rate, and loss coefficients.
    """

    model_config = ConfigDict(frozen=True)

    learning_rate: float = Field(
        3e-4,
        gt=0,
        description="Learning rate for Adam optimizer",
    )
    clip_range: float = Field(
        0.2,
        gt=0,
        le=1.0,
        description="PPO clipping parameter (epsilon)",
    )
    clip_range_vf: float | None = Field(
        None,
        ge=0,
        description="Value function clipping range (None = no clipping)",
    )
    n_epochs: int = Field(
        10,
        ge=1,
        description="Number of epochs per PPO update",
    )
    batch_size: int = Field(
        64,
        ge=1,
        description="Minibatch size for training",
    )
    n_steps: int = Field(
        2048,
        ge=1,
        description="Number of steps to collect per rollout",
    )
    gamma: float = Field(
        0.99,
        ge=0,
        le=1.0,
        description="Discount factor for rewards",
    )
    gae_lambda: float = Field(
        0.95,
        ge=0,
        le=1.0,
        description="GAE lambda for advantage estimation",
    )
    ent_coef: float = Field(
        0.01,
        ge=0,
        description="Entropy coefficient for exploration",
    )
    vf_coef: float = Field(
        0.5,
        ge=0,
        description="Value function loss coefficient",
    )
    max_grad_norm: float = Field(
        0.5,
        ge=0,
        description="Maximum gradient norm for clipping",
    )


class NetworkConfig(BaseModel):
    """Neural network architecture configuration.

    These parameters define the structure of the actor-critic network,
    including hidden layer sizes and activation functions.
    """

    model_config = ConfigDict(frozen=True)

    hidden_sizes: list[int] = Field(
        default=[64, 64],
        min_length=1,
        description="Hidden layer sizes for policy/value networks",
    )
    activation: Literal["tanh", "relu", "elu"] = Field(
        "tanh",
        description="Activation function for hidden layers",
    )
    share_features: bool = Field(
        True,
        description="Whether actor and critic share feature layers",
    )
    ortho_init: bool = Field(
        True,
        description="Use orthogonal weight initialization",
    )


class TrainingConfig(BaseModel):
    """Training loop configuration.

    These parameters control the overall training process,
    including total timesteps, device selection, and early stopping.
    """

    model_config = ConfigDict(frozen=True)

    total_timesteps: int = Field(
        1_000_000,
        ge=1,
        description="Total training timesteps",
    )
    seed: int | None = Field(
        None,
        description="Random seed for reproducibility",
    )
    device: Literal["cpu", "cuda", "auto"] = Field(
        "auto",
        description="Device for computation (auto selects cuda if available)",
    )
    normalize_advantage: bool = Field(
        True,
        description="Normalize advantages per minibatch",
    )
    target_kl: float | None = Field(
        None,
        gt=0,
        description="Target KL divergence for early stopping (None = disabled)",
    )


class LoggingConfig(BaseModel):
    """Logging and checkpointing configuration.

    These parameters control how training progress is logged
    and when model checkpoints are saved.
    """

    model_config = ConfigDict(frozen=True)

    log_interval: int = Field(
        10,
        ge=1,
        description="Frequency of logging (in updates)",
    )
    save_interval: int = Field(
        50,
        ge=1,
        description="Frequency of model checkpoints (in updates)",
    )
    tensorboard: bool = Field(
        True,
        description="Enable TensorBoard logging",
    )
    log_dir: str = Field(
        "logs/",
        description="Directory for logs and checkpoints",
    )


class PPOConfig(BaseModel):
    """Complete PPO configuration.

    This is the main configuration class that combines all PPO-related
    settings into a single, validated configuration object.

    The configuration can be:
    - Instantiated with defaults: `PPOConfig()`
    - Loaded from YAML: `PPOConfig.from_yaml("config.yaml")`
    - Saved to YAML: `config.to_yaml("config.yaml")`

    All nested configurations are frozen (immutable) for thread safety.
    """

    model_config = ConfigDict(frozen=True)

    core: PPOCoreConfig = Field(
        default_factory=PPOCoreConfig,
        description="Core PPO algorithm hyperparameters",
    )
    network: NetworkConfig = Field(
        default_factory=NetworkConfig,
        description="Neural network architecture configuration",
    )
    training: TrainingConfig = Field(
        default_factory=TrainingConfig,
        description="Training loop configuration",
    )
    logging: LoggingConfig = Field(
        default_factory=LoggingConfig,
        description="Logging and checkpointing configuration",
    )

    @classmethod
    def from_yaml(cls, path: str) -> "PPOConfig":
        """Load configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            A validated PPOConfig instance.

        Raises:
            FileNotFoundError: If the configuration file doesn't exist.
            ValidationError: If the configuration is invalid.
        """
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data or {})

    def to_yaml(self, path: str) -> None:
        """Save configuration to a YAML file.

        Args:
            path: Path where the YAML file will be saved.
        """
        with open(path, "w") as f:
            yaml.dump(
                self.model_dump(),
                f,
                default_flow_style=False,
                sort_keys=False,
            )
