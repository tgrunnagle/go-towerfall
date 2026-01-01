"""Observation space module for RL training.

This module provides tools for converting game state into normalized observation
vectors suitable for neural network training.
"""

from bot.observation.map_encoder import (
    MapEncoder,
    MapEncodingConfig,
)
from bot.observation.normalizer import (
    NormalizationConstants,
    normalize_angle,
    normalize_boolean,
    normalize_position,
    normalize_velocity,
)
from bot.observation.observation_space import (
    ObservationBuilder,
    ObservationConfig,
)

__all__ = [
    "MapEncoder",
    "MapEncodingConfig",
    "NormalizationConstants",
    "ObservationBuilder",
    "ObservationConfig",
    "normalize_angle",
    "normalize_boolean",
    "normalize_position",
    "normalize_velocity",
]
