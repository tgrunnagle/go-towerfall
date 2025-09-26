"""
Training module for RL bot system.

This module provides training session management, batch episode execution,
and APIs for managing accelerated training environments.
"""

from .model_manager import (
    ModelManager,
    RLModel
)

__all__ = [
    # Model Manager
    'ModelManager',
    'RLModel'
]