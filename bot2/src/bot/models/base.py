"""Base models for game state representation.

This module contains foundational Pydantic models used across the game state system.
"""

from pydantic import BaseModel, ConfigDict, Field


class Point(BaseModel):
    """Represents a 2D coordinate point."""

    model_config = ConfigDict(populate_by_name=True)

    x: float
    y: float


class BaseObjectState(BaseModel):
    """Base model for all game object states.

    Maps to Go `BaseGameObject.GetState()`.
    """

    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(alias="id")
    object_type: str = Field(alias="objectType")
