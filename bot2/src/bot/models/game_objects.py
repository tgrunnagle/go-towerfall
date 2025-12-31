"""Game object state models.

This module contains Pydantic models for individual game objects:
- PlayerState: Player character state
- ArrowState: Arrow projectile state
- BlockState: Static block/platform state
- BulletState: Bullet projectile state
"""

from pydantic import ConfigDict, Field

from bot.models.base import BaseObjectState, Point


class PlayerState(BaseObjectState):
    """Represents player game object state.

    Maps to Go `PlayerGameObject`.
    """

    model_config = ConfigDict(populate_by_name=True)

    name: str = Field(alias="name")
    x: float = Field(alias="x")
    y: float = Field(alias="y")
    dx: float = Field(alias="dx")  # X velocity
    dy: float = Field(alias="dy")  # Y velocity
    direction: float = Field(alias="dir")  # Direction in radians (0=right)
    radius: float = Field(alias="rad")
    health: int = Field(alias="h")
    dead: bool = Field(alias="dead")
    shooting: bool = Field(alias="sht")
    shooting_start_time: float | None = Field(default=None, alias="shts")
    jump_count: int = Field(alias="jc")
    arrow_count: int = Field(alias="ac")


class ArrowState(BaseObjectState):
    """Represents arrow projectile state.

    Maps to Go `ArrowGameObject`.
    """

    model_config = ConfigDict(populate_by_name=True)

    x: float = Field(alias="x")
    y: float = Field(alias="y")
    dx: float = Field(alias="dx")
    dy: float = Field(alias="dy")
    direction: float | None = Field(default=None, alias="dir")
    grounded: bool = Field(alias="ag")
    destroyed: bool = Field(default=False, alias="d")
    destroyed_at_x: float | None = Field(default=None, alias="dAtX")
    destroyed_at_y: float | None = Field(default=None, alias="dAtY")


class BlockState(BaseObjectState):
    """Represents static block/platform.

    Maps to Go `BlockGameObject`.
    """

    model_config = ConfigDict(populate_by_name=True)

    points: list[Point] = Field(alias="pts")


class BulletState(BaseObjectState):
    """Represents bullet projectile state.

    Maps to Go `BulletGameObject`.
    """

    model_config = ConfigDict(populate_by_name=True)

    x: float = Field(alias="x")
    y: float = Field(alias="y")
    dx: float = Field(alias="dx")  # Normalized X direction
    dy: float = Field(alias="dy")  # Normalized Y direction
    destroyed: bool = Field(default=False, alias="d")
