"""Game constants matching Go server values.

This module contains physics and game constants that match the Go server implementation.
These constants are useful for bot logic, observation normalization, and simulation.
"""

from pydantic import BaseModel


class GameConstants(BaseModel):
    """Physics and game constants matching Go server values.

    All values are class-level defaults that match the server implementation.
    This model can be instantiated if custom values are needed for testing.
    """

    # Physics constants
    GRAVITY_METERS_PER_SEC2: float = 20.0
    MAX_VELOCITY_METERS_PER_SEC: float = 30.0
    PX_PER_METER: float = 20.0

    # Room constants
    ROOM_SIZE_METERS_X: float = 40.0
    ROOM_SIZE_METERS_Y: float = 40.0
    ROOM_SIZE_PIXELS_X: float = 800.0  # ROOM_SIZE_METERS_X * PX_PER_METER
    ROOM_SIZE_PIXELS_Y: float = 800.0  # ROOM_SIZE_METERS_Y * PX_PER_METER
    ROOM_WRAP_DISTANCE_METERS: float = 2.0
    ROOM_WRAP_DISTANCE_PX: float = 40.0  # ROOM_WRAP_DISTANCE_METERS * PX_PER_METER

    # Player constants
    PLAYER_RADIUS: float = 20.0
    PLAYER_SPEED_X_METERS_PER_SEC: float = 15.0
    PLAYER_JUMP_SPEED_METERS_PER_SEC: float = 20.0
    PLAYER_STARTING_HEALTH: int = 100
    PLAYER_MAX_JUMPS: int = 2
    PLAYER_RESPAWN_TIME_SEC: float = 5.0
    PLAYER_MASS_KG: float = 50.0
    PLAYER_STARTING_ARROWS: int = 4
    PLAYER_MAX_ARROWS: int = 4

    # Bullet constants
    BULLET_DISTANCE_PX: float = 1024.0
    BULLET_LIFETIME_SEC: float = 0.1

    # Arrow constants
    ARROW_MAX_POWER_NEWTON: float = 100.0
    ARROW_MAX_POWER_TIME_SEC: float = 2.0
    ARROW_MASS_KG: float = 0.1
    ARROW_LENGTH_METERS: float = 1.0
    ARROW_LENGTH_PX: float = 20.0  # ARROW_LENGTH_METERS * PX_PER_METER
    ARROW_DESTROY_DISTANCE_METERS: float = 5.0
    ARROW_DESTROY_DISTANCE_PX: float = (
        100.0  # ARROW_DESTROY_DISTANCE_METERS * PX_PER_METER
    )
    ARROW_GROUNDED_RADIUS_METERS: float = 0.5
    ARROW_GROUNDED_RADIUS_PX: float = (
        10.0  # ARROW_GROUNDED_RADIUS_METERS * PX_PER_METER
    )

    # Block constants
    BLOCK_SIZE_UNIT_METERS: float = 1.0
    BLOCK_SIZE_UNIT_PIXELS: float = 20.0  # BLOCK_SIZE_UNIT_METERS * PX_PER_METER


# Module-level instance for convenient access
GAME_CONSTANTS = GameConstants()


# Object type constants
class ObjectTypes:
    """Object type string constants matching Go server."""

    PLAYER = "player"
    BULLET = "bullet"
    BLOCK = "block"
    ARROW = "arrow"


# State key constants (short names sent by server)
class StateKeys:
    """State key constants matching Go server short names."""

    ID = "id"
    NAME = "name"
    X = "x"
    Y = "y"
    WIDTH = "w"
    HEIGHT = "h"
    DX = "dx"
    DY = "dy"
    LAST_LOC_UPDATE_TIME = "llut"
    DIRECTION = "dir"
    RADIUS = "rad"
    POINTS = "pts"
    HEALTH = "h"
    DESTROYED = "d"
    DESTROYED_AT_X = "dAtX"
    DESTROYED_AT_Y = "dAtY"
    DEAD = "dead"
    ARROW_GROUNDED = "ag"
    SHOOTING = "sht"
    SHOOTING_START_TIME = "shts"
    JUMP_COUNT = "jc"
    ARROW_COUNT = "ac"
