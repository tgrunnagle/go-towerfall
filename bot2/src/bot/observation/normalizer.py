"""Normalization utilities for observation space.

All values are normalized to the [-1, 1] range for neural network compatibility.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class NormalizationConstants:
    """Constants used for normalizing game state values.

    These values are derived from the game constants in
    backend/pkg/server/constants/constants.go.
    """

    # Room dimensions in pixels (40m * 20px/m = 800px)
    ROOM_WIDTH: float = 800.0
    ROOM_HEIGHT: float = 800.0

    # Player max velocity in px/s (30 m/s * 20 px/m = 600 px/s)
    MAX_PLAYER_VELOCITY: float = 600.0

    # Arrow max velocity in px/s (arrows can travel faster than players)
    MAX_ARROW_VELOCITY: float = 1000.0

    # Player constants
    MAX_HEALTH: float = 100.0
    MAX_ARROWS: int = 4
    MAX_JUMPS: int = 2

    # Arrow constants
    MAX_ARROW_POWER_TIME: float = 2.0  # seconds to reach max power


# Default constants instance for convenience
NORMALIZATION_CONSTANTS = NormalizationConstants()


def normalize_position(
    x: float,
    y: float,
    room_width: float = NORMALIZATION_CONSTANTS.ROOM_WIDTH,
    room_height: float = NORMALIZATION_CONSTANTS.ROOM_HEIGHT,
) -> tuple[float, float]:
    """Normalize position to [-1, 1] range.

    Args:
        x: X position in pixels [0, room_width]
        y: Y position in pixels [0, room_height]
        room_width: Room width in pixels
        room_height: Room height in pixels

    Returns:
        Tuple of (normalized_x, normalized_y) in [-1, 1] range
    """
    norm_x = (x / (room_width / 2)) - 1.0
    norm_y = (y / (room_height / 2)) - 1.0
    return (
        max(-1.0, min(1.0, norm_x)),
        max(-1.0, min(1.0, norm_y)),
    )


def normalize_relative_position(
    rel_x: float,
    rel_y: float,
    max_distance: float = NORMALIZATION_CONSTANTS.ROOM_WIDTH,
) -> tuple[float, float]:
    """Normalize relative position to [-1, 1] range.

    Args:
        rel_x: Relative X position in pixels
        rel_y: Relative Y position in pixels
        max_distance: Maximum distance for normalization

    Returns:
        Tuple of (normalized_rel_x, normalized_rel_y) in [-1, 1] range
    """
    norm_x = rel_x / max_distance
    norm_y = rel_y / max_distance
    return (
        max(-1.0, min(1.0, norm_x)),
        max(-1.0, min(1.0, norm_y)),
    )


def normalize_velocity(
    dx: float,
    dy: float,
    max_velocity: float = NORMALIZATION_CONSTANTS.MAX_PLAYER_VELOCITY,
) -> tuple[float, float]:
    """Normalize velocity to [-1, 1] range.

    Args:
        dx: X velocity in pixels/second
        dy: Y velocity in pixels/second
        max_velocity: Maximum velocity for normalization

    Returns:
        Tuple of (normalized_dx, normalized_dy) in [-1, 1] range
    """
    norm_dx = dx / max_velocity
    norm_dy = dy / max_velocity
    return (
        max(-1.0, min(1.0, norm_dx)),
        max(-1.0, min(1.0, norm_dy)),
    )


def normalize_angle(angle: float) -> float:
    """Normalize angle from [0, 2π] to [-1, 1] range.

    Args:
        angle: Angle in radians [0, 2π]

    Returns:
        Normalized angle in [-1, 1] range
    """
    # Normalize to [0, 2π] first to handle any out-of-range values
    normalized_angle = angle % (2 * math.pi)
    # Convert to [-1, 1]: (angle / π) - 1
    return (normalized_angle / math.pi) - 1.0


def normalize_boolean(value: bool) -> float:
    """Convert boolean to [-1, 1] range.

    Args:
        value: Boolean value

    Returns:
        -1.0 for False, 1.0 for True
    """
    return 1.0 if value else -1.0


def normalize_distance(
    distance: float,
    max_distance: float = NORMALIZATION_CONSTANTS.ROOM_WIDTH,
) -> float:
    """Normalize distance to [0, 1] range (always positive).

    Args:
        distance: Distance in pixels
        max_distance: Maximum distance for normalization

    Returns:
        Normalized distance in [0, 1] range
    """
    return min(1.0, abs(distance) / max_distance)


def normalize_count(
    count: int,
    max_count: int,
) -> float:
    """Normalize a count to [-1, 1] range.

    Args:
        count: Current count
        max_count: Maximum possible count

    Returns:
        Normalized count in [-1, 1] range
    """
    if max_count == 0:
        return 0.0
    return (count / (max_count / 2)) - 1.0


def normalize_health(
    health: float,
    max_health: float = NORMALIZATION_CONSTANTS.MAX_HEALTH,
) -> float:
    """Normalize health to [-1, 1] range.

    Args:
        health: Current health
        max_health: Maximum health

    Returns:
        Normalized health in [-1, 1] range
    """
    return (health / (max_health / 2)) - 1.0


def calculate_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Calculate Euclidean distance between two points.

    Args:
        x1: X coordinate of first point
        y1: Y coordinate of first point
        x2: X coordinate of second point
        y2: Y coordinate of second point

    Returns:
        Euclidean distance between the points
    """
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def calculate_angle_to(x1: float, y1: float, x2: float, y2: float) -> float:
    """Calculate angle from point 1 to point 2.

    Args:
        x1: X coordinate of first point
        y1: Y coordinate of first point
        x2: X coordinate of second point
        y2: Y coordinate of second point

    Returns:
        Angle in radians [0, 2π]
    """
    angle = math.atan2(y2 - y1, x2 - x1)
    # Convert from [-π, π] to [0, 2π]
    if angle < 0:
        angle += 2 * math.pi
    return angle
