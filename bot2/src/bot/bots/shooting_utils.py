"""Shooting utilities for rule-based bot aiming and shooting.

This module provides ballistic calculations and aiming logic for the rule-based bot:
- Target lead prediction for moving targets
- Gravity compensation for arrow drop
- Arrow speed and power calculations
- Shooting decision logic
"""

import math
from dataclasses import dataclass

from bot.models import GAME_CONSTANTS, PlayerState

# Derived constants for ballistic calculations
GRAVITY_PX_PER_SEC2 = (
    GAME_CONSTANTS.GRAVITY_METERS_PER_SEC2 * GAME_CONSTANTS.PX_PER_METER
)  # 400 px/s^2


def calculate_max_arrow_speed() -> float:
    """Calculate maximum arrow speed at full power.

    Returns:
        Maximum arrow speed in pixels per second.
    """
    return (
        math.sqrt(
            2
            * 1.0
            * GAME_CONSTANTS.ARROW_MAX_POWER_NEWTON
            / GAME_CONSTANTS.ARROW_MASS_KG
        )
        * GAME_CONSTANTS.PX_PER_METER
    )


def calculate_arrow_speed(power_ratio: float) -> float:
    """Calculate arrow speed for a given power ratio.

    Args:
        power_ratio: Power level from 0.0 to 1.0.

    Returns:
        Arrow speed in pixels per second.
    """
    power_ratio = max(0.0, min(1.0, power_ratio))
    return (
        math.sqrt(
            2
            * power_ratio
            * GAME_CONSTANTS.ARROW_MAX_POWER_NEWTON
            / GAME_CONSTANTS.ARROW_MASS_KG
        )
        * GAME_CONSTANTS.PX_PER_METER
    )


def calculate_aim_point(
    own_x: float,
    own_y: float,
    target_x: float,
    target_y: float,
    target_dx: float,
    target_dy: float,
    arrow_speed: float,
) -> tuple[float, float]:
    """Calculate the point to aim at to hit a moving target.

    Uses linear prediction: estimates where target will be when arrow arrives.

    Args:
        own_x: Bot current X position.
        own_y: Bot current Y position.
        target_x: Target current X position.
        target_y: Target current Y position.
        target_dx: Target X velocity (pixels per second).
        target_dy: Target Y velocity (pixels per second).
        arrow_speed: Arrow speed in pixels per second.

    Returns:
        Tuple of (aim_x, aim_y) - point to aim at.
    """
    if arrow_speed <= 0:
        return (target_x, target_y)

    # Calculate distance to target
    dx = target_x - own_x
    dy = target_y - own_y
    distance = math.hypot(dx, dy)

    if distance <= 0:
        return (target_x, target_y)

    # Estimate travel time
    travel_time = distance / arrow_speed

    # Predict target position at arrival time
    predicted_x = target_x + target_dx * travel_time
    predicted_y = target_y + target_dy * travel_time

    return (predicted_x, predicted_y)


def compensate_for_gravity(
    own_x: float,
    own_y: float,
    aim_x: float,
    aim_y: float,
    arrow_speed: float,
) -> tuple[float, float]:
    """Adjust aim point upward to compensate for gravity drop.

    Uses simplified ballistic trajectory: y_drop = 0.5 * g * t^2

    Args:
        own_x: Bot current X position.
        own_y: Bot current Y position.
        aim_x: Initial aim X position (from calculate_aim_point).
        aim_y: Initial aim Y position (from calculate_aim_point).
        arrow_speed: Arrow speed in pixels per second.

    Returns:
        Tuple of (adjusted_aim_x, adjusted_aim_y) - gravity-compensated aim point.
    """
    if arrow_speed <= 0:
        return (aim_x, aim_y)

    dx = aim_x - own_x
    dy = aim_y - own_y
    distance = math.hypot(dx, dy)

    if distance <= 0:
        return (aim_x, aim_y)

    # Estimate travel time
    travel_time = distance / arrow_speed

    # Calculate gravity drop during flight
    # In game coordinates, lower Y = higher on screen
    gravity_drop = 0.5 * GRAVITY_PX_PER_SEC2 * travel_time * travel_time

    # Aim higher to compensate (subtract Y since lower Y is higher)
    adjusted_aim_y = aim_y - gravity_drop

    return (aim_x, adjusted_aim_y)


def calculate_optimal_power(distance: float) -> float:
    """Calculate optimal power ratio for a given distance.

    For closer targets, use less power for faster shot.
    For distant targets, use more power for reach.

    Uses default thresholds. For configurable thresholds, use
    calculate_optimal_power_with_config().

    Args:
        distance: Distance to target in pixels.

    Returns:
        Power ratio between 0.0 and 1.0.
    """
    # Default thresholds (kept for backwards compatibility)
    return calculate_optimal_power_with_thresholds(
        distance=distance,
        min_power=0.2,
        close_range=100.0,
        max_range=600.0,
    )


def calculate_optimal_power_with_thresholds(
    distance: float,
    min_power: float,
    close_range: float,
    max_range: float,
) -> float:
    """Calculate optimal power ratio for a given distance with custom thresholds.

    For closer targets, use less power for faster shot.
    For distant targets, use more power for reach.

    Args:
        distance: Distance to target in pixels.
        min_power: Minimum power ratio for close range shots.
        close_range: Distance threshold below which min power is used.
        max_range: Distance threshold above which max power (1.0) is used.

    Returns:
        Power ratio between min_power and 1.0.
    """
    if distance <= close_range:
        return min_power
    elif distance >= max_range:
        return 1.0
    else:
        # Linear interpolation
        ratio = (distance - close_range) / (max_range - close_range)
        return min_power + ratio * (1.0 - min_power)


@dataclass
class ShootingConfig:
    """Configuration for shooting behavior."""

    # Range thresholds (pixels)
    min_shooting_range: float = 50.0  # Too close - don't shoot
    max_shooting_range: float = 500.0  # Too far - don't shoot
    optimal_range: float = 200.0  # Preferred distance

    # Power settings
    min_power_ratio: float = 0.2  # Minimum charge before release
    max_power_ratio: float = 1.0  # Maximum charge

    # Timing
    shot_cooldown_sec: float = 0.5  # Minimum time between shots

    # Aim prediction
    use_lead_prediction: bool = True  # Predict moving target position
    use_gravity_compensation: bool = True  # Compensate for arrow drop

    # Accuracy settings (optional future use)
    aim_noise_factor: float = 0.0  # Add randomness to aim (0 = perfect)

    # Power calculation thresholds (pixels)
    # Used by calculate_optimal_power_with_config()
    close_range_threshold: float = 100.0  # Distance below which min power is used
    max_range_threshold: float = 600.0  # Distance above which max power is used


def should_shoot(
    own_player: PlayerState,
    target: PlayerState,
    config: ShootingConfig,
) -> bool:
    """Decide whether the bot should attempt to shoot.

    Conditions:
    1. Target is within shooting range
    2. Bot has arrows available
    3. Bot is not currently shooting (charging)
    4. Target is alive

    Args:
        own_player: Bot current state.
        target: Target player state.
        config: Shooting configuration.

    Returns:
        True if bot should initiate a shot.
    """
    # Check arrow count
    if own_player.arrow_count <= 0:
        return False

    # Check if already shooting
    if own_player.shooting:
        return False

    # Check if target is alive
    if target.dead:
        return False

    # Calculate distance
    dx = target.x - own_player.x
    dy = target.y - own_player.y
    distance = math.hypot(dx, dy)

    # Check range
    if distance < config.min_shooting_range:
        return False
    if distance > config.max_shooting_range:
        return False

    return True


def should_release_shot(
    own_player: PlayerState,
    target: PlayerState,
    shooting_start_time: float,
    current_time: float,
    config: ShootingConfig,
) -> bool:
    """Decide whether to release a charged shot.

    Release when:
    1. Minimum power achieved
    2. Optimal power reached for distance, OR
    3. Maximum power reached

    Args:
        own_player: Bot current state.
        target: Target player state.
        shooting_start_time: When charging started (seconds).
        current_time: Current time (seconds).
        config: Shooting configuration.

    Returns:
        True if shot should be released.
    """
    # Calculate current power ratio
    charge_time = current_time - shooting_start_time
    power_ratio = min(charge_time / GAME_CONSTANTS.ARROW_MAX_POWER_TIME_SEC, 1.0)

    # Always release at max power
    if power_ratio >= config.max_power_ratio:
        return True

    # Check if minimum power achieved
    if power_ratio < config.min_power_ratio:
        return False

    # Calculate optimal power for distance using config thresholds
    dx = target.x - own_player.x
    dy = target.y - own_player.y
    distance = math.hypot(dx, dy)
    optimal_power = calculate_optimal_power_with_thresholds(
        distance=distance,
        min_power=config.min_power_ratio,
        close_range=config.close_range_threshold,
        max_range=config.max_range_threshold,
    )

    # Release when we reach optimal power
    return power_ratio >= optimal_power
