"""Discrete action space for ML bot training.

This module defines the discrete action space for RL agents, following the requirement
that key press and release are treated as separate actions. The action space maps
agent decisions to game inputs via the GameClient.

Action Space (27 discrete actions):
- Movement (8 actions): Press/release for A, D, W, S keys
- Aim Direction (16 actions): 16 buckets covering 0 to 2π radians
- Shooting (2 actions): Mouse button press/release
- No-Op (1 action): Do nothing

Usage:
    from bot.actions import Action, ACTION_SPACE_SIZE, execute_action

    # Get action space size for Gymnasium
    action_space = gym.spaces.Discrete(ACTION_SPACE_SIZE)

    # Execute an action
    await execute_action(client, Action.MOVE_RIGHT_PRESS)
"""

from __future__ import annotations

import math
from enum import IntEnum

from bot.client import GameClient


class Action(IntEnum):
    """Discrete action enumeration for RL agents.

    Actions are organized into categories:
    - Movement (0-7): Key press/release for W/A/S/D
    - Aim Direction (8-23): 16 directional buckets
    - Shooting (24-25): Mouse button press/release
    - No-Op (26): Do nothing
    """

    # Movement actions (8 actions)
    # Left/Right movement
    MOVE_LEFT_PRESS = 0
    MOVE_LEFT_RELEASE = 1
    MOVE_RIGHT_PRESS = 2
    MOVE_RIGHT_RELEASE = 3

    # Jump/Dive
    JUMP_PRESS = 4
    JUMP_RELEASE = 5
    DIVE_PRESS = 6
    DIVE_RELEASE = 7

    # Aim direction actions (16 actions)
    # Each bucket is π/8 radians (22.5 degrees)
    AIM_0 = 8  # 0 radians - Right
    AIM_1 = 9  # π/8 radians
    AIM_2 = 10  # π/4 radians - Right-down (45°)
    AIM_3 = 11  # 3π/8 radians
    AIM_4 = 12  # π/2 radians - Down (90°)
    AIM_5 = 13  # 5π/8 radians
    AIM_6 = 14  # 3π/4 radians - Left-down (135°)
    AIM_7 = 15  # 7π/8 radians
    AIM_8 = 16  # π radians - Left (180°)
    AIM_9 = 17  # 9π/8 radians
    AIM_10 = 18  # 5π/4 radians - Left-up (225°)
    AIM_11 = 19  # 11π/8 radians
    AIM_12 = 20  # 3π/2 radians - Up (270°)
    AIM_13 = 21  # 13π/8 radians
    AIM_14 = 22  # 7π/4 radians - Right-up (315°)
    AIM_15 = 23  # 15π/8 radians

    # Shooting actions (2 actions)
    SHOOT_START = 24  # Start drawing bow
    SHOOT_RELEASE = 25  # Release arrow

    # No-op action (1 action)
    NO_OP = 26


# Total action space size for Gymnasium
ACTION_SPACE_SIZE: int = 27

# Number of aim buckets
NUM_AIM_BUCKETS: int = 16

# Radians per aim bucket
RADIANS_PER_AIM_BUCKET: float = (2 * math.pi) / NUM_AIM_BUCKETS  # π/8


def aim_action_to_radians(action: Action | int) -> float:
    """Convert an aim action to radians.

    Args:
        action: An aim action (AIM_0 through AIM_15) or its integer value.

    Returns:
        The direction in radians (0 to 2π).

    Raises:
        ValueError: If the action is not an aim action.

    Example:
        >>> aim_action_to_radians(Action.AIM_0)
        0.0
        >>> aim_action_to_radians(Action.AIM_4)
        1.5707963267948966  # π/2
        >>> aim_action_to_radians(Action.AIM_8)
        3.141592653589793   # π
    """
    action_value = int(action)

    if action_value < Action.AIM_0 or action_value > Action.AIM_15:
        raise ValueError(
            f"Action {action} is not an aim action. "
            f"Expected action in range [{Action.AIM_0}, {Action.AIM_15}]."
        )

    bucket = action_value - Action.AIM_0
    return bucket * RADIANS_PER_AIM_BUCKET


def radians_to_aim_action(radians: float) -> Action:
    """Convert radians to the nearest aim action.

    Args:
        radians: Direction in radians (any value, will be normalized to 0-2π).

    Returns:
        The closest aim action for the given direction.

    Example:
        >>> radians_to_aim_action(0.0)
        Action.AIM_0
        >>> radians_to_aim_action(math.pi / 2)
        Action.AIM_4
        >>> radians_to_aim_action(math.pi)
        Action.AIM_8
    """
    # Normalize to [0, 2π)
    normalized = radians % (2 * math.pi)

    # Find the closest bucket
    bucket = round(normalized / RADIANS_PER_AIM_BUCKET) % NUM_AIM_BUCKETS

    return Action(Action.AIM_0 + bucket)


def is_aim_action(action: Action | int) -> bool:
    """Check if an action is an aim action.

    Args:
        action: The action to check.

    Returns:
        True if the action is an aim action (AIM_0 through AIM_15).
    """
    action_value = int(action)
    return Action.AIM_0 <= action_value <= Action.AIM_15


def is_movement_action(action: Action | int) -> bool:
    """Check if an action is a movement action.

    Args:
        action: The action to check.

    Returns:
        True if the action is a movement action (MOVE_*, JUMP_*, DIVE_*).
    """
    action_value = int(action)
    return Action.MOVE_LEFT_PRESS <= action_value <= Action.DIVE_RELEASE


def is_shoot_action(action: Action | int) -> bool:
    """Check if an action is a shooting action.

    Args:
        action: The action to check.

    Returns:
        True if the action is a shooting action (SHOOT_START or SHOOT_RELEASE).
    """
    action_value = int(action)
    return action_value in (Action.SHOOT_START, Action.SHOOT_RELEASE)


# Key mappings for movement actions
_MOVEMENT_KEY_MAP: dict[Action, tuple[str, bool]] = {
    Action.MOVE_LEFT_PRESS: ("A", True),
    Action.MOVE_LEFT_RELEASE: ("A", False),
    Action.MOVE_RIGHT_PRESS: ("D", True),
    Action.MOVE_RIGHT_RELEASE: ("D", False),
    Action.JUMP_PRESS: ("W", True),
    Action.JUMP_RELEASE: ("W", False),
    Action.DIVE_PRESS: ("S", True),
    Action.DIVE_RELEASE: ("S", False),
}


async def execute_action(client: GameClient, action: Action | int) -> None:
    """Execute a discrete action through the game client.

    This function translates discrete actions into the appropriate GameClient
    method calls (keyboard input, mouse input, or direction updates).

    Args:
        client: The GameClient instance to send actions through.
        action: The discrete action to execute.

    Raises:
        ValueError: If the action is not a valid Action value.

    Example:
        async with GameClient(mode=ClientMode.REST) as client:
            await client.create_game(...)
            await execute_action(client, Action.MOVE_RIGHT_PRESS)
            await execute_action(client, Action.AIM_4)  # Aim down
            await execute_action(client, Action.SHOOT_START)
    """
    action = Action(action)  # Ensure it's an Action enum

    # No-op: do nothing
    if action == Action.NO_OP:
        return

    # Movement actions: send keyboard input
    if action in _MOVEMENT_KEY_MAP:
        key, pressed = _MOVEMENT_KEY_MAP[action]
        await client.send_keyboard_input(key, pressed)
        return

    # Aim actions: send direction update
    if is_aim_action(action):
        radians = aim_action_to_radians(action)
        await _send_direction(client, radians)
        return

    # Shoot actions: send mouse input
    if action == Action.SHOOT_START:
        # Use center of canvas for click position (direction is controlled by aim)
        await client.send_mouse_input("left", True, 0.0, 0.0)
        return

    if action == Action.SHOOT_RELEASE:
        await client.send_mouse_input("left", False, 0.0, 0.0)
        return


async def _send_direction(client: GameClient, direction: float) -> None:
    """Send a direction update to the game server.

    Args:
        client: The GameClient instance.
        direction: Direction in radians.
    """
    await client.send_direction(direction)


def action_to_string(action: Action | int) -> str:
    """Get a human-readable string for an action.

    Args:
        action: The action to describe.

    Returns:
        A human-readable string describing the action.
    """
    action = Action(action)
    return action.name


def get_action_category(action: Action | int) -> str:
    """Get the category of an action.

    Args:
        action: The action to categorize.

    Returns:
        One of: "movement", "aim", "shoot", or "noop".
    """
    if is_movement_action(action):
        return "movement"
    if is_aim_action(action):
        return "aim"
    if is_shoot_action(action):
        return "shoot"
    return "noop"
