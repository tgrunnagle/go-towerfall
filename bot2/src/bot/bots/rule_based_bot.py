"""Rule-based bot implementation for TowerFall."""

import math
import time
from dataclasses import dataclass, field

from bot.bots.base_bot import BaseBot
from bot.bots.shooting_utils import (
    ShootingConfig,
    calculate_aim_point,
    calculate_arrow_speed,
    compensate_for_gravity,
    should_release_shot,
    should_shoot,
)
from bot.client import GameClient
from bot.models import GAME_CONSTANTS, GameState, PlayerState


@dataclass
class RuleBasedBotConfig:
    """Configuration for rule-based bot behavior."""

    # Edge avoidance
    edge_margin: float = 50.0  # Distance from edge to start avoiding (pixels)

    # Movement
    dead_zone: float = 20.0  # Horizontal distance below which bot stops moving (pixels)

    # Jumping
    vertical_jump_threshold: float = 40.0  # Height difference to trigger jump (pixels)
    stuck_velocity_threshold: float = (
        0.5  # Velocity below which bot is stuck (pixels/tick)
    )
    stuck_distance_threshold: float = (
        50.0  # Distance to target to consider stuck (pixels)
    )

    # Center position for idling
    center_x: float = GAME_CONSTANTS.ROOM_SIZE_PIXELS_X / 2
    center_dead_zone: float = 50.0  # Dead zone when moving to center

    # Shooting configuration
    shooting: ShootingConfig = field(default_factory=ShootingConfig)


class RuleBasedBot(BaseBot):
    """Rule-based bot that moves toward the nearest enemy while avoiding edges.

    This bot implements simple heuristic-based movement and shooting:
    - Moves horizontally toward the nearest alive enemy
    - Avoids map edges to prevent falling off
    - Jumps when the enemy is above or when stuck
    - Idles at center when no enemies are present
    - Aims at enemies and shoots arrows with appropriate timing and power
    """

    def __init__(
        self, player_id: str, config: RuleBasedBotConfig | None = None
    ) -> None:
        """Initialize the rule-based bot.

        Args:
            player_id: The unique identifier for this bot's player.
            config: Optional configuration for tuning bot behavior.
        """
        super().__init__(player_id)
        self.config = config or RuleBasedBotConfig()

        # Shooting state
        self._shooting_start_time: float | None = None
        self._last_shot_time: float = 0.0

    def _find_nearest_enemy(self) -> PlayerState | None:
        """Find the nearest alive enemy player.

        Returns:
            The PlayerState of the nearest enemy, or None if no enemies exist.
        """
        own_player = self.get_own_player()
        if own_player is None:
            return None

        enemies = self.get_enemies()
        if not enemies:
            return None

        def distance_to(enemy: PlayerState) -> float:
            dx = enemy.x - own_player.x
            dy = enemy.y - own_player.y
            return math.hypot(dx, dy)

        return min(enemies, key=distance_to)

    def _decide_horizontal_movement(self, target_x: float) -> str | None:
        """Decide horizontal movement direction toward a target x position.

        Args:
            target_x: The x coordinate to move toward.

        Returns:
            "a" for left, "d" for right, or None if within dead zone.
        """
        own_player = self.get_own_player()
        if own_player is None:
            return None

        dx = target_x - own_player.x

        if dx > self.config.dead_zone:
            return "d"  # Move right
        elif dx < -self.config.dead_zone:
            return "a"  # Move left
        return None

    def _is_near_left_edge(self) -> bool:
        """Check if bot is near left map edge.

        Returns:
            True if the bot is within edge_margin of the left edge.
        """
        own_player = self.get_own_player()
        if own_player is None:
            return False
        return own_player.x < self.config.edge_margin

    def _is_near_right_edge(self) -> bool:
        """Check if bot is near right map edge.

        Returns:
            True if the bot is within edge_margin of the right edge.
        """
        own_player = self.get_own_player()
        if own_player is None:
            return False
        if self.current_state is None:
            return False
        return own_player.x > (
            self.current_state.canvas_size_x - self.config.edge_margin
        )

    def _apply_edge_avoidance(self, desired_direction: str | None) -> str | None:
        """Override movement direction to avoid edges.

        Args:
            desired_direction: The originally desired movement direction.

        Returns:
            The adjusted movement direction after applying edge avoidance.
        """
        if self._is_near_left_edge():
            # Do not allow moving left when near left edge
            if desired_direction == "a":
                return "d"  # Move away from edge instead

        if self._is_near_right_edge():
            # Do not allow moving right when near right edge
            if desired_direction == "d":
                return "a"  # Move away from edge instead

        return desired_direction

    def _should_jump(self, target: PlayerState) -> bool:
        """Decide if the bot should jump.

        Args:
            target: The target enemy player.

        Returns:
            True if the bot should jump.
        """
        own_player = self.get_own_player()
        if own_player is None:
            return False

        # Jump if enemy is significantly above us
        # Note: In TowerFall, lower y values are higher on screen
        if target.y < own_player.y - self.config.vertical_jump_threshold:
            # Enemy is above, consider jumping
            # Only jump if we have jumps available
            if own_player.jump_count < GAME_CONSTANTS.PLAYER_MAX_JUMPS:
                return True

        # Jump if we are stuck (velocity near zero but not at target x)
        dx = abs(target.x - own_player.x)
        if (
            dx > self.config.stuck_distance_threshold
            and abs(own_player.dx) < self.config.stuck_velocity_threshold
        ):
            return True

        return False

    def _move_to_center(self) -> list[tuple[str, bool]]:
        """Generate actions to move toward the center of the map.

        Returns:
            List of (key, is_pressed) tuples to move toward center.
        """
        actions: list[tuple[str, bool]] = []

        own_player = self.get_own_player()
        if own_player is None:
            return [("w", False), ("a", False), ("d", False)]

        dx = self.config.center_x - own_player.x

        if dx > self.config.center_dead_zone:
            actions.append(("d", True))
            actions.append(("a", False))
        elif dx < -self.config.center_dead_zone:
            actions.append(("a", True))
            actions.append(("d", False))
        else:
            actions.append(("a", False))
            actions.append(("d", False))

        actions.append(("w", False))
        return actions

    async def decide_actions(
        self,
    ) -> list[tuple[str, bool] | tuple[str, bool, float, float]]:
        """Decide movement and shooting actions based on current state.

        Returns:
            List of actions, each is either:
            - (key, is_pressed) for keyboard inputs
            - ("mouse_left", is_pressed, aim_x, aim_y) for mouse inputs
        """
        actions: list[tuple[str, bool] | tuple[str, bool, float, float]] = []

        own_player = self.get_own_player()
        if own_player is None or own_player.dead:
            # Release all keys and shooting if dead or not spawned
            return self._release_all_controls()

        target = self._find_nearest_enemy()

        if target is None:
            # No enemies - idle at center and stop shooting
            movement_actions = self._move_to_center()
            if self._shooting_start_time is not None:
                # Cancel any ongoing shot
                movement_actions.append(("mouse_left", False, 0.0, 0.0))
                self._shooting_start_time = None
            return movement_actions

        # Horizontal movement toward enemy
        horizontal = self._decide_horizontal_movement(target.x)
        horizontal = self._apply_edge_avoidance(horizontal)

        # Apply horizontal movement
        if horizontal == "a":
            actions.append(("a", True))
            actions.append(("d", False))
        elif horizontal == "d":
            actions.append(("d", True))
            actions.append(("a", False))
        else:
            actions.append(("a", False))
            actions.append(("d", False))

        # Vertical movement (jumping)
        if self._should_jump(target):
            actions.append(("w", True))
        else:
            actions.append(("w", False))

        # Shooting actions
        shooting_actions = self._decide_shooting_actions(own_player, target)
        actions.extend(shooting_actions)

        return actions

    def _decide_shooting_actions(
        self, own_player: PlayerState, target: PlayerState
    ) -> list[tuple[str, bool, float, float]]:
        """Decide shooting-related actions.

        Args:
            own_player: Bot current state.
            target: Target player state.

        Returns:
            List of mouse actions as ("mouse_left", pressed, aim_x, aim_y) tuples.
        """
        actions: list[tuple[str, bool, float, float]] = []
        shooting_config = self.config.shooting

        current_time = time.time()

        # Check shooting cooldown
        if current_time - self._last_shot_time < shooting_config.shot_cooldown_sec:
            return actions

        # Calculate aim point with medium power estimate
        arrow_speed = calculate_arrow_speed(0.5)

        if shooting_config.use_lead_prediction:
            aim_x, aim_y = calculate_aim_point(
                own_player.x,
                own_player.y,
                target.x,
                target.y,
                target.dx,
                target.dy,
                arrow_speed,
            )
        else:
            aim_x, aim_y = target.x, target.y

        if shooting_config.use_gravity_compensation:
            aim_x, aim_y = compensate_for_gravity(
                own_player.x,
                own_player.y,
                aim_x,
                aim_y,
                arrow_speed,
            )

        # Currently shooting (charging)?
        if self._shooting_start_time is not None:
            # Check if we should release
            if should_release_shot(
                own_player,
                target,
                self._shooting_start_time,
                current_time,
                shooting_config,
            ):
                # Release shot at aim point
                actions.append(("mouse_left", False, aim_x, aim_y))
                self._shooting_start_time = None
                self._last_shot_time = current_time
            # Continue charging (no action needed, maintain current aim)
        else:
            # Not shooting - should we start?
            if should_shoot(own_player, target, shooting_config):
                # Start charging
                actions.append(("mouse_left", True, aim_x, aim_y))
                self._shooting_start_time = current_time

        return actions

    def _release_all_controls(
        self,
    ) -> list[tuple[str, bool] | tuple[str, bool, float, float]]:
        """Release all movement and shooting controls.

        Returns:
            List of actions to release all controls.
        """
        actions: list[tuple[str, bool] | tuple[str, bool, float, float]] = [
            ("w", False),
            ("a", False),
            ("d", False),
        ]
        if self._shooting_start_time is not None:
            actions.append(("mouse_left", False, 0.0, 0.0))
            self._shooting_start_time = None
        return actions


class RuleBasedBotRunner:
    """Runs a rule-based bot connected to a game server.

    This class handles the integration between the RuleBasedBot and the GameClient,
    managing state updates and sending only changed key presses and mouse inputs.
    """

    def __init__(
        self,
        client: GameClient,
        config: RuleBasedBotConfig | None = None,
    ) -> None:
        """Initialize the bot runner.

        Args:
            client: The game client for server communication.
            config: Optional configuration for the rule-based bot.
        """
        self.client = client
        self.config = config
        self.bot: RuleBasedBot | None = None
        self._previous_keyboard_actions: dict[str, bool] = {}
        self._previous_mouse_state: bool = False

    async def on_game_state(self, state: GameState) -> None:
        """Handle incoming game state updates.

        This method should be called whenever a new game state is received.
        It updates the bot's state and sends any necessary key presses and mouse inputs.

        Args:
            state: The current game state from the server.

        Raises:
            ValueError: If the client's player_id is not set.
        """
        if self.bot is None:
            if self.client.player_id is None:
                raise ValueError(
                    "Client player_id must be set before calling on_game_state"
                )
            self.bot = RuleBasedBot(self.client.player_id, self.config)

        self.bot.update_state(state)
        actions = await self.bot.decide_actions()

        for action in actions:
            if action[0] == "mouse_left":
                # Mouse action: ("mouse_left", pressed, aim_x, aim_y)
                _, pressed, aim_x, aim_y = action
                if self._previous_mouse_state != pressed:
                    await self.client.send_mouse_input("left", pressed, aim_x, aim_y)
                    self._previous_mouse_state = pressed
            else:
                # Keyboard action: (key, pressed)
                key, pressed = action[0], action[1]
                if self._previous_keyboard_actions.get(key) != pressed:
                    await self.client.send_keyboard_input(key, pressed)
                    self._previous_keyboard_actions[key] = pressed

    def reset(self) -> None:
        """Reset the bot runner state for a new game/episode."""
        self._previous_keyboard_actions = {}
        self._previous_mouse_state = False
        if self.bot is not None:
            self.bot._shooting_start_time = None
            self.bot._last_shot_time = 0.0
