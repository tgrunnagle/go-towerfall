"""Neural network bot implementation for TowerFall."""

from dataclasses import dataclass

import numpy as np
import torch

from bot.actions import Action, aim_action_to_radians, is_aim_action
from bot.agent.network import ActorCriticNetwork
from bot.bots.base_bot import BaseBot, BotAction
from bot.client import GameClient
from bot.models import GameState
from bot.observation.observation_space import (
    DEFAULT_CONFIG,
    ObservationBuilder,
    ObservationConfig,
)


@dataclass
class NeuralNetBotConfig:
    """Configuration for neural network bot behavior."""

    observation_config: ObservationConfig | None = None
    device: str = "cpu"  # Device for inference ("cpu" or "cuda")


class NeuralNetBot(BaseBot):
    """Neural network bot that uses ActorCriticNetwork for inference.

    Uses a trained ActorCriticNetwork to make decisions based on game state.
    The bot converts game state to observations, runs inference, and translates
    discrete actions to game inputs.
    """

    def __init__(
        self,
        player_id: str,
        network: ActorCriticNetwork,
        config: NeuralNetBotConfig | None = None,
    ) -> None:
        """Initialize the neural network bot.

        Args:
            player_id: Unique identifier for this bot's player
            network: Trained ActorCriticNetwork for inference
            config: Optional configuration for tuning bot behavior
        """
        super().__init__(player_id)
        self.network = network
        self.config = config or NeuralNetBotConfig()

        # Initialize observation builder
        obs_config = self.config.observation_config or DEFAULT_CONFIG
        self.observation_builder = ObservationBuilder(config=obs_config)

        # Set device
        self.device = torch.device(self.config.device)
        self.network.to(self.device)
        self.network.eval()  # Set to evaluation mode

        # Track previous state for action translation
        self._previous_movement_keys: dict[str, bool] = {
            "a": False,
            "d": False,
            "w": False,
            "s": False,
        }
        # Initialize to -1.0 (sentinel value, not a valid aim angle)
        # First actual aim will be sent, and shooting without aim uses 0.0 as default
        self._previous_aim_direction: float = -1.0
        self._previous_shooting: bool = False

    async def decide_actions(self) -> list[BotAction]:
        """Decide actions using neural network inference.

        Returns:
            List of BotAction tuples for keyboard/mouse inputs
        """
        actions: list[BotAction] = []

        own_player = self.get_own_player()
        if own_player is None or own_player.dead or self.current_state is None:
            # Release all keys and shooting if dead or not spawned
            return self._release_all_controls()

        # Build observation from current game state
        observation = self.observation_builder.build(self.current_state, self.player_id)

        # Convert to PyTorch tensor and add batch dimension
        obs_tensor = torch.from_numpy(observation).float().unsqueeze(0).to(self.device)

        # Run inference with deterministic action selection
        with torch.no_grad():
            action_idx, _, _, _ = self.network.get_action_and_value(
                obs_tensor, deterministic=True
            )

        # Extract action as integer
        action = Action(action_idx.item())

        # Translate discrete action to game inputs
        actions = self._translate_action(action, own_player.x, own_player.y)

        return actions

    def _translate_action(
        self, action: Action, player_x: float, player_y: float
    ) -> list[BotAction]:
        """Translate discrete action to BotAction tuples.

        Args:
            action: Discrete action from neural network
            player_x: Player x position (for aim calculation)
            player_y: Player y position (for aim calculation)

        Returns:
            List of BotAction tuples representing the action
        """
        actions: list[BotAction] = []

        # No-op action
        if action == Action.NO_OP:
            return actions

        # Movement actions (A, D, W, S press/release)
        if Action.MOVE_LEFT_PRESS <= action <= Action.DIVE_RELEASE:
            actions.extend(self._handle_movement_action(action))

        # Aim actions
        elif is_aim_action(action):
            actions.extend(self._handle_aim_action(action, player_x, player_y))

        # Shooting actions
        elif action in (Action.SHOOT_START, Action.SHOOT_RELEASE):
            actions.extend(self._handle_shoot_action(action, player_x, player_y))

        return actions

    def _handle_movement_action(self, action: Action) -> list[BotAction]:
        """Handle movement key press/release actions.

        Args:
            action: Movement action (MOVE_*, JUMP_*, DIVE_*)

        Returns:
            List of keyboard actions to press/release movement keys
        """
        actions: list[BotAction] = []

        # Map action to key and pressed state
        key_map: dict[Action, tuple[str, bool]] = {
            Action.MOVE_LEFT_PRESS: ("a", True),
            Action.MOVE_LEFT_RELEASE: ("a", False),
            Action.MOVE_RIGHT_PRESS: ("d", True),
            Action.MOVE_RIGHT_RELEASE: ("d", False),
            Action.JUMP_PRESS: ("w", True),
            Action.JUMP_RELEASE: ("w", False),
            Action.DIVE_PRESS: ("s", True),
            Action.DIVE_RELEASE: ("s", False),
        }

        if action in key_map:
            key, pressed = key_map[action]
            # Only send if state changed
            if self._previous_movement_keys.get(key) != pressed:
                actions.append((key, pressed))  # type: ignore[arg-type]
                self._previous_movement_keys[key] = pressed

        return actions

    def _handle_aim_action(
        self, action: Action, player_x: float, player_y: float
    ) -> list[BotAction]:
        """Handle aim direction actions.

        Args:
            action: Aim action (AIM_0 through AIM_15)
            player_x: Player x position
            player_y: Player y position

        Returns:
            List containing mouse action with aim position
        """
        actions: list[BotAction] = []

        # Convert action to radians
        radians = aim_action_to_radians(action)

        # Only update if direction changed
        if self._previous_aim_direction != radians:
            # Convert radians to aim position (use distance of 100 pixels from player)
            aim_distance = 100.0
            aim_x = player_x + aim_distance * np.cos(radians)
            aim_y = player_y + aim_distance * np.sin(radians)

            # Send as mouse movement (not pressed, just position update)
            # Use special "aim update" by sending mouse position without click
            actions.append(("mouse_left", self._previous_shooting, aim_x, aim_y))
            self._previous_aim_direction = radians

        return actions

    def _handle_shoot_action(
        self, action: Action, player_x: float, player_y: float
    ) -> list[BotAction]:
        """Handle shooting actions.

        Args:
            action: Shoot action (SHOOT_START or SHOOT_RELEASE)
            player_x: Player x position
            player_y: Player y position

        Returns:
            List containing mouse action for shooting
        """
        actions: list[BotAction] = []

        # Calculate aim position from current aim direction
        # Use 0.0 (right) as default if no aim direction set yet
        aim_radians = (
            self._previous_aim_direction if self._previous_aim_direction >= 0 else 0.0
        )
        aim_distance = 100.0
        aim_x = player_x + aim_distance * np.cos(aim_radians)
        aim_y = player_y + aim_distance * np.sin(aim_radians)

        if action == Action.SHOOT_START:
            if not self._previous_shooting:
                actions.append(("mouse_left", True, aim_x, aim_y))
                self._previous_shooting = True
        elif action == Action.SHOOT_RELEASE:
            if self._previous_shooting:
                actions.append(("mouse_left", False, aim_x, aim_y))
                self._previous_shooting = False

        return actions

    def _release_all_controls(self) -> list[BotAction]:
        """Release all movement and shooting controls.

        Returns:
            List of actions to release all controls
        """
        actions: list[BotAction] = [
            ("w", False),
            ("a", False),
            ("d", False),
            ("s", False),
        ]

        if self._previous_shooting:
            actions.append(("mouse_left", False, 0.0, 0.0))
            self._previous_shooting = False

        # Reset state
        self._previous_movement_keys = {
            "a": False,
            "d": False,
            "w": False,
            "s": False,
        }
        self._previous_aim_direction = -1.0

        return actions

    def reset(self) -> None:
        """Reset the bot state for a new game/episode.

        This method should be called when starting a new game or episode
        to clear any internal state tracking.
        """
        self._previous_movement_keys = {
            "a": False,
            "d": False,
            "w": False,
            "s": False,
        }
        self._previous_aim_direction = -1.0
        self._previous_shooting = False


class NeuralNetBotRunner:
    """Runs a neural network bot connected to a game server.

    Handles integration between NeuralNetBot and GameClient,
    managing state updates and sending only changed inputs.
    """

    def __init__(
        self,
        network: ActorCriticNetwork,
        client: GameClient | None = None,
        config: NeuralNetBotConfig | None = None,
    ) -> None:
        """Initialize the bot runner.

        Args:
            network: Trained ActorCriticNetwork for inference
            client: GameClient for server communication (can be set later)
            config: Optional configuration for the neural net bot
        """
        self.client = client
        self.network = network
        self.config = config
        self.bot: NeuralNetBot | None = None
        self._previous_keyboard_actions: dict[str, bool] = {}
        self._previous_mouse_state: bool = False
        self._previous_aim_pos: tuple[float, float] = (0.0, 0.0)

    async def on_game_state(self, state: GameState) -> None:
        """Handle incoming game state updates.

        This method should be called whenever a new game state is received.
        It updates the bot's state and sends any necessary key presses and mouse inputs.

        Args:
            state: Current game state from the server

        Raises:
            ValueError: If the client's player_id is not set
        """
        if self.bot is None:
            if self.client.player_id is None:
                raise ValueError(
                    "Client player_id must be set before calling on_game_state"
                )
            self.bot = NeuralNetBot(self.client.player_id, self.network, self.config)

        self.bot.update_state(state)
        actions = await self.bot.decide_actions()

        for action in actions:
            if len(action) == 4:
                # Mouse action: ("mouse_left", pressed, aim_x, aim_y)
                _, pressed, aim_x, aim_y = action  # type: ignore[misc]
                aim_pos = (aim_x, aim_y)
                # Send if either mouse state or aim position changed
                if (
                    self._previous_mouse_state != pressed
                    or self._previous_aim_pos != aim_pos
                ):
                    await self.client.send_mouse_input("left", pressed, aim_x, aim_y)
                    self._previous_mouse_state = pressed
                    self._previous_aim_pos = aim_pos
            else:
                # Keyboard action: (key, pressed)
                key, pressed = action  # type: ignore[misc]
                if self._previous_keyboard_actions.get(key) != pressed:
                    await self.client.send_keyboard_input(key, pressed)
                    self._previous_keyboard_actions[key] = pressed

    def reset(self) -> None:
        """Reset the bot runner state for a new game/episode."""
        self._previous_keyboard_actions = {}
        self._previous_mouse_state = False
        self._previous_aim_pos = (0.0, 0.0)
        if self.bot is not None:
            self.bot.reset()
