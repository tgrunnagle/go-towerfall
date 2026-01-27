"""Opponent management for TowerFall RL training environments.

This module provides opponent implementations that can be used during training:
- RuleBasedOpponent: A rule-based bot that plays against the ML agent
- ModelOpponent: A trained neural network opponent for self-play training
- NoOpponent: A no-op implementation for single-player training

The opponent manager integrates with the training environment lifecycle,
managing opponent connections and game state updates.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol

import torch

from bot.bots.rule_based_bot import RuleBasedBotConfig, RuleBasedBotRunner
from bot.client import ClientMode, GameClient
from bot.models import GameState

if TYPE_CHECKING:
    from bot.agent.network import ActorCriticNetwork


class OpponentProtocol(Protocol):
    """Protocol for opponent implementations.

    Opponents must implement start/stop lifecycle methods and handle
    game state updates for reactive behavior.
    """

    async def start(self, room_code: str, room_password: str = "") -> None:
        """Start the opponent (connect to game, begin acting).

        Args:
            room_code: Room code to join.
            room_password: Optional room password.
        """
        ...

    async def stop(self) -> None:
        """Stop the opponent (disconnect from game)."""
        ...

    async def on_game_state(self, state: GameState) -> None:
        """Handle game state update (for opponents that need to react).

        Args:
            state: Current game state.
        """
        ...

    def reset(self) -> None:
        """Reset opponent state for a new episode."""
        ...


class RuleBasedOpponent:
    """Rule-based bot opponent for training environment.

    Manages a RuleBasedBot instance connected to the game as a second player.
    The opponent runs in a background task, reacting to game state updates
    pushed from the training environment.

    In REST mode training, the environment pushes game state updates to the
    opponent via on_game_state() after each step. The opponent then decides
    and executes its actions.

    Attributes:
        http_url: Base URL for game server REST API.
        ws_url: WebSocket URL for game server.
        player_name: Name for the opponent player.
    """

    def __init__(
        self,
        http_url: str,
        ws_url: str,
        player_name: str = "RuleBot",
        config: RuleBasedBotConfig | None = None,
    ):
        """Initialize the rule-based opponent.

        Args:
            http_url: Base URL for game server REST API.
            ws_url: WebSocket URL for game server.
            player_name: Name for the opponent player.
            config: Optional configuration for the rule-based bot.
        """
        self.http_url = http_url
        self.ws_url = ws_url
        self.player_name = player_name
        self.config = config

        self._client: GameClient | None = None
        self._runner: RuleBasedBotRunner | None = None
        self._running = False
        self._logger = logging.getLogger(__name__)

    async def start(self, room_code: str, room_password: str = "") -> None:
        """Start the opponent and connect to the game.

        Joins the existing game room using REST mode for action submission.
        The opponent will react to game state updates pushed via on_game_state().

        Args:
            room_code: Room code to join.
            room_password: Optional room password.
        """
        # Create client in REST mode for training synchronization
        # This allows the opponent to act when the training env pushes state
        self._client = GameClient(
            http_url=self.http_url,
            ws_url=self.ws_url,
            mode=ClientMode.REST,
        )
        await self._client.connect()

        # Join the existing game room
        await self._client.join_game(
            room_code=room_code,
            player_name=self.player_name,
            room_password=room_password,
            is_spectator=False,
        )

        # Initialize the bot runner
        self._runner = RuleBasedBotRunner(client=self._client, config=self.config)
        self._running = True

        self._logger.info(
            "RuleBasedOpponent started: joined room %s as %s",
            room_code,
            self.player_name,
        )

    async def stop(self) -> None:
        """Stop the opponent and disconnect from the game."""
        self._running = False

        if self._client is not None:
            await self._client.close()
            self._client = None

        self._runner = None
        self._logger.info("RuleBasedOpponent stopped")

    async def on_game_state(self, state: GameState) -> None:
        """Handle game state update.

        When the training environment steps, it pushes the new game state
        to the opponent. The opponent then decides and executes its actions.

        Args:
            state: Current game state.
        """
        if not self._running or self._runner is None:
            return

        try:
            await self._runner.on_game_state(state)
        except Exception as e:
            self._logger.warning("Error processing game state: %s", e)

    def reset(self) -> None:
        """Reset opponent state for a new episode.

        Clears the bot's internal state (e.g., shooting timers) without
        disconnecting from the game.
        """
        if self._runner is not None:
            self._runner.reset()


class NoOpponent:
    """Null opponent for single-player training scenarios.

    Implements the OpponentProtocol with no-op methods, allowing the
    environment to be used without a second player.
    """

    async def start(self, room_code: str, room_password: str = "") -> None:
        """No-op start."""
        pass

    async def stop(self) -> None:
        """No-op stop."""
        pass

    async def on_game_state(self, state: GameState) -> None:
        """No-op state update."""
        pass

    def reset(self) -> None:
        """No-op reset."""
        pass


def create_opponent(
    opponent_type: str,
    http_url: str,
    ws_url: str,
    player_name: str = "Opponent",
    config: RuleBasedBotConfig | None = None,
    model: "ActorCriticNetwork | None" = None,
    device: torch.device | None = None,
) -> OpponentProtocol:
    """Factory function to create opponent instances.

    Args:
        opponent_type: Type of opponent ("rule_based", "none", "model").
        http_url: Base URL for game server REST API.
        ws_url: WebSocket URL for game server.
        player_name: Name for the opponent player.
        config: Optional configuration for rule-based bot.
        model: Neural network model (required for "model" type).
        device: PyTorch device for model inference.

    Returns:
        Opponent instance implementing OpponentProtocol.

    Raises:
        ValueError: If opponent_type is not recognized or model missing for "model" type.
    """
    if opponent_type == "rule_based":
        return RuleBasedOpponent(
            http_url=http_url,
            ws_url=ws_url,
            player_name=player_name,
            config=config,
        )
    elif opponent_type == "none":
        return NoOpponent()
    elif opponent_type == "model":
        if model is None:
            raise ValueError("model parameter required for opponent_type='model'")
        from bot.gym.model_opponent import ModelOpponent

        return ModelOpponent(
            model=model,
            http_url=http_url,
            ws_url=ws_url,
            player_name=player_name,
            device=device,
        )
    else:
        raise ValueError(f"Unknown opponent type: {opponent_type}")
