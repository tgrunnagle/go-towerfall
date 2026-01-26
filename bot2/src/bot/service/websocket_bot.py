"""WebSocket bot client for connecting bots to game servers.

This module provides the WebSocketBotClient class that connects bot runners
(NeuralNetBotRunner or RuleBasedBotRunner) to game servers via WebSocket.
It manages the WebSocket lifecycle and forwards game state updates to the
bot runner for action decisions.
"""

import logging
from dataclasses import dataclass
from typing import Protocol

from bot.client import ClientMode, GameClient
from bot.models import GameState


class BotRunnerProtocol(Protocol):
    """Protocol for bot runners that can process game state updates.

    Both NeuralNetBotRunner and RuleBasedBotRunner implement this interface.

    Attributes:
        client: GameClient instance for sending actions to the server.
            WebSocketBotClient will set this attribute after constructing the runner.
    """

    client: GameClient | None

    async def on_game_state(self, state: GameState) -> None:
        """Handle incoming game state and execute bot actions.

        Args:
            state: Current game state from the server.
        """
        ...

    def reset(self) -> None:
        """Reset bot state for a new game/episode."""
        ...


@dataclass
class WebSocketBotClientConfig:
    """Configuration for WebSocketBotClient.

    Attributes:
        http_url: Base URL for game server REST API (for join_game).
        ws_url: WebSocket URL for real-time communication.
        player_name: Display name for the bot in the game.
        timeout: Connection timeout in seconds.
    """

    http_url: str = "http://localhost:4000"
    ws_url: str = "ws://localhost:4000/ws"
    player_name: str = "Bot"
    timeout: float = 30.0


class WebSocketBotClient:
    """WebSocket client that connects a bot runner to a game server.

    Manages the WebSocket lifecycle and forwards game state updates
    to the bot runner for action decisions. Supports any bot runner
    implementing the BotRunnerProtocol.

    The WebSocketBotClient creates its own GameClient and injects it into
    the runner's client attribute during start(). Bot runners should be
    instantiated with a GameClient (which will be replaced) or create
    a placeholder client that will be replaced by WebSocketBotClient.

    Example:
        # Create a placeholder client for the runner
        placeholder_client = GameClient(http_url="http://localhost:4000")
        runner = NeuralNetBotRunner(network=network, client=placeholder_client)

        # WebSocketBotClient will replace the client during start()
        bot_client = WebSocketBotClient(runner=runner, config=config)
        await bot_client.start(room_code="ABC123", room_password="secret")
        # Bot is now playing in the game...
        await bot_client.stop()
    """

    def __init__(
        self,
        runner: BotRunnerProtocol,
        config: WebSocketBotClientConfig | None = None,
    ) -> None:
        """Initialize the WebSocket bot client.

        Args:
            runner: Bot runner that will process game state and decide actions.
            config: Optional configuration for connection settings.
        """
        self.runner = runner
        self.config = config or WebSocketBotClientConfig()
        self._client: GameClient | None = None
        self._running = False
        self._logger = logging.getLogger(__name__)

    async def start(self, room_code: str, room_password: str = "") -> None:
        """Start the bot client and join the game room.

        Creates a GameClient in WebSocket mode, connects to the server,
        joins the specified room, and begins receiving game state updates.

        Args:
            room_code: Room code to join.
            room_password: Optional room password.

        Raises:
            RuntimeError: If the client is already started.
        """
        # Check if already started to prevent resource leaks
        if self._running or self._client is not None:
            raise RuntimeError(
                "WebSocketBotClient is already started. Call stop() first."
            )

        # Create client in WebSocket mode for real-time updates
        self._client = GameClient(
            http_url=self.config.http_url,
            ws_url=self.config.ws_url,
            mode=ClientMode.WEBSOCKET,
        )

        # Inject the client into the runner
        # BotRunnerProtocol defines client as a required attribute
        self.runner.client = self._client

        # Connect to server
        await self._client.connect()

        # Join the game room
        await self._client.join_game(
            room_code=room_code,
            player_name=self.config.player_name,
            room_password=room_password,
            is_spectator=False,
        )

        # Register message handler for WebSocket updates
        self._client.register_message_handler(self._handle_message)

        self._running = True
        self._logger.info(
            "WebSocketBotClient started: joined room %s as %s",
            room_code,
            self.config.player_name,
        )

    async def stop(self) -> None:
        """Stop the bot client and disconnect from the game.

        Gracefully closes the WebSocket connection and cleans up resources.
        Safe to call even if not started.
        """
        self._running = False

        if self._client is not None:
            await self._client.close()
            self._client = None

        self._logger.info("WebSocketBotClient stopped")

    def reset(self) -> None:
        """Reset the bot runner state for a new game/episode.

        Delegates to the bot runner's reset() method without disconnecting.
        """
        if self.runner is not None:
            self.runner.reset()

    async def _handle_message(self, message: dict) -> None:
        """Handle incoming WebSocket messages.

        Filters for GameState messages and forwards them to the bot runner.

        Args:
            message: Raw WebSocket message as dict.
        """
        # Only process if running
        if not self._running:
            return

        # Filter for GameState messages
        if message.get("type") != "GameState":
            return

        # Get the cached game state from the client
        # The client automatically parses GameState messages and caches them
        if self._client is None or self._client._game_state is None:
            return

        try:
            # Forward game state to the bot runner
            await self.runner.on_game_state(self._client._game_state)
        except Exception as e:
            # Catch errors from bot runner to prevent WebSocket listener crash
            self._logger.warning("Error processing game state in bot runner: %s", e)

    @property
    def player_id(self) -> str | None:
        """Get the player ID assigned by the server after joining.

        Returns:
            Player ID if connected, None otherwise.
        """
        if self._client is not None:
            return self._client.player_id
        return None

    @property
    def is_connected(self) -> bool:
        """Check if the client is currently connected to a game.

        Returns:
            True if connected and running, False otherwise.
        """
        return self._running and self._client is not None
