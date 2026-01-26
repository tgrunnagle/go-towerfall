"""Bot manager for coordinating bot lifecycle in the Bot Service.

This module provides the BotManager class that serves as the central coordinator
for bot lifecycle management. It tracks active bot instances, loads trained neural
network models from the ModelRegistry, and handles spawning/destroying bot connections.
"""

import asyncio
import logging
import uuid
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from bot.bots.neural_net_bot import NeuralNetBotConfig, NeuralNetBotRunner
from bot.bots.rule_based_bot import RuleBasedBotConfig, RuleBasedBotRunner
from bot.service.websocket_bot import WebSocketBotClient, WebSocketBotClientConfig
from bot.training.registry import ModelMetadata, ModelNotFoundError, ModelRegistry


class BotConfig(BaseModel):
    """Configuration for spawning a bot.

    Attributes:
        bot_type: Type of bot to spawn ("rule_based" or "neural_network")
        model_id: Model ID for neural network bots (e.g., "ppo_gen_005")
        generation: Generation number as alternative to model_id
        player_name: Display name for the bot in the game
    """

    bot_type: Literal["rule_based", "neural_network"]
    model_id: str | None = Field(
        default=None, description="Model ID for neural_network type"
    )
    generation: int | None = Field(
        default=None, description="Generation number (alternative to model_id)"
    )
    player_name: str = Field(default="Bot", description="Display name for the bot")

    @model_validator(mode="after")
    def validate_bot_config(self) -> "BotConfig":
        """Validate that bot configuration is consistent with bot type."""
        if self.bot_type == "rule_based":
            if self.model_id is not None or self.generation is not None:
                raise ValueError(
                    "Rule-based bots cannot specify model_id or generation"
                )
        elif self.bot_type == "neural_network":
            if self.model_id is None and self.generation is None:
                raise ValueError(
                    "Neural network bots require either model_id or generation"
                )
        return self


class SpawnBotRequest(BaseModel):
    """Request to spawn a bot into a game room.

    Attributes:
        room_code: Room code to join
        room_password: Room password (empty string if none)
        bot_config: Configuration for the bot
    """

    room_code: str
    room_password: str = ""
    bot_config: BotConfig


class SpawnBotResponse(BaseModel):
    """Response from spawning a bot.

    Attributes:
        success: Whether the spawn was initiated successfully
        bot_id: Unique identifier for the spawned bot (if success)
        error: Error message (if not success)
    """

    success: bool
    bot_id: str | None = None
    error: str | None = None


class BotInfo(BaseModel):
    """Information about an active bot.

    Attributes:
        bot_id: Unique identifier for the bot
        bot_type: Type of bot ("rule_based" or "neural_network")
        model_id: Model ID if neural network bot
        player_name: Display name of the bot
        room_code: Room the bot is in
        is_connected: Whether the bot is currently connected
    """

    bot_id: str
    bot_type: str
    model_id: str | None = None
    player_name: str
    room_code: str
    is_connected: bool


class _BotEntry:
    """Internal tracking entry for active bots."""

    def __init__(
        self,
        bot_id: str,
        client: WebSocketBotClient,
        bot_config: BotConfig,
        room_code: str,
        model_id: str | None = None,
    ) -> None:
        self.bot_id = bot_id
        self.client = client
        self.bot_config = bot_config
        self.room_code = room_code
        self.model_id = model_id


class BotManager:
    """Manages bot lifecycle for the Bot Service.

    Tracks active bots, loads models from ModelRegistry, and coordinates
    spawning/destroying bot connections.

    Example:
        registry = ModelRegistry("/path/to/registry")
        manager = BotManager(
            registry=registry,
            http_url="http://localhost:4000",
            ws_url="ws://localhost:4000/ws",
        )

        # Spawn a neural network bot
        request = SpawnBotRequest(
            room_code="ABC123",
            room_password="secret",
            bot_config=BotConfig(
                bot_type="neural_network",
                generation=5,
                player_name="NeuralBot",
            ),
        )
        response = await manager.spawn_bot(request)
        if response.success:
            print(f"Bot spawned: {response.bot_id}")

        # Later, destroy the bot
        await manager.destroy_bot(response.bot_id)
    """

    def __init__(
        self,
        registry: ModelRegistry | None,
        http_url: str,
        ws_url: str,
        default_device: str = "cpu",
    ) -> None:
        """Initialize the BotManager.

        Args:
            registry: ModelRegistry for loading trained models (None if no ML models)
            http_url: Base URL for game server REST API
            ws_url: WebSocket URL for game server
            default_device: Device for neural network inference ("cpu" or "cuda")
        """
        self._registry = registry
        self._http_url = http_url
        self._ws_url = ws_url
        self._device = default_device
        self._bots: dict[str, _BotEntry] = {}
        self._tasks: set[asyncio.Task] = set()
        self._logger = logging.getLogger(__name__)

    async def spawn_bot(self, request: SpawnBotRequest) -> SpawnBotResponse:
        """Spawn a bot to join a game room.

        Creates the bot instance and starts connecting in the background.
        Returns immediately with the bot_id.

        Args:
            request: Spawn request with room info and bot config

        Returns:
            SpawnBotResponse with success status and bot_id
        """
        bot_id = self._generate_bot_id()

        try:
            # Create bot client
            client, model_id = self._create_bot_client(request.bot_config)

            # Track the bot
            entry = _BotEntry(
                bot_id=bot_id,
                client=client,
                bot_config=request.bot_config,
                room_code=request.room_code,
                model_id=model_id,
            )
            self._bots[bot_id] = entry

            # Start connection in background - don't await
            task = asyncio.create_task(self._connect_bot(bot_id, client, request))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

            self._logger.info(
                "Bot spawned: %s (%s)", bot_id, request.bot_config.bot_type
            )
            return SpawnBotResponse(success=True, bot_id=bot_id)

        except ModelNotFoundError as e:
            return SpawnBotResponse(success=False, error=str(e))
        except ValueError as e:
            return SpawnBotResponse(success=False, error=str(e))
        except Exception as e:
            self._logger.error("Failed to spawn bot: %s", e)
            return SpawnBotResponse(success=False, error=str(e))

    async def destroy_bot(self, bot_id: str) -> bool:
        """Stop and remove a bot.

        Args:
            bot_id: ID of the bot to destroy

        Returns:
            True if bot was found and destroyed, False if not found
        """
        entry = self._bots.pop(bot_id, None)
        if entry is None:
            return False

        try:
            await entry.client.stop()
            self._logger.info("Bot destroyed: %s", bot_id)
        except Exception as e:
            self._logger.warning("Error stopping bot %s: %s", bot_id, e)

        return True

    def get_bot(self, bot_id: str) -> BotInfo | None:
        """Get information about a specific bot.

        Args:
            bot_id: ID of the bot to look up

        Returns:
            BotInfo if found, None otherwise
        """
        entry = self._bots.get(bot_id)
        if entry is None:
            return None

        return BotInfo(
            bot_id=entry.bot_id,
            bot_type=entry.bot_config.bot_type,
            model_id=entry.model_id,
            player_name=entry.bot_config.player_name,
            room_code=entry.room_code,
            is_connected=entry.client.is_connected,
        )

    def list_bots(self) -> list[BotInfo]:
        """List all active bots.

        Returns:
            List of BotInfo for all tracked bots
        """
        result = []
        for bot_id in self._bots:
            bot_info = self.get_bot(bot_id)
            if bot_info is not None:
                result.append(bot_info)
        return result

    def list_models(self) -> list[ModelMetadata]:
        """List available trained models.

        Returns:
            List of ModelMetadata from the registry
        """
        if self._registry is None:
            return []
        return self._registry.list_models()

    async def _await_pending_tasks(self) -> None:
        """Await completion of pending background tasks.

        This method is primarily for testing to ensure background connection
        tasks have completed before assertions.
        """
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

    async def shutdown(self) -> None:
        """Gracefully shut down all bots.

        Should be called when the service is stopping.
        """
        self._logger.info(
            "Shutting down BotManager with %d active bots", len(self._bots)
        )

        # Stop all bots concurrently
        bot_ids = list(self._bots.keys())
        tasks = [self.destroy_bot(bot_id) for bot_id in bot_ids]
        await asyncio.gather(*tasks, return_exceptions=True)

        # Wait for any pending background tasks to complete
        if self._tasks:
            self._logger.info("Waiting for %d background tasks", len(self._tasks))
            await asyncio.gather(*self._tasks, return_exceptions=True)

        self._logger.info("BotManager shutdown complete")

    def _generate_bot_id(self) -> str:
        """Generate a unique bot ID."""
        return f"bot_{uuid.uuid4().hex[:12]}"

    def _create_bot_client(
        self, config: BotConfig
    ) -> tuple[WebSocketBotClient, str | None]:
        """Create a WebSocket bot client with appropriate runner.

        Args:
            config: Bot configuration

        Returns:
            Tuple of (WebSocketBotClient, model_id or None)

        Raises:
            ModelNotFoundError: If neural network model not found
            ValueError: If invalid configuration
        """
        # Create WebSocket config
        ws_config = WebSocketBotClientConfig(
            http_url=self._http_url,
            ws_url=self._ws_url,
            player_name=config.player_name,
        )

        model_id = None

        if config.bot_type == "rule_based":
            # Create rule-based bot runner
            runner = RuleBasedBotRunner(
                client=None,
                config=RuleBasedBotConfig(),
            )
        else:  # neural_network
            # Load model from registry
            network, metadata = self._load_model(config)
            model_id = metadata.model_id

            # Create neural network bot runner
            runner = NeuralNetBotRunner(
                network=network,
                client=None,
                config=NeuralNetBotConfig(device=self._device),
            )

        return WebSocketBotClient(runner=runner, config=ws_config), model_id

    def _load_model(self, config: BotConfig):
        """Load a neural network model from the registry.

        Args:
            config: Bot configuration with model_id or generation

        Returns:
            Tuple of (ActorCriticNetwork, ModelMetadata)

        Raises:
            ModelNotFoundError: If model not found
            ValueError: If no registry or invalid config
        """
        if self._registry is None:
            raise ValueError("Neural network bot requires a ModelRegistry")

        if config.model_id:
            return self._registry.get_model(config.model_id, device=self._device)
        elif config.generation is not None:
            result = self._registry.get_model_by_generation(
                config.generation, device=self._device
            )
            if result is None:
                raise ModelNotFoundError(
                    f"No model found for generation {config.generation}"
                )
            return result
        else:
            raise ValueError("Neural network bot requires model_id or generation")

    async def _connect_bot(
        self,
        bot_id: str,
        client: WebSocketBotClient,
        request: SpawnBotRequest,
    ) -> None:
        """Connect bot in background with error handling.

        Args:
            bot_id: Unique bot identifier
            client: WebSocket bot client to connect
            request: Spawn request with room details
        """
        try:
            await client.start(request.room_code, request.room_password)
            self._logger.info("Bot %s connected to room %s", bot_id, request.room_code)
        except Exception as e:
            self._logger.error("Bot %s failed to connect: %s", bot_id, e)
            # Remove from tracking on failure
            self._bots.pop(bot_id, None)
