"""Model opponent for self-play training.

This module provides a ModelOpponent class that wraps a trained neural network
and runs it as an opponent player in the training environment.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from bot.actions import Action, execute_action
from bot.client import ClientMode, GameClient
from bot.models import GameState
from bot.observation import ObservationBuilder

if TYPE_CHECKING:
    from bot.agent.network import ActorCriticNetwork


class ModelOpponent:
    """Trained model opponent for self-play training.

    Wraps a trained ActorCriticNetwork and runs it as an opponent player
    in the game environment.

    Design Decision - REST Mode vs WebSocket:
        The opponent operates in REST mode rather than WebSocket mode. While
        WebSocket would provide real-time game state streaming, REST mode is
        preferred for training because:

        1. **Synchronization**: REST mode ensures the opponent's actions are
           synchronized with the training environment's step cycle. The training
           env pushes state via on_game_state(), opponent computes action, and
           submits via REST - all in lockstep.

        2. **Determinism**: Eliminates timing-dependent behavior that could make
           training non-reproducible. The opponent acts exactly once per state
           update, not based on WebSocket message arrival timing.

        3. **Resource efficiency**: No background WebSocket listener task needed,
           reducing async complexity and potential race conditions.

    The model opponent uses deterministic action selection for consistent
    behavior during evaluation and training.

    Attributes:
        model: The trained neural network model.
        http_url: Base URL for game server REST API.
        ws_url: WebSocket URL for game server.
        player_name: Display name for the opponent.
        device: PyTorch device for inference.

    Example:
        opponent = ModelOpponent(
            model=trained_network,
            http_url="http://localhost:4000",
            ws_url="ws://localhost:4000/ws",
            player_name="Gen0Bot",
        )
        await opponent.start(room_code="ABC123")
        # ... training loop with game state updates ...
        await opponent.stop()
    """

    def __init__(
        self,
        model: "ActorCriticNetwork",
        http_url: str,
        ws_url: str,
        player_name: str = "ModelBot",
        device: torch.device | None = None,
    ) -> None:
        """Initialize the model opponent.

        Args:
            model: Trained neural network model.
            http_url: Base URL for game server REST API.
            ws_url: WebSocket URL for game server.
            player_name: Display name for the opponent.
            device: PyTorch device for inference. Defaults to CPU.
        """
        self.model = model
        self.http_url = http_url
        self.ws_url = ws_url
        self.player_name = player_name
        self.device = device or torch.device("cpu")

        self._client: GameClient | None = None
        self._observation_builder: ObservationBuilder | None = None
        self._running = False
        self._logger = logging.getLogger(__name__)

        # Move model to device and set to eval mode
        self.model.to(self.device)
        self.model.eval()

    async def start(self, room_code: str, room_password: str = "") -> None:
        """Start the model opponent and connect to the game.

        Joins the existing game room using REST mode for action submission.
        The opponent will react to game state updates pushed via on_game_state().

        Args:
            room_code: Room code to join.
            room_password: Optional room password.
        """
        # Create client in REST mode for training synchronization
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

        # Initialize observation builder
        self._observation_builder = ObservationBuilder()
        self._running = True

        self._logger.info(
            "ModelOpponent started: joined room %s as %s",
            room_code,
            self.player_name,
        )

    async def stop(self) -> None:
        """Stop the model opponent and disconnect from the game."""
        self._running = False

        if self._client is not None:
            await self._client.close()
            self._client = None

        self._observation_builder = None
        self._logger.info("ModelOpponent stopped")

    async def on_game_state(self, state: GameState) -> None:
        """Handle game state update.

        When the training environment steps, it pushes the new game state
        to the opponent. The opponent then uses the neural network to decide
        and execute its action.

        Args:
            state: Current game state.
        """
        if not self._running or self._client is None or self._observation_builder is None:
            return

        try:
            await self._act_on_state(state)
        except Exception as e:
            self._logger.warning("Error processing game state: %s", e)

    async def _act_on_state(self, state: GameState) -> None:
        """Select and execute action based on game state.

        Args:
            state: Current game state.
        """
        if self._client is None or self._observation_builder is None:
            return

        player_id = self._client.player_id
        if player_id is None:
            return

        # Check if player exists in state
        if player_id not in state.players:
            return

        # Build observation vector
        obs = self._observation_builder.build(
            game_state=state,
            own_player_id=player_id,
        )

        # Convert to tensor
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        obs_tensor = obs_tensor.unsqueeze(0)  # Add batch dimension

        # Get action from model (deterministic for consistent behavior)
        with torch.no_grad():
            action_tensor, _, _, _ = self.model.get_action_and_value(
                obs_tensor, deterministic=True
            )
            action_idx = int(action_tensor.item())

        # Execute action
        action = Action(action_idx)
        await execute_action(self._client, action)

    def reset(self) -> None:
        """Reset opponent state for a new episode.

        The model opponent doesn't maintain episode-level state,
        so this is a no-op. Provided for OpponentProtocol compatibility.
        """
        pass
