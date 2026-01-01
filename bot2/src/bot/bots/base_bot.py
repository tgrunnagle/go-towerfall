"""Abstract base class for bot implementations."""

from abc import ABC, abstractmethod

from bot.models import GameState, PlayerState


class BaseBot(ABC):
    """Abstract base class for bot implementations.

    All bot implementations should extend this class and implement
    the decide_actions method to determine movement actions based
    on the current game state.
    """

    def __init__(self, player_id: str) -> None:
        """Initialize the bot.

        Args:
            player_id: The unique identifier for this bot's player.
        """
        self.player_id = player_id
        self.current_state: GameState | None = None

    def update_state(self, state: GameState) -> None:
        """Update the bot's knowledge of game state.

        Args:
            state: The current game state from the server.
        """
        self.current_state = state

    @abstractmethod
    async def decide_actions(self) -> list[tuple[str, bool]]:
        """Decide which actions to take based on current state.

        Returns:
            List of (key, is_pressed) tuples for keyboard inputs.
            Valid keys are: "w" (jump), "a" (left), "s" (dive), "d" (right).
        """
        pass

    def get_own_player(self) -> PlayerState | None:
        """Get this bot's player state.

        Returns:
            The PlayerState for this bot, or None if not found.
        """
        if self.current_state is None:
            return None
        return self.current_state.players.get(self.player_id)

    def get_enemies(self) -> list[PlayerState]:
        """Get list of enemy player states (alive only).

        Returns:
            List of PlayerState objects for all alive enemy players.
        """
        if self.current_state is None:
            return []
        return [
            p
            for pid, p in self.current_state.players.items()
            if pid != self.player_id and not p.dead
        ]
