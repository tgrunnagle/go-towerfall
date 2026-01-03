"""Game state container models.

This module contains models for game state updates and the overall game state container.
"""

from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from bot.models.game_objects import ArrowState, BlockState, BulletState, PlayerState


class GameUpdateEvent(BaseModel):
    """Represents an event within a game update."""

    model_config = ConfigDict(populate_by_name=True)

    type: str
    data: dict[str, Any] = Field(default_factory=dict)


class GameUpdate(BaseModel):
    """Represents a game state update from WebSocket.

    Maps to Go `types.GameUpdate`.
    """

    model_config = ConfigDict(populate_by_name=True)

    full_update: bool = Field(alias="fullUpdate")
    object_states: dict[str, dict[str, Any] | None] = Field(alias="objectStates")
    events: list[GameUpdateEvent] = Field(default_factory=list)
    training_complete: bool = Field(default=False, alias="trainingComplete")


class GameState(BaseModel):
    """High-level container for parsed game state.

    This model provides typed access to all game objects organized by type.
    Use the `from_update` class method to parse a GameUpdate into this typed structure.
    """

    model_config = ConfigDict(populate_by_name=True)

    players: dict[str, PlayerState] = Field(default_factory=dict)
    arrows: dict[str, ArrowState] = Field(default_factory=dict)
    blocks: dict[str, BlockState] = Field(default_factory=dict)
    bullets: dict[str, BulletState] = Field(default_factory=dict)
    canvas_size_x: int = Field(default=800)
    canvas_size_y: int = Field(default=800)
    is_game_over: bool = Field(default=False)

    # Object type constants matching Go server
    OBJECT_TYPE_PLAYER: ClassVar[str] = "player"
    OBJECT_TYPE_ARROW: ClassVar[str] = "arrow"
    OBJECT_TYPE_BLOCK: ClassVar[str] = "block"
    OBJECT_TYPE_BULLET: ClassVar[str] = "bullet"

    @classmethod
    def from_update(
        cls,
        update: GameUpdate,
        existing_state: "GameState | None" = None,
        canvas_size_x: int = 800,
        canvas_size_y: int = 800,
    ) -> "GameState":
        """Parse a GameUpdate into a typed GameState.

        Args:
            update: The GameUpdate message from the server.
            existing_state: Optional existing state to merge with (for incremental updates).
            canvas_size_x: Canvas width in pixels.
            canvas_size_y: Canvas height in pixels.

        Returns:
            A new GameState with parsed and typed objects.
        """
        # Start with existing state or empty collections
        if existing_state and not update.full_update:
            players = dict(existing_state.players)
            arrows = dict(existing_state.arrows)
            blocks = dict(existing_state.blocks)
            bullets = dict(existing_state.bullets)
            canvas_size_x = existing_state.canvas_size_x
            canvas_size_y = existing_state.canvas_size_y
        else:
            players = {}
            arrows = {}
            blocks = {}
            bullets = {}

        # Process object states
        for object_id, state in update.object_states.items():
            # None value means object was destroyed
            if state is None:
                players.pop(object_id, None)
                arrows.pop(object_id, None)
                blocks.pop(object_id, None)
                bullets.pop(object_id, None)
                continue

            object_type = state.get("objectType")
            if object_type == cls.OBJECT_TYPE_PLAYER:
                players[object_id] = PlayerState.model_validate(state)
            elif object_type == cls.OBJECT_TYPE_ARROW:
                arrows[object_id] = ArrowState.model_validate(state)
            elif object_type == cls.OBJECT_TYPE_BLOCK:
                blocks[object_id] = BlockState.model_validate(state)
            elif object_type == cls.OBJECT_TYPE_BULLET:
                bullets[object_id] = BulletState.model_validate(state)

        return cls(
            players=players,
            arrows=arrows,
            blocks=blocks,
            bullets=bullets,
            canvas_size_x=canvas_size_x,
            canvas_size_y=canvas_size_y,
            is_game_over=update.training_complete,
        )

    def get_player_by_name(self, name: str) -> PlayerState | None:
        """Find a player by their name."""
        for player in self.players.values():
            if player.name == name:
                return player
        return None

    def get_living_players(self) -> list[PlayerState]:
        """Get all players that are not dead."""
        return [p for p in self.players.values() if not p.dead]

    def get_active_arrows(self) -> list[ArrowState]:
        """Get all arrows that are not destroyed."""
        return [a for a in self.arrows.values() if not a.destroyed]
