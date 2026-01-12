"""Custom exceptions for the game server manager.

This module defines the exception hierarchy for game server management errors.
"""


class GameServerError(Exception):
    """Base exception for game server manager errors."""

    pass


class GameCreationError(GameServerError):
    """Raised when game creation fails."""

    pass


class MaxGamesExceededError(GameServerError):
    """Raised when maximum concurrent games limit is reached."""

    pass


class GameNotFoundError(GameServerError):
    """Raised when a game instance is not found."""

    pass
