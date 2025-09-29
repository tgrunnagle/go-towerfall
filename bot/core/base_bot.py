from abc import ABC, abstractmethod
from core.game_state import GameState
from enum import Enum
from dataclasses import dataclass


class ActionPriority(Enum):
    """Priority levels for different actions"""

    CRITICAL = 1.0  # Immediate survival actions
    HIGH = 0.8  # Important tactical actions
    MEDIUM = 0.6  # Strategic actions
    LOW = 0.4  # Opportunistic actions
    MINIMAL = 0.2  # Background actions


@dataclass
class Action:
    """Represents a potential action the bot can take"""

    action_type: str
    parameters: dict[str, any]
    priority: ActionPriority
    confidence: float  # 0.0 to 1.0
    expected_outcome: str
    duration: float | None  # How long to hold the action


class BaseBot(ABC):

    @abstractmethod
    def process_state_and_get_action(
        self, game_state: GameState
    ) -> tuple[Action | None, dict]:
        """
        Processes the current game state and returns an action to take, if any,
        along with any analysis that went into the action decision
        """
        pass
