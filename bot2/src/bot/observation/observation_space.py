"""Observation space builder for RL training.

Converts game state into normalized observation vectors suitable for
neural network training.
"""

import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from bot.models import ArrowState, GameState, PlayerState
from bot.observation.map_encoder import (
    DEFAULT_MAP_CONFIG,
    MapEncoder,
    MapEncodingConfig,
)
from bot.observation.normalizer import (
    NORMALIZATION_CONSTANTS,
    NormalizationConstants,
    calculate_angle_to,
    calculate_distance,
    normalize_angle,
    normalize_boolean,
    normalize_count,
    normalize_distance,
    normalize_health,
    normalize_position,
    normalize_relative_position,
    normalize_velocity,
)


@dataclass(frozen=True)
class ObservationConfig:
    """Configuration for observation space dimensions.

    Attributes:
        max_other_players: Maximum number of other players to observe
        max_tracked_arrows: Maximum number of arrows to track
        include_map: Whether to include map geometry in observations
        map_encoding: Configuration for map encoding (grid dimensions, etc.)
    """

    max_other_players: int = 3
    max_tracked_arrows: int = 8
    include_map: bool = True
    map_encoding: MapEncodingConfig = DEFAULT_MAP_CONFIG

    @property
    def own_player_size(self) -> int:
        """Size of own player observation vector."""
        return 14

    @property
    def per_player_size(self) -> int:
        """Size of each other player observation vector."""
        return 12

    @property
    def per_arrow_size(self) -> int:
        """Size of each arrow observation vector."""
        return 8

    @property
    def other_players_size(self) -> int:
        """Total size of other players observation section."""
        return self.max_other_players * self.per_player_size

    @property
    def arrows_size(self) -> int:
        """Total size of arrows observation section."""
        return self.max_tracked_arrows * self.per_arrow_size

    @property
    def map_size(self) -> int:
        """Total size of map observation section."""
        if self.include_map:
            return self.map_encoding.total_size
        return 0

    @property
    def total_size(self) -> int:
        """Total size of observation vector."""
        return (
            self.own_player_size
            + self.other_players_size
            + self.arrows_size
            + self.map_size
        )


# Default configuration
DEFAULT_CONFIG = ObservationConfig()


class ObservationBuilder:
    """Builds normalized observation vectors from game state.

    The observation vector contains:
    - Own player state (14 values)
    - Other players' states (max_other_players * 12 values)
    - Arrow states (max_tracked_arrows * 8 values)
    - Map geometry (grid_width * grid_height values, if include_map=True)

    All values are normalized to [-1, 1] range.
    """

    def __init__(
        self,
        config: ObservationConfig | None = None,
        constants: NormalizationConstants | None = None,
    ):
        """Initialize the observation builder.

        Args:
            config: Observation space configuration
            constants: Normalization constants
        """
        self.config = config or DEFAULT_CONFIG
        self.constants = constants or NORMALIZATION_CONSTANTS

        # Initialize map encoder if map encoding is enabled
        if self.config.include_map:
            self._map_encoder = MapEncoder(config=self.config.map_encoding)
        else:
            self._map_encoder = None

    def build(
        self,
        game_state: GameState,
        own_player_id: str,
    ) -> NDArray[np.float32]:
        """Convert game state to normalized observation vector.

        Args:
            game_state: Current game state
            own_player_id: ID of the player to build observation for

        Returns:
            Normalized observation vector of shape (total_size,)

        Raises:
            ValueError: If own_player_id is not found in game state
        """
        # Get own player
        own_player = game_state.players.get(own_player_id)
        if own_player is None:
            raise ValueError(f"Player {own_player_id} not found in game state")

        # Initialize observation vector
        obs = np.zeros(self.config.total_size, dtype=np.float32)
        offset = 0

        # 1. Own player state
        own_obs = self._normalize_own_player(own_player)
        obs[offset : offset + self.config.own_player_size] = own_obs
        offset += self.config.own_player_size

        # 2. Other players' states (sorted by distance)
        other_players = [
            player for pid, player in game_state.players.items() if pid != own_player_id
        ]
        # Sort by distance to own player
        other_players.sort(
            key=lambda p: calculate_distance(own_player.x, own_player.y, p.x, p.y)
        )

        for i in range(self.config.max_other_players):
            if i < len(other_players):
                player_obs = self._normalize_other_player(other_players[i], own_player)
            else:
                # Zero-pad if fewer players than max
                player_obs = np.zeros(self.config.per_player_size, dtype=np.float32)
            obs[offset : offset + self.config.per_player_size] = player_obs
            offset += self.config.per_player_size

        # 3. Arrow states (sorted by distance, active arrows first)
        arrows = list(game_state.arrows.values())
        # Filter out destroyed arrows and sort by distance
        active_arrows = [a for a in arrows if not a.destroyed]
        active_arrows.sort(
            key=lambda a: calculate_distance(own_player.x, own_player.y, a.x, a.y)
        )

        for i in range(self.config.max_tracked_arrows):
            if i < len(active_arrows):
                arrow_obs = self._normalize_arrow(
                    active_arrows[i], own_player, own_player_id
                )
            else:
                # Zero-pad if fewer arrows than max
                arrow_obs = np.zeros(self.config.per_arrow_size, dtype=np.float32)
            obs[offset : offset + self.config.per_arrow_size] = arrow_obs
            offset += self.config.per_arrow_size

        # 4. Map geometry (if enabled)
        if self.config.include_map and self._map_encoder is not None:
            blocks = list(game_state.blocks.values())
            map_obs = self._map_encoder.encode(blocks)
            obs[offset : offset + self.config.map_size] = map_obs
            offset += self.config.map_size

        return obs

    def _normalize_own_player(
        self,
        player: PlayerState,
    ) -> NDArray[np.float32]:
        """Normalize own player state to observation vector.

        Own player observation (14 values):
        [0-1] Position (x, y)
        [2-3] Velocity (dx, dy)
        [4]   Direction (aim angle)
        [5]   Health
        [6]   Is Dead
        [7]   Is Shooting
        [8]   Jump Count
        [9]   Arrow Count
        [10-11] Shooting power (if shooting, else 0)
        [12-13] Reserved

        Args:
            player: Own player state

        Returns:
            Normalized observation vector of shape (14,)
        """
        obs = np.zeros(self.config.own_player_size, dtype=np.float32)

        # Position [0-1]
        pos_x, pos_y = normalize_position(
            player.x,
            player.y,
            self.constants.ROOM_WIDTH,
            self.constants.ROOM_HEIGHT,
        )
        obs[0] = pos_x
        obs[1] = pos_y

        # Velocity [2-3]
        vel_x, vel_y = normalize_velocity(
            player.dx,
            player.dy,
            self.constants.MAX_PLAYER_VELOCITY,
        )
        obs[2] = vel_x
        obs[3] = vel_y

        # Direction [4]
        obs[4] = normalize_angle(player.direction)

        # Health [5]
        obs[5] = normalize_health(player.health, self.constants.MAX_HEALTH)

        # Is Dead [6]
        obs[6] = normalize_boolean(player.dead)

        # Is Shooting [7]
        obs[7] = normalize_boolean(player.shooting)

        # Jump Count [8]
        obs[8] = normalize_count(player.jump_count, self.constants.MAX_JUMPS)

        # Arrow Count [9]
        obs[9] = normalize_count(player.arrow_count, self.constants.MAX_ARROWS)

        # Shooting power [10-11]
        # Calculate power ratio based on how long the shot has been held
        # Power ratio goes from 0 to 1 over MAX_ARROW_POWER_TIME seconds
        if player.shooting and player.shooting_start_time is not None:
            # Backend sends shooting_start_time in milliseconds, convert to seconds
            current_time_ms = time.time() * 1000.0
            elapsed_time = (current_time_ms - player.shooting_start_time) / 1000.0
            power_ratio = min(
                1.0, max(0.0, elapsed_time / self.constants.MAX_ARROW_POWER_TIME)
            )
            # Normalize to [-1, 1] range: 0 power = -1, max power = 1
            obs[10] = (power_ratio * 2.0) - 1.0
        else:
            obs[10] = -1.0  # Not shooting = minimum power
        obs[11] = 0.0  # Reserved for future use

        # Reserved [12-13]
        obs[12] = 0.0
        obs[13] = 0.0

        return obs

    def _normalize_other_player(
        self,
        player: PlayerState,
        own_player: PlayerState,
    ) -> NDArray[np.float32]:
        """Normalize other player state relative to own player.

        Other player observation (12 values):
        [0-1] Relative position (x, y)
        [2-3] Velocity (dx, dy)
        [4]   Direction
        [5]   Is Dead
        [6]   Is Shooting
        [7]   Distance (normalized)
        [8]   Angle to player
        [9-11] Reserved

        Args:
            player: Other player state
            own_player: Own player state for relative calculations

        Returns:
            Normalized observation vector of shape (12,)
        """
        obs = np.zeros(self.config.per_player_size, dtype=np.float32)

        # Relative position [0-1]
        rel_x = player.x - own_player.x
        rel_y = player.y - own_player.y
        rel_pos_x, rel_pos_y = normalize_relative_position(
            rel_x, rel_y, self.constants.ROOM_WIDTH
        )
        obs[0] = rel_pos_x
        obs[1] = rel_pos_y

        # Velocity [2-3]
        vel_x, vel_y = normalize_velocity(
            player.dx,
            player.dy,
            self.constants.MAX_PLAYER_VELOCITY,
        )
        obs[2] = vel_x
        obs[3] = vel_y

        # Direction [4]
        obs[4] = normalize_angle(player.direction)

        # Is Dead [5]
        obs[5] = normalize_boolean(player.dead)

        # Is Shooting [6]
        obs[6] = normalize_boolean(player.shooting)

        # Distance [7]
        distance = calculate_distance(own_player.x, own_player.y, player.x, player.y)
        obs[7] = normalize_distance(distance, self.constants.ROOM_WIDTH)

        # Angle to player [8]
        angle = calculate_angle_to(own_player.x, own_player.y, player.x, player.y)
        obs[8] = normalize_angle(angle)

        # Reserved [9-11]
        obs[9] = 0.0
        obs[10] = 0.0
        obs[11] = 0.0

        return obs

    def _normalize_arrow(
        self,
        arrow: ArrowState,
        own_player: PlayerState,
        own_player_id: str,
    ) -> NDArray[np.float32]:
        """Normalize arrow state relative to own player.

        Arrow observation (8 values):
        [0-1] Relative position (x, y)
        [2-3] Velocity (dx, dy)
        [4]   Direction
        [5]   Is Grounded
        [6]   Is Own Arrow (1 if shot by this player)
        [7]   Distance

        Args:
            arrow: Arrow state
            own_player: Own player state for relative calculations
            own_player_id: ID of own player (for ownership check)

        Returns:
            Normalized observation vector of shape (8,)
        """
        obs = np.zeros(self.config.per_arrow_size, dtype=np.float32)

        # Relative position [0-1]
        rel_x = arrow.x - own_player.x
        rel_y = arrow.y - own_player.y
        rel_pos_x, rel_pos_y = normalize_relative_position(
            rel_x, rel_y, self.constants.ROOM_WIDTH
        )
        obs[0] = rel_pos_x
        obs[1] = rel_pos_y

        # Velocity [2-3]
        vel_x, vel_y = normalize_velocity(
            arrow.dx,
            arrow.dy,
            self.constants.MAX_ARROW_VELOCITY,
        )
        obs[2] = vel_x
        obs[3] = vel_y

        # Direction [4]
        # Arrow direction can be None if not set, default to 0
        arrow_direction = arrow.direction if arrow.direction is not None else 0.0
        obs[4] = normalize_angle(arrow_direction)

        # Is Grounded [5]
        obs[5] = normalize_boolean(arrow.grounded)

        # Is Own Arrow [6]
        is_own_arrow = arrow.owner_id == own_player_id if arrow.owner_id else False
        obs[6] = normalize_boolean(is_own_arrow)

        # Distance [7]
        distance = calculate_distance(own_player.x, own_player.y, arrow.x, arrow.y)
        obs[7] = normalize_distance(distance, self.constants.ROOM_WIDTH)

        return obs

    def get_gymnasium_space(self):
        """Get gymnasium Box space for observation.

        Returns:
            gymnasium.spaces.Box with bounds [-1, 1]
        """
        import gymnasium as gym

        return gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.config.total_size,),
            dtype=np.float32,
        )
