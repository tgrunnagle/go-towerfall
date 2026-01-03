"""Tests for observation space normalization."""

import math

import numpy as np
import pytest

from bot.models import ArrowState, GameState, PlayerState
from bot.observation import (
    ObservationBuilder,
    ObservationConfig,
    normalize_angle,
    normalize_boolean,
    normalize_position,
    normalize_velocity,
)
from bot.observation.normalizer import (
    NORMALIZATION_CONSTANTS,
    calculate_angle_to,
    calculate_distance,
    normalize_count,
    normalize_distance,
    normalize_health,
    normalize_relative_position,
)

# =============================================================================
# Normalizer Utility Tests
# =============================================================================


class TestNormalizationConstants:
    """Test NormalizationConstants values."""

    def test_room_dimensions(self):
        """Test room dimensions match expected values."""
        assert NORMALIZATION_CONSTANTS.ROOM_WIDTH == 800.0
        assert NORMALIZATION_CONSTANTS.ROOM_HEIGHT == 800.0

    def test_max_velocities(self):
        """Test max velocity values."""
        assert NORMALIZATION_CONSTANTS.MAX_PLAYER_VELOCITY == 600.0
        assert NORMALIZATION_CONSTANTS.MAX_ARROW_VELOCITY == 1000.0

    def test_player_constants(self):
        """Test player-related constants."""
        assert NORMALIZATION_CONSTANTS.MAX_HEALTH == 100.0
        assert NORMALIZATION_CONSTANTS.MAX_ARROWS == 4
        assert NORMALIZATION_CONSTANTS.MAX_JUMPS == 2

    def test_constants_frozen(self):
        """Test that constants are immutable."""
        with pytest.raises(Exception):  # FrozenInstanceError
            NORMALIZATION_CONSTANTS.ROOM_WIDTH = 1000.0  # type: ignore


class TestNormalizePosition:
    """Test normalize_position function."""

    def test_center_position(self):
        """Test position at center of room."""
        x, y = normalize_position(400, 400)
        assert x == pytest.approx(0.0)
        assert y == pytest.approx(0.0)

    def test_origin_position(self):
        """Test position at origin."""
        x, y = normalize_position(0, 0)
        assert x == pytest.approx(-1.0)
        assert y == pytest.approx(-1.0)

    def test_max_position(self):
        """Test position at max room boundary."""
        x, y = normalize_position(800, 800)
        assert x == pytest.approx(1.0)
        assert y == pytest.approx(1.0)

    def test_quarter_position(self):
        """Test position at quarter point."""
        x, y = normalize_position(200, 600)
        assert x == pytest.approx(-0.5)
        assert y == pytest.approx(0.5)

    def test_clipping_over_bounds(self):
        """Test values are clipped to [-1, 1]."""
        x, y = normalize_position(1000, 1000)
        assert x == 1.0
        assert y == 1.0

    def test_clipping_under_bounds(self):
        """Test negative values are clipped to [-1, 1]."""
        x, y = normalize_position(-100, -100)
        assert x == -1.0
        assert y == -1.0


class TestNormalizeRelativePosition:
    """Test normalize_relative_position function."""

    def test_same_position(self):
        """Test relative position when at same point."""
        x, y = normalize_relative_position(0, 0)
        assert x == pytest.approx(0.0)
        assert y == pytest.approx(0.0)

    def test_max_relative_distance(self):
        """Test at maximum distance."""
        x, y = normalize_relative_position(800, 0)
        assert x == pytest.approx(1.0)
        assert y == pytest.approx(0.0)

    def test_negative_relative_position(self):
        """Test negative relative position."""
        x, y = normalize_relative_position(-400, -400)
        assert x == pytest.approx(-0.5)
        assert y == pytest.approx(-0.5)

    def test_clipping(self):
        """Test clipping for out-of-range values."""
        x, y = normalize_relative_position(1600, -1600)
        assert x == 1.0
        assert y == -1.0


class TestNormalizeVelocity:
    """Test normalize_velocity function."""

    def test_zero_velocity(self):
        """Test zero velocity."""
        dx, dy = normalize_velocity(0, 0)
        assert dx == pytest.approx(0.0)
        assert dy == pytest.approx(0.0)

    def test_max_velocity(self):
        """Test max velocity normalization."""
        dx, dy = normalize_velocity(600, -600)
        assert dx == pytest.approx(1.0)
        assert dy == pytest.approx(-1.0)

    def test_half_velocity(self):
        """Test half max velocity."""
        dx, dy = normalize_velocity(300, 300)
        assert dx == pytest.approx(0.5)
        assert dy == pytest.approx(0.5)

    def test_custom_max_velocity(self):
        """Test with custom max velocity (arrows)."""
        dx, dy = normalize_velocity(500, -500, max_velocity=1000)
        assert dx == pytest.approx(0.5)
        assert dy == pytest.approx(-0.5)

    def test_clipping(self):
        """Test velocity clipping."""
        dx, dy = normalize_velocity(1200, -1200)
        assert dx == 1.0
        assert dy == -1.0


class TestNormalizeAngle:
    """Test normalize_angle function."""

    def test_zero_angle(self):
        """Test angle at 0."""
        result = normalize_angle(0)
        assert result == pytest.approx(-1.0)

    def test_pi_angle(self):
        """Test angle at π."""
        result = normalize_angle(math.pi)
        assert result == pytest.approx(0.0)

    def test_two_pi_angle(self):
        """Test angle at 2π (wraps to 0)."""
        result = normalize_angle(2 * math.pi)
        assert result == pytest.approx(-1.0)

    def test_half_pi_angle(self):
        """Test angle at π/2."""
        result = normalize_angle(math.pi / 2)
        assert result == pytest.approx(-0.5)

    def test_three_half_pi_angle(self):
        """Test angle at 3π/2."""
        result = normalize_angle(3 * math.pi / 2)
        assert result == pytest.approx(0.5)

    def test_negative_angle_wraps(self):
        """Test negative angle wraps correctly."""
        result = normalize_angle(-math.pi / 2)
        # -π/2 wraps to 3π/2
        assert result == pytest.approx(0.5)


class TestNormalizeBoolean:
    """Test normalize_boolean function."""

    def test_true(self):
        """Test True converts to 1.0."""
        assert normalize_boolean(True) == 1.0

    def test_false(self):
        """Test False converts to -1.0."""
        assert normalize_boolean(False) == -1.0


class TestNormalizeDistance:
    """Test normalize_distance function."""

    def test_zero_distance(self):
        """Test zero distance."""
        result = normalize_distance(0)
        assert result == pytest.approx(0.0)

    def test_max_distance(self):
        """Test max distance."""
        result = normalize_distance(800)
        assert result == pytest.approx(1.0)

    def test_half_distance(self):
        """Test half distance."""
        result = normalize_distance(400)
        assert result == pytest.approx(0.5)

    def test_over_max_clips(self):
        """Test distance over max clips to 1.0."""
        result = normalize_distance(1600)
        assert result == 1.0

    def test_negative_uses_absolute(self):
        """Test negative distance uses absolute value."""
        result = normalize_distance(-400)
        assert result == pytest.approx(0.5)


class TestNormalizeCount:
    """Test normalize_count function."""

    def test_zero_count(self):
        """Test count at 0."""
        result = normalize_count(0, 4)
        assert result == pytest.approx(-1.0)

    def test_max_count(self):
        """Test count at max."""
        result = normalize_count(4, 4)
        assert result == pytest.approx(1.0)

    def test_half_count(self):
        """Test count at half."""
        result = normalize_count(2, 4)
        assert result == pytest.approx(0.0)

    def test_zero_max_count(self):
        """Test with zero max count returns 0."""
        result = normalize_count(5, 0)
        assert result == 0.0


class TestNormalizeHealth:
    """Test normalize_health function."""

    def test_full_health(self):
        """Test full health."""
        result = normalize_health(100)
        assert result == pytest.approx(1.0)

    def test_zero_health(self):
        """Test zero health."""
        result = normalize_health(0)
        assert result == pytest.approx(-1.0)

    def test_half_health(self):
        """Test half health."""
        result = normalize_health(50)
        assert result == pytest.approx(0.0)


class TestCalculateDistance:
    """Test calculate_distance function."""

    def test_same_point(self):
        """Test distance between same point is 0."""
        result = calculate_distance(100, 100, 100, 100)
        assert result == pytest.approx(0.0)

    def test_horizontal_distance(self):
        """Test horizontal distance."""
        result = calculate_distance(0, 0, 100, 0)
        assert result == pytest.approx(100.0)

    def test_vertical_distance(self):
        """Test vertical distance."""
        result = calculate_distance(0, 0, 0, 100)
        assert result == pytest.approx(100.0)

    def test_diagonal_distance(self):
        """Test diagonal distance (3-4-5 triangle)."""
        result = calculate_distance(0, 0, 3, 4)
        assert result == pytest.approx(5.0)


class TestCalculateAngleTo:
    """Test calculate_angle_to function."""

    def test_angle_right(self):
        """Test angle to point on the right (0 radians)."""
        result = calculate_angle_to(0, 0, 100, 0)
        assert result == pytest.approx(0.0)

    def test_angle_up(self):
        """Test angle to point above (π/2 radians)."""
        result = calculate_angle_to(0, 0, 0, 100)
        assert result == pytest.approx(math.pi / 2)

    def test_angle_left(self):
        """Test angle to point on the left (π radians)."""
        result = calculate_angle_to(0, 0, -100, 0)
        assert result == pytest.approx(math.pi)

    def test_angle_down(self):
        """Test angle to point below (3π/2 radians)."""
        result = calculate_angle_to(0, 0, 0, -100)
        assert result == pytest.approx(3 * math.pi / 2)


# =============================================================================
# ObservationConfig Tests
# =============================================================================


class TestObservationConfig:
    """Test ObservationConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ObservationConfig()
        assert config.max_other_players == 3
        assert config.max_tracked_arrows == 8
        assert config.include_map is True

    def test_own_player_size(self):
        """Test own player observation size."""
        config = ObservationConfig()
        assert config.own_player_size == 14

    def test_per_player_size(self):
        """Test per-player observation size."""
        config = ObservationConfig()
        assert config.per_player_size == 12

    def test_per_arrow_size(self):
        """Test per-arrow observation size."""
        config = ObservationConfig()
        assert config.per_arrow_size == 8

    def test_other_players_size(self):
        """Test total other players section size."""
        config = ObservationConfig()
        assert config.other_players_size == 3 * 12  # 36

    def test_arrows_size(self):
        """Test total arrows section size."""
        config = ObservationConfig()
        assert config.arrows_size == 8 * 8  # 64

    def test_map_size_enabled(self):
        """Test map size when enabled (default)."""
        config = ObservationConfig()
        assert config.map_size == 20 * 15  # 300

    def test_map_size_disabled(self):
        """Test map size when disabled."""
        config = ObservationConfig(include_map=False)
        assert config.map_size == 0

    def test_total_size_without_map(self):
        """Test total observation vector size without map."""
        config = ObservationConfig(include_map=False)
        # 14 + (3 * 12) + (8 * 8) = 14 + 36 + 64 = 114
        assert config.total_size == 114

    def test_total_size_with_map(self):
        """Test total observation vector size with map (default)."""
        config = ObservationConfig()
        # 14 + (3 * 12) + (8 * 8) + (20 * 15) = 14 + 36 + 64 + 300 = 414
        assert config.total_size == 414

    def test_custom_config(self):
        """Test custom configuration without map."""
        config = ObservationConfig(
            max_other_players=5, max_tracked_arrows=10, include_map=False
        )
        assert config.max_other_players == 5
        assert config.max_tracked_arrows == 10
        # 14 + (5 * 12) + (10 * 8) = 14 + 60 + 80 = 154
        assert config.total_size == 154

    def test_custom_config_with_map(self):
        """Test custom configuration with map."""
        config = ObservationConfig(max_other_players=5, max_tracked_arrows=10)
        assert config.max_other_players == 5
        assert config.max_tracked_arrows == 10
        # 14 + (5 * 12) + (10 * 8) + (20 * 15) = 14 + 60 + 80 + 300 = 454
        assert config.total_size == 454


# =============================================================================
# ObservationBuilder Tests
# =============================================================================


def create_player_state(
    player_id: str = "player1",
    x: float = 400,
    y: float = 400,
    dx: float = 0,
    dy: float = 0,
    direction: float = 0,
    health: int = 100,
    dead: bool = False,
    shooting: bool = False,
    jump_count: int = 0,
    arrow_count: int = 4,
) -> PlayerState:
    """Create a PlayerState for testing.

    Uses Pydantic model_validate with dict to match the server's JSON format.
    """
    return PlayerState.model_validate(
        {
            "id": player_id,
            "objectType": "player",
            "name": f"TestPlayer_{player_id}",
            "x": x,
            "y": y,
            "dx": dx,
            "dy": dy,
            "dir": direction,
            "rad": 20.0,
            "h": health,
            "dead": dead,
            "sht": shooting,
            "shts": 0.0,
            "jc": jump_count,
            "ac": arrow_count,
        }
    )


def create_arrow_state(
    arrow_id: str = "arrow1",
    x: float = 500,
    y: float = 400,
    dx: float = 100,
    dy: float = 0,
    direction: float = 0,
    grounded: bool = False,
    destroyed: bool = False,
) -> ArrowState:
    """Create an ArrowState for testing.

    Uses Pydantic model_validate with dict to match the server's JSON format.
    """
    return ArrowState.model_validate(
        {
            "id": arrow_id,
            "objectType": "arrow",
            "x": x,
            "y": y,
            "dx": dx,
            "dy": dy,
            "dir": direction,
            "ag": grounded,
            "d": destroyed,
            "dAtX": 0.0,
            "dAtY": 0.0,
        }
    )


def create_game_state(
    players: dict[str, PlayerState] | None = None,
    arrows: dict[str, ArrowState] | None = None,
) -> GameState:
    """Create a GameState for testing."""
    return GameState(
        players=players or {},
        arrows=arrows or {},
        blocks={},
        bullets={},
        canvas_size_x=800,
        canvas_size_y=800,
    )


class TestObservationBuilder:
    """Test ObservationBuilder class."""

    def test_init_default_config(self):
        """Test initialization with default config (includes map)."""
        builder = ObservationBuilder()
        assert builder.config.total_size == 414  # 114 + 300 (map)

    def test_init_config_without_map(self):
        """Test initialization without map encoding."""
        config = ObservationConfig(include_map=False)
        builder = ObservationBuilder(config=config)
        assert builder.config.total_size == 114

    def test_init_custom_config(self):
        """Test initialization with custom config."""
        config = ObservationConfig(
            max_other_players=2, max_tracked_arrows=4, include_map=False
        )
        builder = ObservationBuilder(config=config)
        assert builder.config.max_other_players == 2
        assert builder.config.max_tracked_arrows == 4

    def test_build_missing_player_raises(self):
        """Test that build raises if player not found."""
        config = ObservationConfig(include_map=False)
        builder = ObservationBuilder(config=config)
        game_state = create_game_state()

        with pytest.raises(ValueError, match="not found"):
            builder.build(game_state, "missing_player")

    def test_build_returns_correct_shape(self):
        """Test that build returns correct observation shape without map."""
        config = ObservationConfig(include_map=False)
        builder = ObservationBuilder(config=config)
        player = create_player_state("player1")
        game_state = create_game_state(players={"player1": player})

        obs = builder.build(game_state, "player1")

        assert obs.shape == (114,)
        assert obs.dtype == np.float32

    def test_build_returns_correct_shape_with_map(self):
        """Test that build returns correct observation shape with map."""
        builder = ObservationBuilder()  # Default includes map
        player = create_player_state("player1")
        game_state = create_game_state(players={"player1": player})

        obs = builder.build(game_state, "player1")

        assert obs.shape == (414,)  # 114 + 300 (map)
        assert obs.dtype == np.float32

    def test_build_values_in_range(self):
        """Test that all observation values are in [-1, 1]."""
        builder = ObservationBuilder()
        player = create_player_state("player1", x=100, y=700, dx=300, dy=-300)
        game_state = create_game_state(players={"player1": player})

        obs = builder.build(game_state, "player1")

        assert np.all(obs >= -1.0)
        assert np.all(obs <= 1.0)

    def test_build_empty_game_state(self):
        """Test observation for game with only own player (no arrows/others)."""
        config = ObservationConfig(include_map=False)
        builder = ObservationBuilder(config=config)
        player = create_player_state("player1")
        game_state = create_game_state(players={"player1": player})

        obs = builder.build(game_state, "player1")

        # Own player section should have values
        own_player_obs = obs[: builder.config.own_player_size]
        assert not np.allclose(own_player_obs, 0)

        # Other players and arrows should be zero-padded
        other_start = builder.config.own_player_size
        other_end = other_start + builder.config.other_players_size
        other_players_obs = obs[other_start:other_end]
        assert np.allclose(other_players_obs, 0)

        arrows_obs = obs[other_end:]
        assert np.allclose(arrows_obs, 0)

    def test_build_with_other_players(self):
        """Test observation with other players."""
        builder = ObservationBuilder()
        player1 = create_player_state("player1", x=400, y=400)
        player2 = create_player_state("player2", x=500, y=400)  # 100px away
        player3 = create_player_state("player3", x=600, y=400)  # 200px away
        game_state = create_game_state(
            players={
                "player1": player1,
                "player2": player2,
                "player3": player3,
            }
        )

        obs = builder.build(game_state, "player1")

        # First other player slot should have non-zero values
        other_start = builder.config.own_player_size
        first_player_obs = obs[
            other_start : other_start + builder.config.per_player_size
        ]
        assert not np.allclose(first_player_obs, 0)

    def test_build_players_sorted_by_distance(self):
        """Test that other players are sorted by distance."""
        builder = ObservationBuilder()
        player1 = create_player_state("player1", x=400, y=400)
        player2 = create_player_state("player2", x=700, y=400)  # 300px away
        player3 = create_player_state("player3", x=450, y=400)  # 50px away (closer)
        game_state = create_game_state(
            players={
                "player1": player1,
                "player2": player2,
                "player3": player3,
            }
        )

        obs = builder.build(game_state, "player1")

        # First other player should be player3 (closer)
        other_start = builder.config.own_player_size
        first_player_obs = obs[
            other_start : other_start + builder.config.per_player_size
        ]

        # Relative position for player3: (50, 0) normalized to (50/800, 0) = (0.0625, 0)
        assert first_player_obs[0] == pytest.approx(50 / 800)
        assert first_player_obs[1] == pytest.approx(0.0)

    def test_build_with_arrows(self):
        """Test observation with arrows."""
        builder = ObservationBuilder()
        player = create_player_state("player1", x=400, y=400)
        arrow = create_arrow_state("arrow1", x=500, y=400, dx=200, dy=0)
        game_state = create_game_state(
            players={"player1": player},
            arrows={"arrow1": arrow},
        )

        obs = builder.build(game_state, "player1")

        # Arrow section should have non-zero values
        arrows_start = (
            builder.config.own_player_size + builder.config.other_players_size
        )
        first_arrow_obs = obs[
            arrows_start : arrows_start + builder.config.per_arrow_size
        ]
        assert not np.allclose(first_arrow_obs, 0)

    def test_build_destroyed_arrows_excluded(self):
        """Test that destroyed arrows are excluded."""
        builder = ObservationBuilder()
        player = create_player_state("player1", x=400, y=400)
        arrow1 = create_arrow_state("arrow1", x=500, y=400, destroyed=True)
        arrow2 = create_arrow_state("arrow2", x=600, y=400, destroyed=False)
        game_state = create_game_state(
            players={"player1": player},
            arrows={"arrow1": arrow1, "arrow2": arrow2},
        )

        obs = builder.build(game_state, "player1")

        # Only arrow2 should be in observation
        arrows_start = (
            builder.config.own_player_size + builder.config.other_players_size
        )
        first_arrow_obs = obs[
            arrows_start : arrows_start + builder.config.per_arrow_size
        ]

        # Relative position for arrow2: (200, 0) normalized
        assert first_arrow_obs[0] == pytest.approx(200 / 800)
        assert first_arrow_obs[1] == pytest.approx(0.0)

    def test_build_arrows_sorted_by_distance(self):
        """Test that arrows are sorted by distance."""
        builder = ObservationBuilder()
        player = create_player_state("player1", x=400, y=400)
        arrow1 = create_arrow_state("arrow1", x=700, y=400)  # 300px away
        arrow2 = create_arrow_state("arrow2", x=450, y=400)  # 50px away (closer)
        game_state = create_game_state(
            players={"player1": player},
            arrows={"arrow1": arrow1, "arrow2": arrow2},
        )

        obs = builder.build(game_state, "player1")

        # First arrow should be arrow2 (closer)
        arrows_start = (
            builder.config.own_player_size + builder.config.other_players_size
        )
        first_arrow_obs = obs[
            arrows_start : arrows_start + builder.config.per_arrow_size
        ]

        # Relative position for arrow2: (50, 0) normalized
        assert first_arrow_obs[0] == pytest.approx(50 / 800)
        assert first_arrow_obs[1] == pytest.approx(0.0)

    def test_build_max_entities_respected(self):
        """Test that max entities limits are respected."""
        config = ObservationConfig(
            max_other_players=2, max_tracked_arrows=2, include_map=False
        )
        builder = ObservationBuilder(config=config)

        player1 = create_player_state("player1", x=400, y=400)
        player2 = create_player_state("player2", x=450, y=400)
        player3 = create_player_state("player3", x=500, y=400)
        player4 = create_player_state("player4", x=550, y=400)

        arrow1 = create_arrow_state("arrow1", x=460, y=400)
        arrow2 = create_arrow_state("arrow2", x=470, y=400)
        arrow3 = create_arrow_state("arrow3", x=480, y=400)

        game_state = create_game_state(
            players={
                "player1": player1,
                "player2": player2,
                "player3": player3,
                "player4": player4,
            },
            arrows={
                "arrow1": arrow1,
                "arrow2": arrow2,
                "arrow3": arrow3,
            },
        )

        obs = builder.build(game_state, "player1")

        # Should only have space for 2 other players and 2 arrows
        expected_size = 14 + (2 * 12) + (2 * 8)
        assert obs.shape == (expected_size,)

    def test_own_player_position_normalized(self):
        """Test own player position normalization."""
        builder = ObservationBuilder()
        player = create_player_state("player1", x=200, y=600)
        game_state = create_game_state(players={"player1": player})

        obs = builder.build(game_state, "player1")

        # Position should be normalized: (200/400 - 1, 600/400 - 1) = (-0.5, 0.5)
        assert obs[0] == pytest.approx(-0.5)
        assert obs[1] == pytest.approx(0.5)

    def test_own_player_velocity_normalized(self):
        """Test own player velocity normalization."""
        builder = ObservationBuilder()
        player = create_player_state("player1", dx=300, dy=-300)
        game_state = create_game_state(players={"player1": player})

        obs = builder.build(game_state, "player1")

        # Velocity should be normalized: (300/600, -300/600) = (0.5, -0.5)
        assert obs[2] == pytest.approx(0.5)
        assert obs[3] == pytest.approx(-0.5)

    def test_own_player_direction_normalized(self):
        """Test own player direction normalization."""
        builder = ObservationBuilder()
        player = create_player_state("player1", direction=math.pi)
        game_state = create_game_state(players={"player1": player})

        obs = builder.build(game_state, "player1")

        # Direction π should normalize to 0.0
        assert obs[4] == pytest.approx(0.0)

    def test_own_player_health_normalized(self):
        """Test own player health normalization."""
        builder = ObservationBuilder()
        player = create_player_state("player1", health=75)
        game_state = create_game_state(players={"player1": player})

        obs = builder.build(game_state, "player1")

        # Health 75 should normalize to: (75/50) - 1 = 0.5
        assert obs[5] == pytest.approx(0.5)

    def test_own_player_dead_normalized(self):
        """Test own player dead flag normalization."""
        builder = ObservationBuilder()
        player = create_player_state("player1", dead=True)
        game_state = create_game_state(players={"player1": player})

        obs = builder.build(game_state, "player1")

        # Dead=True should normalize to 1.0
        assert obs[6] == pytest.approx(1.0)

    def test_own_player_shooting_normalized(self):
        """Test own player shooting flag normalization."""
        builder = ObservationBuilder()
        player = create_player_state("player1", shooting=True)
        game_state = create_game_state(players={"player1": player})

        obs = builder.build(game_state, "player1")

        # Shooting=True should normalize to 1.0
        assert obs[7] == pytest.approx(1.0)

    def test_own_player_jump_count_normalized(self):
        """Test own player jump count normalization."""
        builder = ObservationBuilder()
        player = create_player_state("player1", jump_count=1)
        game_state = create_game_state(players={"player1": player})

        obs = builder.build(game_state, "player1")

        # Jump count 1 of max 2: (1/1) - 1 = 0.0
        assert obs[8] == pytest.approx(0.0)

    def test_own_player_arrow_count_normalized(self):
        """Test own player arrow count normalization."""
        builder = ObservationBuilder()
        player = create_player_state("player1", arrow_count=2)
        game_state = create_game_state(players={"player1": player})

        obs = builder.build(game_state, "player1")

        # Arrow count 2 of max 4: (2/2) - 1 = 0.0
        assert obs[9] == pytest.approx(0.0)

    def test_boundary_values_player_at_edges(self):
        """Test player at room edges produces valid observations."""
        builder = ObservationBuilder()

        # Test at origin
        player1 = create_player_state("player1", x=0, y=0)
        game_state = create_game_state(players={"player1": player1})
        obs = builder.build(game_state, "player1")
        assert np.all(obs >= -1.0)
        assert np.all(obs <= 1.0)

        # Test at max corner
        player2 = create_player_state("player2", x=800, y=800)
        game_state = create_game_state(players={"player2": player2})
        obs = builder.build(game_state, "player2")
        assert np.all(obs >= -1.0)
        assert np.all(obs <= 1.0)

    def test_boundary_values_max_velocity(self):
        """Test player at max velocity produces valid observations."""
        builder = ObservationBuilder()
        player = create_player_state("player1", dx=600, dy=-600)
        game_state = create_game_state(players={"player1": player})

        obs = builder.build(game_state, "player1")

        assert obs[2] == pytest.approx(1.0)
        assert obs[3] == pytest.approx(-1.0)

    def test_dead_player_handling(self):
        """Test dead player observations are handled correctly."""
        builder = ObservationBuilder()
        player1 = create_player_state("player1", x=400, y=400, dead=False)
        player2 = create_player_state("player2", x=500, y=400, dead=True)
        game_state = create_game_state(players={"player1": player1, "player2": player2})

        obs = builder.build(game_state, "player1")

        # Dead player should still be in observation
        other_start = builder.config.own_player_size
        first_player_obs = obs[
            other_start : other_start + builder.config.per_player_size
        ]

        # Check dead flag is normalized to 1.0
        assert first_player_obs[5] == pytest.approx(1.0)

    def test_full_game_state(self):
        """Test with maximum players and arrows (without map)."""
        config = ObservationConfig(
            max_other_players=3, max_tracked_arrows=8, include_map=False
        )
        builder = ObservationBuilder(config=config)

        players = {"player1": create_player_state("player1", x=400, y=400)}
        for i in range(3):
            players[f"other{i}"] = create_player_state(
                f"other{i}", x=450 + i * 50, y=400
            )

        arrows = {}
        for i in range(8):
            arrows[f"arrow{i}"] = create_arrow_state(f"arrow{i}", x=460 + i * 10, y=400)

        game_state = create_game_state(players=players, arrows=arrows)
        obs = builder.build(game_state, "player1")

        # All sections should have values
        assert obs.shape == (114,)
        assert not np.allclose(obs, 0)
        assert np.all(obs >= -1.0)
        assert np.all(obs <= 1.0)

    def test_full_game_state_with_map(self):
        """Test with maximum players, arrows, and map."""
        config = ObservationConfig(max_other_players=3, max_tracked_arrows=8)
        builder = ObservationBuilder(config=config)

        players = {"player1": create_player_state("player1", x=400, y=400)}
        for i in range(3):
            players[f"other{i}"] = create_player_state(
                f"other{i}", x=450 + i * 50, y=400
            )

        arrows = {}
        for i in range(8):
            arrows[f"arrow{i}"] = create_arrow_state(f"arrow{i}", x=460 + i * 10, y=400)

        game_state = create_game_state(players=players, arrows=arrows)
        obs = builder.build(game_state, "player1")

        # All sections should have values (114 + 300 map)
        assert obs.shape == (414,)
        assert not np.allclose(obs, 0)
        assert np.all(obs >= -1.0)
        assert np.all(obs <= 1.0)


class TestObservationBuilderGymnasium:
    """Test ObservationBuilder gymnasium integration."""

    def test_get_gymnasium_space(self):
        """Test gymnasium space creation with default config (includes map)."""
        builder = ObservationBuilder()
        space = builder.get_gymnasium_space()

        assert space.shape == (414,)  # 114 + 300 (map)
        assert space.low.min() == -1.0
        assert space.high.max() == 1.0
        assert space.dtype == np.float32

    def test_get_gymnasium_space_without_map(self):
        """Test gymnasium space creation without map."""
        config = ObservationConfig(include_map=False)
        builder = ObservationBuilder(config=config)
        space = builder.get_gymnasium_space()

        assert space.shape == (114,)
        assert space.low.min() == -1.0
        assert space.high.max() == 1.0
        assert space.dtype == np.float32

    def test_get_gymnasium_space_custom_config(self):
        """Test gymnasium space with custom config (without map)."""
        config = ObservationConfig(
            max_other_players=5, max_tracked_arrows=10, include_map=False
        )
        builder = ObservationBuilder(config=config)
        space = builder.get_gymnasium_space()

        expected_size = 14 + (5 * 12) + (10 * 8)
        assert space.shape == (expected_size,)

    def test_get_gymnasium_space_custom_config_with_map(self):
        """Test gymnasium space with custom config (with map)."""
        config = ObservationConfig(max_other_players=5, max_tracked_arrows=10)
        builder = ObservationBuilder(config=config)
        space = builder.get_gymnasium_space()

        expected_size = 14 + (5 * 12) + (10 * 8) + (20 * 15)
        assert space.shape == (expected_size,)

    def test_observation_in_gymnasium_space(self):
        """Test that built observations are in gymnasium space."""
        builder = ObservationBuilder()
        player = create_player_state("player1")
        game_state = create_game_state(players={"player1": player})

        obs = builder.build(game_state, "player1")
        space = builder.get_gymnasium_space()

        assert space.contains(obs)
