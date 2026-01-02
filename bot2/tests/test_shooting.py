"""Unit tests for rule-based bot aiming and shooting behaviors."""

import math
from unittest.mock import patch

import pytest

from bot.bots import RuleBasedBot, RuleBasedBotConfig, ShootingConfig
from bot.bots.shooting_utils import (
    GRAVITY_PX_PER_SEC2,
    calculate_aim_point,
    calculate_arrow_speed,
    calculate_max_arrow_speed,
    calculate_optimal_power,
    compensate_for_gravity,
    should_release_shot,
    should_shoot,
)
from bot.models import GAME_CONSTANTS, GameState, GameUpdate, PlayerState


class TestCalculateMaxArrowSpeed:
    """Tests for calculate_max_arrow_speed function."""

    def test_max_arrow_speed(self) -> None:
        """Maximum arrow speed should match expected value from constants."""
        max_speed = calculate_max_arrow_speed()
        # sqrt(2 * 1.0 * 100.0 / 0.1) * 20.0 = sqrt(2000) * 20 ≈ 894.4
        expected = (
            math.sqrt(
                2
                * 1.0
                * GAME_CONSTANTS.ARROW_MAX_POWER_NEWTON
                / GAME_CONSTANTS.ARROW_MASS_KG
            )
            * GAME_CONSTANTS.PX_PER_METER
        )
        assert max_speed == pytest.approx(expected)
        assert max_speed == pytest.approx(894.4, rel=0.01)


class TestCalculateArrowSpeed:
    """Tests for calculate_arrow_speed function."""

    def test_arrow_speed_at_full_power(self) -> None:
        """Arrow speed at full power should equal max arrow speed."""
        full_power_speed = calculate_arrow_speed(1.0)
        max_speed = calculate_max_arrow_speed()
        assert full_power_speed == pytest.approx(max_speed)

    def test_arrow_speed_at_zero_power(self) -> None:
        """Arrow speed at zero power should be zero."""
        zero_power_speed = calculate_arrow_speed(0.0)
        assert zero_power_speed == 0.0

    def test_arrow_speed_at_half_power(self) -> None:
        """Arrow speed at half power should be sqrt(0.5) times max speed."""
        half_power_speed = calculate_arrow_speed(0.5)
        max_speed = calculate_max_arrow_speed()
        expected = max_speed * math.sqrt(0.5)
        assert half_power_speed == pytest.approx(expected)

    def test_arrow_speed_clamped_to_valid_range(self) -> None:
        """Power ratio should be clamped to 0.0-1.0 range."""
        # Negative power should be clamped to 0
        neg_power_speed = calculate_arrow_speed(-0.5)
        assert neg_power_speed == 0.0

        # Power > 1 should be clamped to 1.0
        over_power_speed = calculate_arrow_speed(1.5)
        max_speed = calculate_max_arrow_speed()
        assert over_power_speed == pytest.approx(max_speed)


class TestCalculateAimPoint:
    """Tests for calculate_aim_point function."""

    def test_aim_point_stationary_target(self) -> None:
        """Aim point should be target position for stationary target."""
        aim_x, aim_y = calculate_aim_point(
            own_x=100,
            own_y=100,
            target_x=300,
            target_y=100,
            target_dx=0,
            target_dy=0,
            arrow_speed=500,
        )
        assert aim_x == 300
        assert aim_y == 100

    def test_aim_point_moving_target_horizontal(self) -> None:
        """Aim point should lead moving target horizontally."""
        aim_x, aim_y = calculate_aim_point(
            own_x=100,
            own_y=100,
            target_x=200,
            target_y=100,
            target_dx=100,  # Moving right at 100 px/sec
            target_dy=0,
            arrow_speed=500,
        )
        # Distance = 100px, travel time ≈ 0.2s, target moves ≈ 20px
        assert aim_x > 200  # Should aim ahead of target
        assert aim_y == pytest.approx(100)

    def test_aim_point_moving_target_vertical(self) -> None:
        """Aim point should lead moving target vertically."""
        aim_x, aim_y = calculate_aim_point(
            own_x=100,
            own_y=100,
            target_x=100,
            target_y=200,
            target_dx=0,
            target_dy=50,  # Moving down at 50 px/sec
            arrow_speed=500,
        )
        # Distance = 100px, travel time ≈ 0.2s, target moves ≈ 10px down
        assert aim_x == pytest.approx(100)
        assert aim_y > 200  # Should aim ahead of target

    def test_aim_point_moving_target_diagonal(self) -> None:
        """Aim point should lead moving target diagonally."""
        aim_x, aim_y = calculate_aim_point(
            own_x=100,
            own_y=100,
            target_x=200,
            target_y=200,
            target_dx=50,  # Moving diagonally
            target_dy=50,
            arrow_speed=500,
        )
        # Distance ≈ 141px, travel time ≈ 0.28s
        assert aim_x > 200
        assert aim_y > 200

    def test_aim_point_zero_arrow_speed(self) -> None:
        """Aim point should be target position if arrow speed is zero."""
        aim_x, aim_y = calculate_aim_point(
            own_x=100,
            own_y=100,
            target_x=300,
            target_y=200,
            target_dx=100,
            target_dy=50,
            arrow_speed=0,
        )
        assert aim_x == 300
        assert aim_y == 200

    def test_aim_point_same_position(self) -> None:
        """Aim point should be target position if bot and target overlap."""
        aim_x, aim_y = calculate_aim_point(
            own_x=100,
            own_y=100,
            target_x=100,
            target_y=100,
            target_dx=50,
            target_dy=50,
            arrow_speed=500,
        )
        assert aim_x == 100
        assert aim_y == 100


class TestCompensateForGravity:
    """Tests for compensate_for_gravity function."""

    def test_gravity_compensation_horizontal(self) -> None:
        """Aim should be adjusted upward to compensate for gravity drop."""
        original_y = 100
        aim_x, aim_y = compensate_for_gravity(
            own_x=100,
            own_y=100,
            aim_x=300,
            aim_y=original_y,
            arrow_speed=500,
        )
        # Distance = 200px, travel time = 0.4s
        # gravity_drop = 0.5 * 400 * 0.16 = 32px
        assert aim_x == 300  # X should not change
        assert aim_y < original_y  # Aim higher (smaller y = higher)

    def test_gravity_compensation_amount(self) -> None:
        """Gravity compensation should follow correct formula."""
        aim_x, aim_y = compensate_for_gravity(
            own_x=0,
            own_y=0,
            aim_x=200,
            aim_y=0,
            arrow_speed=500,
        )
        # Distance = 200px, travel time = 0.4s
        # gravity_drop = 0.5 * 400 * 0.4^2 = 32px
        travel_time = 200 / 500
        expected_drop = 0.5 * GRAVITY_PX_PER_SEC2 * travel_time * travel_time
        assert aim_y == pytest.approx(-expected_drop)

    def test_gravity_compensation_zero_speed(self) -> None:
        """No compensation should be applied if arrow speed is zero."""
        aim_x, aim_y = compensate_for_gravity(
            own_x=100,
            own_y=100,
            aim_x=300,
            aim_y=200,
            arrow_speed=0,
        )
        assert aim_x == 300
        assert aim_y == 200

    def test_gravity_compensation_same_position(self) -> None:
        """No compensation if aim point is at bot position."""
        aim_x, aim_y = compensate_for_gravity(
            own_x=100,
            own_y=100,
            aim_x=100,
            aim_y=100,
            arrow_speed=500,
        )
        assert aim_x == 100
        assert aim_y == 100


class TestCalculateOptimalPower:
    """Tests for calculate_optimal_power function."""

    def test_optimal_power_close_range(self) -> None:
        """Close targets should use minimum power."""
        power = calculate_optimal_power(distance=50)
        assert power == 0.2  # MIN_POWER

        power = calculate_optimal_power(distance=100)
        assert power == 0.2  # At CLOSE_RANGE threshold

    def test_optimal_power_far_range(self) -> None:
        """Far targets should use maximum power."""
        power = calculate_optimal_power(distance=600)
        assert power == 1.0  # At MAX_RANGE

        power = calculate_optimal_power(distance=700)
        assert power == 1.0  # Beyond MAX_RANGE

    def test_optimal_power_mid_range(self) -> None:
        """Mid-range targets should use interpolated power."""
        # At 350px (midpoint between 100 and 600)
        power = calculate_optimal_power(distance=350)
        # Ratio = (350 - 100) / (600 - 100) = 0.5
        # Power = 0.2 + 0.5 * 0.8 = 0.6
        assert power == pytest.approx(0.6)

    def test_optimal_power_zero_distance(self) -> None:
        """Zero distance should use minimum power."""
        power = calculate_optimal_power(distance=0)
        assert power == 0.2


class TestShouldShoot:
    """Tests for should_shoot function."""

    @pytest.fixture
    def default_config(self) -> ShootingConfig:
        """Create default shooting configuration."""
        return ShootingConfig()

    @pytest.fixture
    def own_player(self) -> PlayerState:
        """Create a player state for the bot."""
        return PlayerState.model_validate(
            {
                "id": "player-1",
                "objectType": "player",
                "name": "Bot",
                "x": 100.0,
                "y": 200.0,
                "dx": 0.0,
                "dy": 0.0,
                "dir": 0.0,
                "rad": 20.0,
                "h": 100,
                "dead": False,
                "sht": False,
                "jc": 0,
                "ac": 4,
            }
        )

    @pytest.fixture
    def target_player(self) -> PlayerState:
        """Create a target player state."""
        return PlayerState.model_validate(
            {
                "id": "player-2",
                "objectType": "player",
                "name": "Enemy",
                "x": 300.0,
                "y": 200.0,
                "dx": 0.0,
                "dy": 0.0,
                "dir": 0.0,
                "rad": 20.0,
                "h": 100,
                "dead": False,
                "sht": False,
                "jc": 0,
                "ac": 4,
            }
        )

    def test_should_shoot_in_range(
        self,
        own_player: PlayerState,
        target_player: PlayerState,
        default_config: ShootingConfig,
    ) -> None:
        """Should shoot when target is in optimal range."""
        # Distance is 200px, within default range (50-500)
        assert should_shoot(own_player, target_player, default_config) is True

    def test_should_not_shoot_no_arrows(
        self,
        own_player: PlayerState,
        target_player: PlayerState,
        default_config: ShootingConfig,
    ) -> None:
        """Should not shoot when out of arrows."""
        own_player.arrow_count = 0
        assert should_shoot(own_player, target_player, default_config) is False

    def test_should_not_shoot_already_shooting(
        self,
        own_player: PlayerState,
        target_player: PlayerState,
        default_config: ShootingConfig,
    ) -> None:
        """Should not shoot when already shooting."""
        own_player.shooting = True
        assert should_shoot(own_player, target_player, default_config) is False

    def test_should_not_shoot_target_dead(
        self,
        own_player: PlayerState,
        target_player: PlayerState,
        default_config: ShootingConfig,
    ) -> None:
        """Should not shoot at dead target."""
        target_player.dead = True
        assert should_shoot(own_player, target_player, default_config) is False

    def test_should_not_shoot_too_close(
        self,
        own_player: PlayerState,
        target_player: PlayerState,
        default_config: ShootingConfig,
    ) -> None:
        """Should not shoot when target is too close."""
        target_player.x = 130.0  # 30px away, less than min_shooting_range (50)
        assert should_shoot(own_player, target_player, default_config) is False

    def test_should_not_shoot_too_far(
        self,
        own_player: PlayerState,
        target_player: PlayerState,
        default_config: ShootingConfig,
    ) -> None:
        """Should not shoot when target is too far."""
        target_player.x = 700.0  # 600px away, more than max_shooting_range (500)
        assert should_shoot(own_player, target_player, default_config) is False


class TestShouldReleaseShot:
    """Tests for should_release_shot function."""

    @pytest.fixture
    def default_config(self) -> ShootingConfig:
        """Create default shooting configuration."""
        return ShootingConfig()

    @pytest.fixture
    def own_player(self) -> PlayerState:
        """Create a player state for the bot."""
        return PlayerState.model_validate(
            {
                "id": "player-1",
                "objectType": "player",
                "name": "Bot",
                "x": 100.0,
                "y": 200.0,
                "dx": 0.0,
                "dy": 0.0,
                "dir": 0.0,
                "rad": 20.0,
                "h": 100,
                "dead": False,
                "sht": True,  # Currently shooting
                "jc": 0,
                "ac": 4,
            }
        )

    @pytest.fixture
    def target_player(self) -> PlayerState:
        """Create a target player state."""
        return PlayerState.model_validate(
            {
                "id": "player-2",
                "objectType": "player",
                "name": "Enemy",
                "x": 300.0,
                "y": 200.0,
                "dx": 0.0,
                "dy": 0.0,
                "dir": 0.0,
                "rad": 20.0,
                "h": 100,
                "dead": False,
                "sht": False,
                "jc": 0,
                "ac": 4,
            }
        )

    def test_should_release_at_max_power(
        self,
        own_player: PlayerState,
        target_player: PlayerState,
        default_config: ShootingConfig,
    ) -> None:
        """Should release when max power is reached."""
        start_time = 0.0
        current_time = 2.5  # Well past max power time (2.0s)
        assert (
            should_release_shot(
                own_player, target_player, start_time, current_time, default_config
            )
            is True
        )

    def test_should_not_release_below_min_power(
        self,
        own_player: PlayerState,
        target_player: PlayerState,
        default_config: ShootingConfig,
    ) -> None:
        """Should not release before minimum power is reached."""
        start_time = 0.0
        current_time = 0.1  # Only 0.1s, power ratio = 0.05, below min (0.2)
        assert (
            should_release_shot(
                own_player, target_player, start_time, current_time, default_config
            )
            is False
        )

    def test_should_release_at_optimal_power(
        self,
        own_player: PlayerState,
        target_player: PlayerState,
        default_config: ShootingConfig,
    ) -> None:
        """Should release when optimal power for distance is reached."""
        # Distance = 200px, optimal power ≈ 0.36
        # optimal = 0.2 + ((200-100)/(600-100)) * 0.8 = 0.2 + 0.2 * 0.8 = 0.36
        start_time = 0.0
        # Time to reach 0.36 power: 0.36 * 2.0 = 0.72s
        current_time = 0.75
        assert (
            should_release_shot(
                own_player, target_player, start_time, current_time, default_config
            )
            is True
        )


class TestRuleBasedBotShooting:
    """Integration tests for RuleBasedBot shooting behavior."""

    @pytest.fixture
    def game_state_with_enemy_in_range(self) -> GameState:
        """Create game state with enemy in shooting range."""
        update = GameUpdate.model_validate(
            {
                "fullUpdate": True,
                "objectStates": {
                    "player-1": {
                        "id": "player-1",
                        "objectType": "player",
                        "name": "Bot",
                        "x": 100.0,
                        "y": 200.0,
                        "dx": 0.0,
                        "dy": 0.0,
                        "dir": 0.0,
                        "rad": 20.0,
                        "h": 100,
                        "dead": False,
                        "sht": False,
                        "jc": 0,
                        "ac": 4,
                    },
                    "player-2": {
                        "id": "player-2",
                        "objectType": "player",
                        "name": "Enemy",
                        "x": 300.0,  # 200px away, in range
                        "y": 200.0,
                        "dx": 0.0,
                        "dy": 0.0,
                        "dir": 0.0,
                        "rad": 20.0,
                        "h": 100,
                        "dead": False,
                        "sht": False,
                        "jc": 0,
                        "ac": 4,
                    },
                },
                "events": [],
            }
        )
        return GameState.from_update(update)

    @pytest.fixture
    def game_state_no_arrows(self) -> GameState:
        """Create game state with bot having no arrows."""
        update = GameUpdate.model_validate(
            {
                "fullUpdate": True,
                "objectStates": {
                    "player-1": {
                        "id": "player-1",
                        "objectType": "player",
                        "name": "Bot",
                        "x": 100.0,
                        "y": 200.0,
                        "dx": 0.0,
                        "dy": 0.0,
                        "dir": 0.0,
                        "rad": 20.0,
                        "h": 100,
                        "dead": False,
                        "sht": False,
                        "jc": 0,
                        "ac": 0,  # No arrows
                    },
                    "player-2": {
                        "id": "player-2",
                        "objectType": "player",
                        "name": "Enemy",
                        "x": 300.0,
                        "y": 200.0,
                        "dx": 0.0,
                        "dy": 0.0,
                        "dir": 0.0,
                        "rad": 20.0,
                        "h": 100,
                        "dead": False,
                        "sht": False,
                        "jc": 0,
                        "ac": 4,
                    },
                },
                "events": [],
            }
        )
        return GameState.from_update(update)

    @pytest.fixture
    def game_state_enemy_dead(self) -> GameState:
        """Create game state with dead enemy."""
        update = GameUpdate.model_validate(
            {
                "fullUpdate": True,
                "objectStates": {
                    "player-1": {
                        "id": "player-1",
                        "objectType": "player",
                        "name": "Bot",
                        "x": 100.0,
                        "y": 200.0,
                        "dx": 0.0,
                        "dy": 0.0,
                        "dir": 0.0,
                        "rad": 20.0,
                        "h": 100,
                        "dead": False,
                        "sht": False,
                        "jc": 0,
                        "ac": 4,
                    },
                    "player-2": {
                        "id": "player-2",
                        "objectType": "player",
                        "name": "Enemy",
                        "x": 300.0,
                        "y": 200.0,
                        "dx": 0.0,
                        "dy": 0.0,
                        "dir": 0.0,
                        "rad": 20.0,
                        "h": 0,
                        "dead": True,  # Dead
                        "sht": False,
                        "jc": 0,
                        "ac": 4,
                    },
                },
                "events": [],
            }
        )
        return GameState.from_update(update)

    @pytest.mark.asyncio
    async def test_bot_starts_shooting_when_enemy_in_range(
        self, game_state_with_enemy_in_range: GameState
    ) -> None:
        """Bot should start charging when enemy is in range."""
        bot = RuleBasedBot("player-1")
        bot.update_state(game_state_with_enemy_in_range)

        # Use time > cooldown (0.5s) so first shot isn't blocked
        with patch("bot.bots.rule_based_bot.time.time", return_value=1.0):
            actions = await bot.decide_actions()

        # Find mouse actions
        mouse_actions = [a for a in actions if a[0] == "mouse_left"]
        assert len(mouse_actions) == 1
        assert mouse_actions[0][1] is True  # pressed = True (start charging)

    @pytest.mark.asyncio
    async def test_bot_does_not_shoot_without_arrows(
        self, game_state_no_arrows: GameState
    ) -> None:
        """Bot should not shoot when out of arrows."""
        bot = RuleBasedBot("player-1")
        bot.update_state(game_state_no_arrows)

        actions = await bot.decide_actions()

        # Should have no mouse actions
        mouse_actions = [a for a in actions if a[0] == "mouse_left"]
        assert len(mouse_actions) == 0

    @pytest.mark.asyncio
    async def test_bot_does_not_shoot_dead_target(
        self, game_state_enemy_dead: GameState
    ) -> None:
        """Bot should not shoot at dead enemies (moves to center instead)."""
        bot = RuleBasedBot("player-1")
        bot.update_state(game_state_enemy_dead)

        actions = await bot.decide_actions()

        # Should have no mouse actions (enemy is dead, so no target)
        mouse_actions = [a for a in actions if a[0] == "mouse_left"]
        assert len(mouse_actions) == 0

    @pytest.mark.asyncio
    async def test_bot_releases_shot_after_charging(
        self, game_state_with_enemy_in_range: GameState
    ) -> None:
        """Bot should release shot after sufficient charging."""
        bot = RuleBasedBot("player-1")
        bot.update_state(game_state_with_enemy_in_range)

        # First call - start charging (time > cooldown so first shot isn't blocked)
        with patch("bot.bots.rule_based_bot.time.time", return_value=1.0):
            actions1 = await bot.decide_actions()
            mouse_actions1 = [a for a in actions1 if a[0] == "mouse_left"]
            assert len(mouse_actions1) == 1
            assert mouse_actions1[0][1] is True  # Start charging

        # Second call - after enough time, release (1s later for sufficient power)
        with patch("bot.bots.rule_based_bot.time.time", return_value=2.0):
            actions2 = await bot.decide_actions()
            mouse_actions2 = [a for a in actions2 if a[0] == "mouse_left"]
            assert len(mouse_actions2) == 1
            assert mouse_actions2[0][1] is False  # Release

    @pytest.mark.asyncio
    async def test_bot_respects_shot_cooldown(
        self, game_state_with_enemy_in_range: GameState
    ) -> None:
        """Bot should respect cooldown between shots."""
        bot = RuleBasedBot("player-1")
        bot.update_state(game_state_with_enemy_in_range)

        # Start and complete a shot (time=1.0 to bypass initial cooldown)
        with patch("bot.bots.rule_based_bot.time.time", return_value=1.0):
            await bot.decide_actions()  # Start charging

        with patch("bot.bots.rule_based_bot.time.time", return_value=2.0):
            await bot.decide_actions()  # Release at time=2.0

        # Immediately after, should not start new shot (cooldown)
        with patch("bot.bots.rule_based_bot.time.time", return_value=2.1):
            actions = await bot.decide_actions()
            mouse_actions = [a for a in actions if a[0] == "mouse_left"]
            assert len(mouse_actions) == 0  # No shooting during cooldown

        # After cooldown, should start new shot (0.5s cooldown passed)
        with patch("bot.bots.rule_based_bot.time.time", return_value=2.6):
            actions = await bot.decide_actions()
            mouse_actions = [a for a in actions if a[0] == "mouse_left"]
            assert len(mouse_actions) == 1
            assert mouse_actions[0][1] is True  # Start new shot

    @pytest.mark.asyncio
    async def test_bot_releases_controls_when_dead(
        self, game_state_with_enemy_in_range: GameState
    ) -> None:
        """Bot should release all controls including shooting when dead."""
        bot = RuleBasedBot("player-1")
        bot.update_state(game_state_with_enemy_in_range)

        # Start shooting (time=1.0 to bypass initial cooldown)
        with patch("bot.bots.rule_based_bot.time.time", return_value=1.0):
            await bot.decide_actions()

        # Bot dies
        game_state_with_enemy_in_range.players["player-1"].dead = True
        bot.update_state(game_state_with_enemy_in_range)

        # Should release all controls
        with patch("bot.bots.rule_based_bot.time.time", return_value=1.5):
            actions = await bot.decide_actions()

            # Check keyboard releases
            keyboard_dict = {a[0]: a[1] for a in actions if len(a) == 2}
            assert keyboard_dict.get("w") is False
            assert keyboard_dict.get("a") is False
            assert keyboard_dict.get("d") is False

            # Check mouse release
            mouse_actions = [a for a in actions if a[0] == "mouse_left"]
            assert len(mouse_actions) == 1
            assert mouse_actions[0][1] is False  # Released

    @pytest.mark.asyncio
    async def test_aim_point_leads_moving_target(
        self, game_state_with_enemy_in_range: GameState
    ) -> None:
        """Bot should aim ahead of moving targets."""
        # Give enemy velocity
        game_state_with_enemy_in_range.players["player-2"].dx = 100.0  # Moving right

        bot = RuleBasedBot("player-1")
        bot.update_state(game_state_with_enemy_in_range)

        # Use time=1.0 to bypass initial cooldown
        with patch("bot.bots.rule_based_bot.time.time", return_value=1.0):
            actions = await bot.decide_actions()
            mouse_actions = [a for a in actions if len(a) == 4]
            assert len(mouse_actions) == 1
            mouse_action = mouse_actions[0]
            aim_x = mouse_action[2]  # type: ignore[index]

            # Aim X should be ahead of enemy's current X (300)
            assert aim_x > 300


class TestRuleBasedBotShootingConfig:
    """Tests for ShootingConfig integration with RuleBasedBot."""

    @pytest.fixture
    def game_state_with_enemy(self) -> GameState:
        """Create game state with enemy."""
        update = GameUpdate.model_validate(
            {
                "fullUpdate": True,
                "objectStates": {
                    "player-1": {
                        "id": "player-1",
                        "objectType": "player",
                        "name": "Bot",
                        "x": 100.0,
                        "y": 200.0,
                        "dx": 0.0,
                        "dy": 0.0,
                        "dir": 0.0,
                        "rad": 20.0,
                        "h": 100,
                        "dead": False,
                        "sht": False,
                        "jc": 0,
                        "ac": 4,
                    },
                    "player-2": {
                        "id": "player-2",
                        "objectType": "player",
                        "name": "Enemy",
                        "x": 200.0,  # 100px away
                        "y": 200.0,
                        "dx": 100.0,  # Moving right
                        "dy": 0.0,
                        "dir": 0.0,
                        "rad": 20.0,
                        "h": 100,
                        "dead": False,
                        "sht": False,
                        "jc": 0,
                        "ac": 4,
                    },
                },
                "events": [],
            }
        )
        return GameState.from_update(update)

    @pytest.mark.asyncio
    async def test_disable_lead_prediction(
        self, game_state_with_enemy: GameState
    ) -> None:
        """Bot should aim at current position when lead prediction disabled."""
        shooting_config = ShootingConfig(use_lead_prediction=False)
        config = RuleBasedBotConfig(shooting=shooting_config)
        bot = RuleBasedBot("player-1", config=config)
        bot.update_state(game_state_with_enemy)

        # Use time=1.0 to bypass initial cooldown
        with patch("bot.bots.rule_based_bot.time.time", return_value=1.0):
            actions = await bot.decide_actions()
            mouse_actions = [a for a in actions if len(a) == 4]
            assert len(mouse_actions) == 1
            mouse_action = mouse_actions[0]
            aim_x = mouse_action[2]  # type: ignore[index]

            # With lead prediction disabled and gravity compensation enabled,
            # aim_x should be exactly at target's current X (200)
            # but aim_y will be adjusted for gravity
            assert aim_x == 200

    @pytest.mark.asyncio
    async def test_disable_gravity_compensation(
        self, game_state_with_enemy: GameState
    ) -> None:
        """Bot should aim directly at target when gravity compensation disabled."""
        shooting_config = ShootingConfig(
            use_lead_prediction=False, use_gravity_compensation=False
        )
        config = RuleBasedBotConfig(shooting=shooting_config)
        bot = RuleBasedBot("player-1", config=config)
        bot.update_state(game_state_with_enemy)

        # Use time=1.0 to bypass initial cooldown
        with patch("bot.bots.rule_based_bot.time.time", return_value=1.0):
            actions = await bot.decide_actions()
            mouse_actions = [a for a in actions if len(a) == 4]
            assert len(mouse_actions) == 1
            mouse_action = mouse_actions[0]
            aim_x = mouse_action[2]  # type: ignore[index]
            aim_y = mouse_action[3]  # type: ignore[index]

            # With both disabled, aim should be exactly at target position
            assert aim_x == 200
            assert aim_y == 200

    @pytest.mark.asyncio
    async def test_custom_shooting_range(
        self, game_state_with_enemy: GameState
    ) -> None:
        """Bot should respect custom shooting range."""
        # Make range very restrictive
        shooting_config = ShootingConfig(
            min_shooting_range=150.0,  # Enemy at 100px is too close
            max_shooting_range=200.0,
        )
        config = RuleBasedBotConfig(shooting=shooting_config)
        bot = RuleBasedBot("player-1", config=config)
        bot.update_state(game_state_with_enemy)

        # Use time=1.0 to bypass initial cooldown
        with patch("bot.bots.rule_based_bot.time.time", return_value=1.0):
            actions = await bot.decide_actions()
            mouse_actions = [a for a in actions if a[0] == "mouse_left"]
            # Should not shoot - enemy at 100px, but min range is 150
            assert len(mouse_actions) == 0


class TestShootingModuleExports:
    """Tests for correct module exports."""

    def test_shooting_utils_exports(self) -> None:
        """Test that shooting_utils exports all functions."""
        from bot.bots.shooting_utils import (
            GRAVITY_PX_PER_SEC2,
            ShootingConfig,
            calculate_aim_point,
            calculate_arrow_speed,
            calculate_max_arrow_speed,
            calculate_optimal_power,
            compensate_for_gravity,
            should_release_shot,
            should_shoot,
        )

        assert GRAVITY_PX_PER_SEC2 > 0
        assert ShootingConfig is not None
        assert calculate_aim_point is not None
        assert calculate_arrow_speed is not None
        assert calculate_max_arrow_speed is not None
        assert calculate_optimal_power is not None
        assert compensate_for_gravity is not None
        assert should_release_shot is not None
        assert should_shoot is not None

    def test_bots_module_exports_shooting(self) -> None:
        """Test that bots module exports shooting utilities."""
        from bot.bots import (
            ShootingConfig,
            calculate_aim_point,
            calculate_arrow_speed,
            calculate_max_arrow_speed,
            calculate_optimal_power,
            compensate_for_gravity,
            should_release_shot,
            should_shoot,
        )

        assert ShootingConfig is not None
        assert calculate_aim_point is not None
        assert calculate_arrow_speed is not None
        assert calculate_max_arrow_speed is not None
        assert calculate_optimal_power is not None
        assert compensate_for_gravity is not None
        assert should_release_shot is not None
        assert should_shoot is not None
