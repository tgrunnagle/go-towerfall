"""Unit tests for the reward function module.

Tests cover:
- RewardConfig dataclass initialization and defaults
- StandardRewardFunction initialization and configuration
- Reset method behavior with and without initial stats
- Reward calculation for kills, deaths, and timestep penalty
- Delta-based reward tracking across multiple steps
- Edge cases (multiple kills/deaths in one step, None stats)
"""

from typing import Any

import pytest

from bot.gym.reward import RewardConfig, StandardRewardFunction
from bot.models import PlayerStatsDTO


def make_player_stats(**overrides: Any) -> PlayerStatsDTO:
    """Create a PlayerStatsDTO for testing."""
    data: dict[str, Any] = {
        "playerId": "player-456",
        "playerName": "TestPlayer",
        "kills": 0,
        "deaths": 0,
    }
    data.update(overrides)
    return PlayerStatsDTO.model_validate(data)


class TestRewardConfig:
    """Tests for RewardConfig dataclass."""

    def test_default_values(self) -> None:
        """Test RewardConfig has correct default values."""
        config = RewardConfig()

        assert config.kill_reward == 1.0
        assert config.death_penalty == -1.0
        assert config.timestep_penalty == -0.001

    def test_custom_values(self) -> None:
        """Test RewardConfig accepts custom values."""
        config = RewardConfig(
            kill_reward=2.0,
            death_penalty=-0.5,
            timestep_penalty=-0.01,
        )

        assert config.kill_reward == 2.0
        assert config.death_penalty == -0.5
        assert config.timestep_penalty == -0.01

    def test_zero_values(self) -> None:
        """Test RewardConfig accepts zero values."""
        config = RewardConfig(
            kill_reward=0.0,
            death_penalty=0.0,
            timestep_penalty=0.0,
        )

        assert config.kill_reward == 0.0
        assert config.death_penalty == 0.0
        assert config.timestep_penalty == 0.0

    def test_positive_death_penalty(self) -> None:
        """Test RewardConfig allows positive death_penalty (for experimentation)."""
        config = RewardConfig(death_penalty=0.5)

        assert config.death_penalty == 0.5


class TestStandardRewardFunctionInit:
    """Tests for StandardRewardFunction initialization."""

    def test_default_config(self) -> None:
        """Test initialization with default configuration."""
        reward_fn = StandardRewardFunction()

        assert reward_fn.config.kill_reward == 1.0
        assert reward_fn.config.death_penalty == -1.0
        assert reward_fn.config.timestep_penalty == -0.001

    def test_custom_config(self) -> None:
        """Test initialization with custom configuration."""
        config = RewardConfig(kill_reward=5.0)
        reward_fn = StandardRewardFunction(config)

        assert reward_fn.config.kill_reward == 5.0

    def test_initial_tracking_state(self) -> None:
        """Test initial tracking state is zeroed."""
        reward_fn = StandardRewardFunction()

        stats = reward_fn.get_episode_stats()
        assert stats["prev_kills"] == 0
        assert stats["prev_deaths"] == 0

    def test_get_config(self) -> None:
        """Test get_config returns the configuration."""
        config = RewardConfig(kill_reward=3.0)
        reward_fn = StandardRewardFunction(config)

        returned_config = reward_fn.get_config()
        assert returned_config is config


class TestStandardRewardFunctionReset:
    """Tests for StandardRewardFunction reset method."""

    def test_reset_with_stats(self) -> None:
        """Test reset initializes tracking from stats."""
        reward_fn = StandardRewardFunction()
        stats = make_player_stats(kills=5, deaths=2)

        reward_fn.reset(stats)

        episode_stats = reward_fn.get_episode_stats()
        assert episode_stats["prev_kills"] == 5
        assert episode_stats["prev_deaths"] == 2

    def test_reset_with_none(self) -> None:
        """Test reset with None resets to zero."""
        reward_fn = StandardRewardFunction()
        # First set some values
        reward_fn.reset(make_player_stats(kills=10, deaths=5))

        # Then reset with None
        reward_fn.reset(None)

        episode_stats = reward_fn.get_episode_stats()
        assert episode_stats["prev_kills"] == 0
        assert episode_stats["prev_deaths"] == 0

    def test_reset_clears_previous_state(self) -> None:
        """Test reset clears any previous tracking state."""
        reward_fn = StandardRewardFunction()

        # Simulate some steps
        reward_fn.reset(make_player_stats(kills=0, deaths=0))
        reward_fn.calculate(make_player_stats(kills=3, deaths=1))

        # Now reset with new initial values
        reward_fn.reset(make_player_stats(kills=0, deaths=0))

        episode_stats = reward_fn.get_episode_stats()
        assert episode_stats["prev_kills"] == 0
        assert episode_stats["prev_deaths"] == 0


class TestStandardRewardFunctionCalculate:
    """Tests for reward calculation."""

    def test_reward_for_single_kill(self) -> None:
        """Test reward for a single kill."""
        reward_fn = StandardRewardFunction()
        reward_fn.reset(make_player_stats(kills=0, deaths=0))

        reward = reward_fn.calculate(make_player_stats(kills=1, deaths=0))

        # +1.0 for kill, -0.001 timestep
        assert reward == pytest.approx(1.0 - 0.001, abs=0.0001)

    def test_reward_for_single_death(self) -> None:
        """Test penalty for a single death."""
        reward_fn = StandardRewardFunction()
        reward_fn.reset(make_player_stats(kills=0, deaths=0))

        reward = reward_fn.calculate(make_player_stats(kills=0, deaths=1))

        # -1.0 for death, -0.001 timestep
        assert reward == pytest.approx(-1.0 - 0.001, abs=0.0001)

    def test_timestep_penalty_only(self) -> None:
        """Test only timestep penalty when no kills/deaths."""
        reward_fn = StandardRewardFunction()
        reward_fn.reset(make_player_stats(kills=0, deaths=0))

        reward = reward_fn.calculate(make_player_stats(kills=0, deaths=0))

        assert reward == pytest.approx(-0.001, abs=0.0001)

    def test_kill_and_death_same_step(self) -> None:
        """Test reward when both kill and death occur in same step."""
        reward_fn = StandardRewardFunction()
        reward_fn.reset(make_player_stats(kills=0, deaths=0))

        reward = reward_fn.calculate(make_player_stats(kills=1, deaths=1))

        # +1.0 kill, -1.0 death, -0.001 timestep = -0.001
        assert reward == pytest.approx(-0.001, abs=0.0001)

    def test_multiple_kills_single_step(self) -> None:
        """Test reward for multiple kills in one step (e.g., double kill)."""
        reward_fn = StandardRewardFunction()
        reward_fn.reset(make_player_stats(kills=0, deaths=0))

        reward = reward_fn.calculate(make_player_stats(kills=3, deaths=0))

        # 3 * 1.0 = 3.0, -0.001 timestep
        assert reward == pytest.approx(3.0 - 0.001, abs=0.0001)

    def test_delta_based_tracking(self) -> None:
        """Test that rewards are delta-based, not absolute."""
        reward_fn = StandardRewardFunction()
        reward_fn.reset(make_player_stats(kills=5, deaths=2))

        # Stats show 6 kills, 2 deaths (1 new kill, 0 new deaths)
        reward = reward_fn.calculate(make_player_stats(kills=6, deaths=2))

        # Only +1.0 for the new kill
        assert reward == pytest.approx(1.0 - 0.001, abs=0.0001)

    def test_consecutive_steps_tracking(self) -> None:
        """Test correct tracking across consecutive steps."""
        reward_fn = StandardRewardFunction()
        reward_fn.reset(make_player_stats(kills=0, deaths=0))

        # Step 1: get a kill
        reward1 = reward_fn.calculate(make_player_stats(kills=1, deaths=0))
        assert reward1 == pytest.approx(1.0 - 0.001, abs=0.0001)

        # Step 2: no change
        reward2 = reward_fn.calculate(make_player_stats(kills=1, deaths=0))
        assert reward2 == pytest.approx(-0.001, abs=0.0001)

        # Step 3: die
        reward3 = reward_fn.calculate(make_player_stats(kills=1, deaths=1))
        assert reward3 == pytest.approx(-1.0 - 0.001, abs=0.0001)

        # Step 4: get another kill
        reward4 = reward_fn.calculate(make_player_stats(kills=2, deaths=1))
        assert reward4 == pytest.approx(1.0 - 0.001, abs=0.0001)

    def test_calculate_with_none_stats(self) -> None:
        """Test calculate returns only timestep penalty when stats are None."""
        reward_fn = StandardRewardFunction()
        reward_fn.reset(make_player_stats(kills=0, deaths=0))

        reward = reward_fn.calculate(None)

        assert reward == pytest.approx(-0.001, abs=0.0001)

    def test_none_stats_does_not_update_tracking(self) -> None:
        """Test that None stats doesn't update internal tracking."""
        reward_fn = StandardRewardFunction()
        reward_fn.reset(make_player_stats(kills=5, deaths=2))

        # Calculate with None
        reward_fn.calculate(None)

        # Tracking should still be at initial values
        stats = reward_fn.get_episode_stats()
        assert stats["prev_kills"] == 5
        assert stats["prev_deaths"] == 2


class TestStandardRewardFunctionCustomConfig:
    """Tests with custom reward configurations."""

    def test_custom_kill_reward(self) -> None:
        """Test reward with custom kill reward value."""
        config = RewardConfig(kill_reward=5.0)
        reward_fn = StandardRewardFunction(config)
        reward_fn.reset(make_player_stats(kills=0, deaths=0))

        reward = reward_fn.calculate(make_player_stats(kills=1, deaths=0))

        assert reward == pytest.approx(5.0 - 0.001, abs=0.0001)

    def test_custom_death_penalty(self) -> None:
        """Test reward with custom death penalty value."""
        config = RewardConfig(death_penalty=-2.0)
        reward_fn = StandardRewardFunction(config)
        reward_fn.reset(make_player_stats(kills=0, deaths=0))

        reward = reward_fn.calculate(make_player_stats(kills=0, deaths=1))

        assert reward == pytest.approx(-2.0 - 0.001, abs=0.0001)

    def test_custom_timestep_penalty(self) -> None:
        """Test reward with custom timestep penalty."""
        config = RewardConfig(timestep_penalty=-0.01)
        reward_fn = StandardRewardFunction(config)
        reward_fn.reset(make_player_stats(kills=0, deaths=0))

        reward = reward_fn.calculate(make_player_stats(kills=0, deaths=0))

        assert reward == pytest.approx(-0.01, abs=0.0001)

    def test_zero_timestep_penalty(self) -> None:
        """Test reward with zero timestep penalty."""
        config = RewardConfig(timestep_penalty=0.0)
        reward_fn = StandardRewardFunction(config)
        reward_fn.reset(make_player_stats(kills=0, deaths=0))

        reward = reward_fn.calculate(make_player_stats(kills=1, deaths=0))

        assert reward == pytest.approx(1.0, abs=0.0001)

    def test_asymmetric_kill_death_values(self) -> None:
        """Test with asymmetric kill/death values."""
        config = RewardConfig(kill_reward=2.0, death_penalty=-0.5)
        reward_fn = StandardRewardFunction(config)
        reward_fn.reset(make_player_stats(kills=0, deaths=0))

        # Kill and death in same step
        reward = reward_fn.calculate(make_player_stats(kills=1, deaths=1))

        # 2.0 - 0.5 - 0.001 = 1.499
        assert reward == pytest.approx(1.499, abs=0.0001)


class TestRewardIntegrationWithEnv:
    """Integration tests for reward function with TowerfallEnv."""

    def test_env_uses_reward_function(self) -> None:
        """Test that TowerfallEnv uses the reward function correctly."""
        from unittest.mock import MagicMock

        from bot.gym import TowerfallEnv
        from bot.models import GameState, PlayerState

        env = TowerfallEnv()
        env._client = MagicMock()
        env._client.player_id = "player-456"

        player_state_dict: dict[str, Any] = {
            "id": "player-456",
            "objectType": "player",
            "name": "TestPlayer",
            "x": 400.0,
            "y": 300.0,
            "dx": 0.0,
            "dy": 0.0,
            "dir": 0.0,
            "rad": 20.0,
            "h": 100,
            "dead": False,
            "sht": False,
            "jc": 2,
            "ac": 3,
        }
        game_state = GameState(
            players={"player-456": PlayerState.model_validate(player_state_dict)},
            canvas_size_x=800,
            canvas_size_y=600,
        )

        # Test reward calculation
        stats = {"player-456": make_player_stats(kills=1, deaths=0)}
        reward = env._calculate_reward(game_state, stats)

        # Should use reward function
        assert reward == pytest.approx(1.0 - 0.001, abs=0.0001)

    def test_env_custom_reward_config(self) -> None:
        """Test TowerfallEnv with custom reward configuration."""
        from bot.gym import RewardConfig, TowerfallEnv

        config = RewardConfig(kill_reward=10.0, timestep_penalty=-0.1)
        env = TowerfallEnv(reward_config=config)

        assert env._reward_fn.config.kill_reward == 10.0
        assert env._reward_fn.config.timestep_penalty == -0.1

    def test_env_reset_resets_reward_function(self) -> None:
        """Test that env.reset() resets the reward function tracking."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from bot.gym import TowerfallEnv
        from bot.models import GameState, PlayerState

        env = TowerfallEnv()

        player_state_dict: dict[str, Any] = {
            "id": "player-456",
            "objectType": "player",
            "name": "TestPlayer",
            "x": 400.0,
            "y": 300.0,
            "dx": 0.0,
            "dy": 0.0,
            "dir": 0.0,
            "rad": 20.0,
            "h": 100,
            "dead": False,
            "sht": False,
            "jc": 2,
            "ac": 3,
        }
        game_state = GameState(
            players={"player-456": PlayerState.model_validate(player_state_dict)},
            canvas_size_x=800,
            canvas_size_y=600,
        )
        info_dict = {
            "room_id": "room-123",
            "room_code": "ABC123",
            "player_id": "player-456",
        }

        # Manually set some tracking state
        env._reward_fn._prev_kills = 100
        env._reward_fn._prev_deaths = 50

        with patch.object(
            env, "_async_reset", new=AsyncMock(return_value=(game_state, info_dict))
        ):
            env._client = MagicMock()
            env._client.player_id = "player-456"

            env.reset()

        # Verify tracking was reset
        stats = env._reward_fn.get_episode_stats()
        assert stats["prev_kills"] == 0
        assert stats["prev_deaths"] == 0


class TestImports:
    """Tests for module imports."""

    def test_import_from_bot_gym(self) -> None:
        """Test importing reward classes from bot.gym."""
        from bot.gym import RewardConfig, RewardFunction, StandardRewardFunction

        assert RewardConfig is not None
        assert RewardFunction is not None
        assert StandardRewardFunction is not None

    def test_import_from_reward_module(self) -> None:
        """Test importing directly from reward module."""
        from bot.gym.reward import RewardConfig, StandardRewardFunction

        assert RewardConfig is not None
        assert StandardRewardFunction is not None
