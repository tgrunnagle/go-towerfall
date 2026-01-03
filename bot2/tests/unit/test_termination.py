"""Unit tests for the termination logic module.

Tests cover:
- TerminationConfig dataclass initialization, defaults, and validation
- TerminationTracker initialization and reset
- Episode statistics tracking (timesteps, deaths, kills, opponent deaths)
- Termination condition checking (max_timesteps, max_deaths, target_kills, etc.)
- Termination reason reporting
- Integration with TowerfallEnv
"""

from typing import Any

import pytest

from bot.gym.termination import TerminationConfig, TerminationTracker


class TestTerminationConfig:
    """Tests for TerminationConfig dataclass."""

    def test_default_values(self) -> None:
        """Test TerminationConfig has correct default values."""
        config = TerminationConfig()

        assert config.max_timesteps == 10_000
        assert config.max_deaths == 5
        assert config.max_opponent_deaths is None
        assert config.target_kills is None
        assert config.target_score_diff is None
        assert config.use_game_over_signal is True

    def test_custom_values(self) -> None:
        """Test TerminationConfig accepts custom values."""
        config = TerminationConfig(
            max_timesteps=5000,
            max_deaths=3,
            max_opponent_deaths=10,
            target_kills=5,
            target_score_diff=3,
            use_game_over_signal=False,
        )

        assert config.max_timesteps == 5000
        assert config.max_deaths == 3
        assert config.max_opponent_deaths == 10
        assert config.target_kills == 5
        assert config.target_score_diff == 3
        assert config.use_game_over_signal is False

    def test_disable_all_conditions(self) -> None:
        """Test TerminationConfig with all conditions disabled."""
        config = TerminationConfig(
            max_timesteps=None,
            max_deaths=None,
            max_opponent_deaths=None,
            target_kills=None,
            target_score_diff=None,
            use_game_over_signal=False,
        )

        assert config.max_timesteps is None
        assert config.max_deaths is None

    def test_validation_max_timesteps_positive(self) -> None:
        """Test validation rejects non-positive max_timesteps."""
        with pytest.raises(ValueError, match="max_timesteps must be positive"):
            TerminationConfig(max_timesteps=0)

        with pytest.raises(ValueError, match="max_timesteps must be positive"):
            TerminationConfig(max_timesteps=-1)

    def test_validation_max_deaths_positive(self) -> None:
        """Test validation rejects non-positive max_deaths."""
        with pytest.raises(ValueError, match="max_deaths must be positive"):
            TerminationConfig(max_deaths=0)

        with pytest.raises(ValueError, match="max_deaths must be positive"):
            TerminationConfig(max_deaths=-1)

    def test_validation_max_opponent_deaths_positive(self) -> None:
        """Test validation rejects non-positive max_opponent_deaths."""
        with pytest.raises(ValueError, match="max_opponent_deaths must be positive"):
            TerminationConfig(max_opponent_deaths=0)

    def test_validation_target_kills_positive(self) -> None:
        """Test validation rejects non-positive target_kills."""
        with pytest.raises(ValueError, match="target_kills must be positive"):
            TerminationConfig(target_kills=0)

    def test_validation_target_score_diff_positive(self) -> None:
        """Test validation rejects non-positive target_score_diff."""
        with pytest.raises(ValueError, match="target_score_diff must be positive"):
            TerminationConfig(target_score_diff=0)


class TestTerminationTrackerInit:
    """Tests for TerminationTracker initialization."""

    def test_default_config(self) -> None:
        """Test initialization with default configuration."""
        tracker = TerminationTracker()

        assert tracker.config.max_timesteps == 10_000
        assert tracker.config.max_deaths == 5

    def test_custom_config(self) -> None:
        """Test initialization with custom configuration."""
        config = TerminationConfig(max_timesteps=500, max_deaths=2)
        tracker = TerminationTracker(config)

        assert tracker.config.max_timesteps == 500
        assert tracker.config.max_deaths == 2

    def test_initial_state_zeroed(self) -> None:
        """Test initial tracking state is zeroed."""
        tracker = TerminationTracker()

        assert tracker.episode_timesteps == 0
        assert tracker.episode_deaths == 0
        assert tracker.episode_kills == 0
        assert tracker.episode_opponent_deaths == 0


class TestTerminationTrackerReset:
    """Tests for TerminationTracker reset method."""

    def test_reset_clears_all_counters(self) -> None:
        """Test reset clears all counters to zero."""
        tracker = TerminationTracker()

        # Simulate some activity
        tracker.increment_timestep()
        tracker.increment_timestep()
        tracker.update_from_stats(
            current_kills=3,
            current_deaths=2,
            prev_kills=0,
            prev_deaths=0,
            current_opponent_deaths=5,
            prev_opponent_deaths=0,
        )

        # Verify counters were set
        assert tracker.episode_timesteps == 2
        assert tracker.episode_kills == 3
        assert tracker.episode_deaths == 2
        assert tracker.episode_opponent_deaths == 5

        # Reset
        tracker.reset()

        # Verify all counters are zero
        assert tracker.episode_timesteps == 0
        assert tracker.episode_deaths == 0
        assert tracker.episode_kills == 0
        assert tracker.episode_opponent_deaths == 0


class TestTerminationTrackerIncrementTimestep:
    """Tests for timestep incrementing."""

    def test_increment_timestep(self) -> None:
        """Test increment_timestep increases counter."""
        tracker = TerminationTracker()

        tracker.increment_timestep()
        assert tracker.episode_timesteps == 1

        tracker.increment_timestep()
        assert tracker.episode_timesteps == 2

        tracker.increment_timestep()
        assert tracker.episode_timesteps == 3


class TestTerminationTrackerUpdateFromStats:
    """Tests for statistics update from game stats."""

    def test_single_kill(self) -> None:
        """Test tracking a single kill."""
        tracker = TerminationTracker()

        tracker.update_from_stats(
            current_kills=1,
            current_deaths=0,
            prev_kills=0,
            prev_deaths=0,
        )

        assert tracker.episode_kills == 1
        assert tracker.episode_deaths == 0

    def test_single_death(self) -> None:
        """Test tracking a single death."""
        tracker = TerminationTracker()

        tracker.update_from_stats(
            current_kills=0,
            current_deaths=1,
            prev_kills=0,
            prev_deaths=0,
        )

        assert tracker.episode_kills == 0
        assert tracker.episode_deaths == 1

    def test_multiple_kills_single_update(self) -> None:
        """Test tracking multiple kills in a single update (e.g., double kill)."""
        tracker = TerminationTracker()

        tracker.update_from_stats(
            current_kills=3,
            current_deaths=0,
            prev_kills=0,
            prev_deaths=0,
        )

        assert tracker.episode_kills == 3

    def test_delta_based_tracking(self) -> None:
        """Test that tracking is delta-based."""
        tracker = TerminationTracker()

        # First update: 2 kills, 1 death
        tracker.update_from_stats(
            current_kills=2,
            current_deaths=1,
            prev_kills=0,
            prev_deaths=0,
        )

        # Second update: 3 kills, 2 deaths (1 new kill, 1 new death)
        tracker.update_from_stats(
            current_kills=3,
            current_deaths=2,
            prev_kills=2,
            prev_deaths=1,
        )

        assert tracker.episode_kills == 3
        assert tracker.episode_deaths == 2

    def test_opponent_deaths_tracking(self) -> None:
        """Test tracking opponent deaths."""
        tracker = TerminationTracker()

        tracker.update_from_stats(
            current_kills=0,
            current_deaths=0,
            prev_kills=0,
            prev_deaths=0,
            current_opponent_deaths=2,
            prev_opponent_deaths=0,
        )

        assert tracker.episode_opponent_deaths == 2

    def test_opponent_deaths_delta_tracking(self) -> None:
        """Test opponent deaths are delta-based."""
        tracker = TerminationTracker()

        # First update
        tracker.update_from_stats(
            current_kills=0,
            current_deaths=0,
            prev_kills=0,
            prev_deaths=0,
            current_opponent_deaths=3,
            prev_opponent_deaths=0,
        )

        # Second update (2 more opponent deaths)
        tracker.update_from_stats(
            current_kills=0,
            current_deaths=0,
            prev_kills=0,
            prev_deaths=0,
            current_opponent_deaths=5,
            prev_opponent_deaths=3,
        )

        assert tracker.episode_opponent_deaths == 5


class TestTerminationTrackerCheckTermination:
    """Tests for termination condition checking."""

    def test_max_timesteps_truncation(self) -> None:
        """Test truncation at max timesteps."""
        config = TerminationConfig(max_timesteps=100, max_deaths=None)
        tracker = TerminationTracker(config)

        # Run up to threshold
        for _ in range(100):
            tracker.increment_timestep()

        terminated, truncated = tracker.check_termination()

        assert terminated is False
        assert truncated is True

    def test_before_max_timesteps_no_truncation(self) -> None:
        """Test no truncation before max timesteps."""
        config = TerminationConfig(max_timesteps=100, max_deaths=None)
        tracker = TerminationTracker(config)

        for _ in range(99):
            tracker.increment_timestep()

        terminated, truncated = tracker.check_termination()

        assert terminated is False
        assert truncated is False

    def test_max_deaths_termination(self) -> None:
        """Test termination at max deaths."""
        config = TerminationConfig(max_deaths=3, max_timesteps=10_000)
        tracker = TerminationTracker(config)

        # 3 deaths
        tracker.update_from_stats(
            current_kills=0,
            current_deaths=3,
            prev_kills=0,
            prev_deaths=0,
        )

        terminated, truncated = tracker.check_termination()

        assert terminated is True
        assert truncated is False

    def test_before_max_deaths_no_termination(self) -> None:
        """Test no termination before max deaths."""
        config = TerminationConfig(max_deaths=3)
        tracker = TerminationTracker(config)

        # 2 deaths
        tracker.update_from_stats(
            current_kills=0,
            current_deaths=2,
            prev_kills=0,
            prev_deaths=0,
        )

        terminated, truncated = tracker.check_termination()

        assert terminated is False
        assert truncated is False

    def test_target_kills_termination(self) -> None:
        """Test termination at target kills."""
        config = TerminationConfig(target_kills=5, max_deaths=None)
        tracker = TerminationTracker(config)

        tracker.update_from_stats(
            current_kills=5,
            current_deaths=0,
            prev_kills=0,
            prev_deaths=0,
        )

        terminated, truncated = tracker.check_termination()

        assert terminated is True
        assert truncated is False

    def test_target_score_diff_positive_termination(self) -> None:
        """Test termination when positive score difference reaches threshold."""
        config = TerminationConfig(target_score_diff=3, max_deaths=None)
        tracker = TerminationTracker(config)

        # 4 kills, 1 death = +3 score diff
        tracker.update_from_stats(
            current_kills=4,
            current_deaths=1,
            prev_kills=0,
            prev_deaths=0,
        )

        terminated, truncated = tracker.check_termination()

        assert terminated is True

    def test_target_score_diff_negative_termination(self) -> None:
        """Test termination when negative score difference reaches threshold."""
        config = TerminationConfig(target_score_diff=3, max_deaths=None)
        tracker = TerminationTracker(config)

        # 0 kills, 4 deaths = -4 score diff (abs = 4 >= 3)
        tracker.update_from_stats(
            current_kills=0,
            current_deaths=4,
            prev_kills=0,
            prev_deaths=0,
        )

        terminated, truncated = tracker.check_termination()

        assert terminated is True

    def test_max_opponent_deaths_termination(self) -> None:
        """Test termination at max opponent deaths."""
        config = TerminationConfig(max_opponent_deaths=3, max_deaths=None)
        tracker = TerminationTracker(config)

        tracker.update_from_stats(
            current_kills=0,
            current_deaths=0,
            prev_kills=0,
            prev_deaths=0,
            current_opponent_deaths=3,
            prev_opponent_deaths=0,
        )

        terminated, truncated = tracker.check_termination()

        assert terminated is True

    def test_game_over_signal_termination(self) -> None:
        """Test termination from game over signal."""
        config = TerminationConfig(use_game_over_signal=True, max_deaths=None)
        tracker = TerminationTracker(config)

        terminated, truncated = tracker.check_termination(is_game_over=True)

        assert terminated is True
        assert truncated is False

    def test_game_over_signal_disabled(self) -> None:
        """Test game over signal can be disabled."""
        config = TerminationConfig(use_game_over_signal=False, max_deaths=None)
        tracker = TerminationTracker(config)

        terminated, truncated = tracker.check_termination(is_game_over=True)

        assert terminated is False
        assert truncated is False

    def test_combined_conditions_first_wins(self) -> None:
        """Test that first condition to trigger wins."""
        config = TerminationConfig(
            max_timesteps=100,
            max_deaths=3,
            target_kills=10,
        )
        tracker = TerminationTracker(config)

        # Die 3 times (should terminate before max_timesteps or target_kills)
        tracker.update_from_stats(
            current_kills=2,  # Less than target_kills
            current_deaths=3,  # Equals max_deaths
            prev_kills=0,
            prev_deaths=0,
        )

        for _ in range(50):  # Less than max_timesteps
            tracker.increment_timestep()

        terminated, truncated = tracker.check_termination()

        assert terminated is True
        assert truncated is False

    def test_no_termination_when_disabled(self) -> None:
        """Test no termination when all conditions disabled."""
        config = TerminationConfig(
            max_timesteps=None,
            max_deaths=None,
            max_opponent_deaths=None,
            target_kills=None,
            target_score_diff=None,
            use_game_over_signal=False,
        )
        tracker = TerminationTracker(config)

        # Lots of activity but nothing should terminate
        for _ in range(1000):
            tracker.increment_timestep()
        tracker.update_from_stats(
            current_kills=100,
            current_deaths=100,
            prev_kills=0,
            prev_deaths=0,
            current_opponent_deaths=100,
            prev_opponent_deaths=0,
        )

        terminated, truncated = tracker.check_termination(is_game_over=True)

        assert terminated is False
        assert truncated is False


class TestTerminationTrackerGetTerminationReason:
    """Tests for termination reason reporting."""

    def test_max_timesteps_reason(self) -> None:
        """Test termination reason for max timesteps."""
        config = TerminationConfig(max_timesteps=100, max_deaths=None)
        tracker = TerminationTracker(config)

        for _ in range(100):
            tracker.increment_timestep()

        terminated, truncated = tracker.check_termination()
        reason = tracker.get_termination_reason(terminated, truncated)

        assert reason == "max_timesteps"

    def test_max_deaths_reason(self) -> None:
        """Test termination reason for max deaths."""
        config = TerminationConfig(max_deaths=3)
        tracker = TerminationTracker(config)

        tracker.update_from_stats(
            current_kills=0,
            current_deaths=3,
            prev_kills=0,
            prev_deaths=0,
        )

        terminated, truncated = tracker.check_termination()
        reason = tracker.get_termination_reason(terminated, truncated)

        assert reason == "max_deaths"

    def test_target_kills_reason(self) -> None:
        """Test termination reason for target kills."""
        config = TerminationConfig(target_kills=5, max_deaths=None)
        tracker = TerminationTracker(config)

        tracker.update_from_stats(
            current_kills=5,
            current_deaths=0,
            prev_kills=0,
            prev_deaths=0,
        )

        terminated, truncated = tracker.check_termination()
        reason = tracker.get_termination_reason(terminated, truncated)

        assert reason == "target_kills"

    def test_score_diff_reason(self) -> None:
        """Test termination reason for score difference."""
        config = TerminationConfig(target_score_diff=3, max_deaths=None)
        tracker = TerminationTracker(config)

        tracker.update_from_stats(
            current_kills=5,
            current_deaths=1,
            prev_kills=0,
            prev_deaths=0,
        )

        terminated, truncated = tracker.check_termination()
        reason = tracker.get_termination_reason(terminated, truncated)

        assert reason == "score_diff"

    def test_opponent_deaths_reason(self) -> None:
        """Test termination reason for opponent deaths."""
        config = TerminationConfig(max_opponent_deaths=5, max_deaths=None)
        tracker = TerminationTracker(config)

        tracker.update_from_stats(
            current_kills=0,
            current_deaths=0,
            prev_kills=0,
            prev_deaths=0,
            current_opponent_deaths=5,
            prev_opponent_deaths=0,
        )

        terminated, truncated = tracker.check_termination()
        reason = tracker.get_termination_reason(terminated, truncated)

        assert reason == "opponent_deaths"

    def test_game_over_reason(self) -> None:
        """Test termination reason for game over signal."""
        config = TerminationConfig(use_game_over_signal=True, max_deaths=None)
        tracker = TerminationTracker(config)

        terminated, truncated = tracker.check_termination(is_game_over=True)
        reason = tracker.get_termination_reason(terminated, truncated)

        assert reason == "game_over"

    def test_no_reason_when_not_terminated(self) -> None:
        """Test no reason when episode not terminated."""
        tracker = TerminationTracker()

        terminated, truncated = tracker.check_termination()
        reason = tracker.get_termination_reason(terminated, truncated)

        assert reason is None


class TestTerminationTrackerGetEpisodeStats:
    """Tests for episode statistics retrieval."""

    def test_get_episode_stats_initial(self) -> None:
        """Test get_episode_stats returns zeros initially."""
        tracker = TerminationTracker()

        stats = tracker.get_episode_stats()

        assert stats == {
            "episode_timesteps": 0,
            "episode_deaths": 0,
            "episode_kills": 0,
            "episode_opponent_deaths": 0,
        }

    def test_get_episode_stats_after_updates(self) -> None:
        """Test get_episode_stats reflects updates."""
        tracker = TerminationTracker()

        tracker.increment_timestep()
        tracker.increment_timestep()
        tracker.update_from_stats(
            current_kills=3,
            current_deaths=2,
            prev_kills=0,
            prev_deaths=0,
            current_opponent_deaths=1,
            prev_opponent_deaths=0,
        )

        stats = tracker.get_episode_stats()

        assert stats == {
            "episode_timesteps": 2,
            "episode_deaths": 2,
            "episode_kills": 3,
            "episode_opponent_deaths": 1,
        }


class TestTerminationIntegrationWithEnv:
    """Integration tests for termination logic with TowerfallEnv."""

    def test_env_accepts_termination_config(self) -> None:
        """Test TowerfallEnv accepts termination configuration."""
        from bot.gym import TerminationConfig, TowerfallEnv

        config = TerminationConfig(max_timesteps=500, max_deaths=2)
        env = TowerfallEnv(termination_config=config)

        assert env._termination_config.max_timesteps == 500
        assert env._termination_config.max_deaths == 2

    def test_env_default_uses_max_episode_steps(self) -> None:
        """Test TowerfallEnv uses max_episode_steps when no termination_config."""
        from bot.gym import TowerfallEnv

        env = TowerfallEnv(max_episode_steps=2000)

        # Should have created TerminationConfig with max_timesteps=2000
        assert env._termination_config.max_timesteps == 2000

    def test_env_reset_resets_termination_tracker(self) -> None:
        """Test that env.reset() resets the termination tracker."""
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
        env._termination_tracker._episode_timesteps = 500
        env._termination_tracker._episode_kills = 10
        env._termination_tracker._episode_deaths = 5

        with patch.object(
            env, "_async_reset", new=AsyncMock(return_value=(game_state, info_dict))
        ):
            env._client = MagicMock()
            env._client.player_id = "player-456"

            env.reset()

        # Verify tracking was reset
        assert env._termination_tracker.episode_timesteps == 0
        assert env._termination_tracker.episode_kills == 0
        assert env._termination_tracker.episode_deaths == 0

    def test_env_step_updates_termination_tracker(self) -> None:
        """Test that env.step() updates termination tracker."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from bot.gym import TowerfallEnv
        from bot.models import GameState, PlayerState, PlayerStatsDTO

        env = TowerfallEnv()
        env._client = MagicMock()
        env._client.player_id = "player-456"
        env._episode_step = 0

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
        stats = {
            "player-456": PlayerStatsDTO.model_validate(
                {
                    "playerId": "player-456",
                    "playerName": "Test",
                    "kills": 1,
                    "deaths": 0,
                }
            )
        }

        with patch.object(
            env, "_async_step", new=AsyncMock(return_value=(game_state, stats))
        ):
            env.step(0)

        assert env._termination_tracker.episode_timesteps == 1
        assert env._termination_tracker.episode_kills == 1
        assert env._termination_tracker.episode_deaths == 0

    def test_env_step_returns_termination_info(self) -> None:
        """Test that env.step() returns termination info in info dict."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from bot.gym import TowerfallEnv
        from bot.models import GameState, PlayerState, PlayerStatsDTO

        env = TowerfallEnv()
        env._client = MagicMock()
        env._client.player_id = "player-456"
        env._episode_step = 0

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
        stats = {
            "player-456": PlayerStatsDTO.model_validate(
                {
                    "playerId": "player-456",
                    "playerName": "Test",
                    "kills": 2,
                    "deaths": 1,
                }
            )
        }

        with patch.object(
            env, "_async_step", new=AsyncMock(return_value=(game_state, stats))
        ):
            _, _, _, _, info = env.step(0)

        assert "episode_timesteps" in info
        assert "episode_deaths" in info
        assert "episode_kills" in info
        assert "termination_reason" in info
        assert info["episode_timesteps"] == 1
        assert info["episode_kills"] == 2
        assert info["episode_deaths"] == 1

    def test_env_truncates_at_max_timesteps(self) -> None:
        """Test env truncates when max timesteps reached."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from bot.gym import TerminationConfig, TowerfallEnv
        from bot.models import GameState, PlayerState, PlayerStatsDTO

        config = TerminationConfig(max_timesteps=5, max_deaths=None)
        env = TowerfallEnv(termination_config=config)
        env._client = MagicMock()
        env._client.player_id = "player-456"
        env._episode_step = 0

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
        stats = {
            "player-456": PlayerStatsDTO.model_validate(
                {
                    "playerId": "player-456",
                    "playerName": "Test",
                    "kills": 0,
                    "deaths": 0,
                }
            )
        }

        with patch.object(
            env, "_async_step", new=AsyncMock(return_value=(game_state, stats))
        ):
            # Run 5 steps
            for i in range(5):
                _, _, terminated, truncated, info = env.step(0)

                if i < 4:
                    assert truncated is False
                else:
                    # 5th step should truncate
                    assert truncated is True
                    assert terminated is False
                    assert info["termination_reason"] == "max_timesteps"

    def test_env_terminates_at_max_deaths(self) -> None:
        """Test env terminates when max deaths reached."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from bot.gym import TerminationConfig, TowerfallEnv
        from bot.models import GameState, PlayerState, PlayerStatsDTO

        config = TerminationConfig(max_deaths=2, max_timesteps=1000)
        env = TowerfallEnv(termination_config=config)
        env._client = MagicMock()
        env._client.player_id = "player-456"
        env._episode_step = 0

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
        # Stats show 2 deaths
        stats = {
            "player-456": PlayerStatsDTO.model_validate(
                {
                    "playerId": "player-456",
                    "playerName": "Test",
                    "kills": 0,
                    "deaths": 2,
                }
            )
        }

        with patch.object(
            env, "_async_step", new=AsyncMock(return_value=(game_state, stats))
        ):
            _, _, terminated, truncated, info = env.step(0)

        assert terminated is True
        assert truncated is False
        assert info["termination_reason"] == "max_deaths"


class TestImports:
    """Tests for module imports."""

    def test_import_from_bot_gym(self) -> None:
        """Test importing termination classes from bot.gym."""
        from bot.gym import TerminationConfig, TerminationTracker

        assert TerminationConfig is not None
        assert TerminationTracker is not None

    def test_import_from_termination_module(self) -> None:
        """Test importing directly from termination module."""
        from bot.gym.termination import TerminationConfig, TerminationTracker

        assert TerminationConfig is not None
        assert TerminationTracker is not None
