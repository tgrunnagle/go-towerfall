"""Unit tests for evaluation module.

Tests cover:
- EvaluationResult dataclass and from_episodes factory
- ComparisonResult dataclass
- EvaluationManager promotion logic
"""

import pytest

from bot.training.evaluation import (
    ComparisonResult,
    EvaluationManager,
    EvaluationResult,
)
from bot.training.successive_config import PromotionCriteria


class TestEvaluationResultFromEpisodes:
    """Tests for EvaluationResult.from_episodes factory method."""

    def test_basic_statistics(self) -> None:
        """Test basic aggregate statistics computation."""
        result = EvaluationResult.from_episodes(
            episode_kills=[10, 8, 12],
            episode_deaths=[5, 4, 6],
            episode_wins=[True, True, False],
            episode_lengths=[100, 90, 110],
            episode_rewards=[50.0, 40.0, 60.0],
        )

        assert result.total_episodes == 3
        assert result.total_kills == 30
        assert result.total_deaths == 15
        assert result.total_wins == 2
        assert result.total_losses == 1
        assert result.kd_ratio == 2.0  # 30/15
        assert result.win_rate == pytest.approx(2 / 3)
        assert result.average_episode_length == 100.0
        assert result.average_reward == 50.0

    def test_zero_deaths_kd_ratio(self) -> None:
        """Test K/D ratio when total deaths is zero."""
        result = EvaluationResult.from_episodes(
            episode_kills=[5, 3],
            episode_deaths=[0, 0],
            episode_wins=[True, True],
            episode_lengths=[50, 60],
            episode_rewards=[20.0, 25.0],
        )

        # Should use max(deaths, 1) to avoid division by zero
        assert result.kd_ratio == 8.0  # 8/1

    def test_empty_episodes(self) -> None:
        """Test handling of empty episode list."""
        result = EvaluationResult.from_episodes(
            episode_kills=[],
            episode_deaths=[],
            episode_wins=[],
            episode_lengths=[],
            episode_rewards=[],
        )

        assert result.total_episodes == 0
        assert result.kd_ratio == 0.0
        assert result.win_rate == 0.0

    def test_single_episode(self) -> None:
        """Test with single episode."""
        result = EvaluationResult.from_episodes(
            episode_kills=[5],
            episode_deaths=[2],
            episode_wins=[True],
            episode_lengths=[80],
            episode_rewards=[30.0],
        )

        assert result.total_episodes == 1
        assert result.kd_ratio == 2.5
        assert result.win_rate == 1.0

    def test_is_statistically_significant(self) -> None:
        """Test statistical significance based on sample size."""
        # With 30+ episodes, should be significant
        result_large = EvaluationResult.from_episodes(
            episode_kills=[1] * 30,
            episode_deaths=[1] * 30,
            episode_wins=[True] * 30,
            episode_lengths=[50] * 30,
            episode_rewards=[10.0] * 30,
        )
        assert result_large.is_statistically_significant is True

        # With fewer episodes, should not be significant
        result_small = EvaluationResult.from_episodes(
            episode_kills=[1] * 20,
            episode_deaths=[1] * 20,
            episode_wins=[True] * 20,
            episode_lengths=[50] * 20,
            episode_rewards=[10.0] * 20,
        )
        assert result_small.is_statistically_significant is False


class TestEvaluationResultDataclass:
    """Tests for EvaluationResult dataclass."""

    def test_all_fields_accessible(self) -> None:
        """Test that all fields are accessible."""
        result = EvaluationResult(
            total_episodes=100,
            total_kills=150,
            total_deaths=100,
            total_wins=60,
            total_losses=40,
            kd_ratio=1.5,
            win_rate=0.6,
            average_episode_length=80.0,
            average_reward=25.0,
            kd_ratio_std=0.3,
            win_rate_std=0.4,
            confidence_interval_95=(1.2, 1.8),
        )

        assert result.total_episodes == 100
        assert result.total_kills == 150
        assert result.kd_ratio == 1.5
        assert result.confidence_interval_95 == (1.2, 1.8)


class TestComparisonResultDataclass:
    """Tests for ComparisonResult dataclass."""

    def test_all_fields_accessible(self) -> None:
        """Test that all fields are accessible."""
        agent_eval = EvaluationResult(
            total_episodes=50,
            total_kills=75,
            total_deaths=50,
            total_wins=30,
            total_losses=20,
            kd_ratio=1.5,
            win_rate=0.6,
            average_episode_length=70.0,
            average_reward=20.0,
            kd_ratio_std=0.2,
            win_rate_std=0.3,
            confidence_interval_95=(1.3, 1.7),
        )

        result = ComparisonResult(
            agent_eval=agent_eval,
            opponent_baseline=None,
            kd_improvement=0.5,
            win_rate_improvement=0.1,
            kd_p_value=0.01,
            is_significantly_better=True,
            meets_criteria=True,
            criteria_details={"min_kd_ratio": True, "kd_improvement": True},
        )

        assert result.agent_eval.kd_ratio == 1.5
        assert result.kd_improvement == 0.5
        assert result.is_significantly_better is True
        assert result.meets_criteria is True
        assert result.criteria_details["min_kd_ratio"] is True


class TestEvaluationManagerCompareToBaseline:
    """Tests for EvaluationManager.compare_to_baseline method."""

    def test_comparison_against_rule_based_bot(self) -> None:
        """Test comparison when opponent_baseline is None (rule-based bot)."""
        criteria = PromotionCriteria(
            min_kd_ratio=1.0,
            kd_improvement=0.1,
            min_eval_episodes=10,
            consecutive_passes=1,
        )
        manager = EvaluationManager(criteria)

        agent_eval = EvaluationResult(
            total_episodes=50,
            total_kills=60,
            total_deaths=50,
            total_wins=30,
            total_losses=20,
            kd_ratio=1.2,
            win_rate=0.6,
            average_episode_length=70.0,
            average_reward=20.0,
            kd_ratio_std=0.2,
            win_rate_std=0.3,
            confidence_interval_95=(1.1, 1.3),
        )

        comparison = manager.compare_to_baseline(agent_eval, opponent_baseline=None)

        # Rule-based bot assumed to have K/D of 1.0
        assert comparison.kd_improvement == pytest.approx(0.2)  # (1.2 - 1.0) / 1.0
        assert comparison.win_rate_improvement == pytest.approx(0.1)  # 0.6 - 0.5

    def test_comparison_against_model_opponent(self) -> None:
        """Test comparison with model opponent baseline."""
        criteria = PromotionCriteria(
            min_kd_ratio=1.0,
            kd_improvement=0.1,
            min_eval_episodes=10,
            consecutive_passes=1,
        )
        manager = EvaluationManager(criteria)

        agent_eval = EvaluationResult(
            total_episodes=50,
            total_kills=75,
            total_deaths=50,
            total_wins=30,
            total_losses=20,
            kd_ratio=1.5,
            win_rate=0.6,
            average_episode_length=70.0,
            average_reward=20.0,
            kd_ratio_std=0.2,
            win_rate_std=0.3,
            confidence_interval_95=(1.3, 1.7),
        )

        opponent_baseline = EvaluationResult(
            total_episodes=50,
            total_kills=60,
            total_deaths=50,
            total_wins=25,
            total_losses=25,
            kd_ratio=1.2,
            win_rate=0.5,
            average_episode_length=65.0,
            average_reward=18.0,
            kd_ratio_std=0.15,
            win_rate_std=0.25,
            confidence_interval_95=(1.1, 1.3),
        )

        comparison = manager.compare_to_baseline(agent_eval, opponent_baseline)

        # (1.5 - 1.2) / 1.2 = 0.25
        assert comparison.kd_improvement == pytest.approx(0.25)
        assert comparison.win_rate_improvement == pytest.approx(0.1)

    def test_criteria_check_passes(self) -> None:
        """Test that criteria check passes when all conditions met."""
        criteria = PromotionCriteria(
            min_kd_ratio=1.0,
            kd_improvement=0.1,
            min_eval_episodes=10,
            consecutive_passes=1,
        )
        manager = EvaluationManager(criteria)

        # Agent with good performance
        agent_eval = EvaluationResult(
            total_episodes=50,
            total_kills=75,
            total_deaths=50,
            total_wins=30,
            total_losses=20,
            kd_ratio=1.5,
            win_rate=0.6,
            average_episode_length=70.0,
            average_reward=20.0,
            kd_ratio_std=0.1,
            win_rate_std=0.3,
            confidence_interval_95=(1.4, 1.6),
        )

        comparison = manager.compare_to_baseline(agent_eval, opponent_baseline=None)

        assert comparison.criteria_details["min_kd_ratio"] is True
        assert comparison.criteria_details["kd_improvement"] is True
        assert comparison.criteria_details["min_eval_episodes"] is True
        assert comparison.meets_criteria is True

    def test_criteria_check_fails_kd_ratio(self) -> None:
        """Test that criteria check fails when K/D ratio too low."""
        criteria = PromotionCriteria(
            min_kd_ratio=1.5,
            kd_improvement=0.1,
            min_eval_episodes=10,
            consecutive_passes=1,
        )
        manager = EvaluationManager(criteria)

        # Agent with K/D below minimum
        agent_eval = EvaluationResult(
            total_episodes=50,
            total_kills=50,
            total_deaths=50,
            total_wins=25,
            total_losses=25,
            kd_ratio=1.0,
            win_rate=0.5,
            average_episode_length=70.0,
            average_reward=15.0,
            kd_ratio_std=0.2,
            win_rate_std=0.3,
            confidence_interval_95=(0.9, 1.1),
        )

        comparison = manager.compare_to_baseline(agent_eval, opponent_baseline=None)

        assert comparison.criteria_details["min_kd_ratio"] is False
        assert comparison.meets_criteria is False

    def test_criteria_check_with_win_rate(self) -> None:
        """Test criteria check when min_win_rate is specified."""
        criteria = PromotionCriteria(
            min_kd_ratio=1.0,
            kd_improvement=0.1,
            min_eval_episodes=10,
            min_win_rate=0.6,
            consecutive_passes=1,
        )
        manager = EvaluationManager(criteria)

        # Agent with good K/D but low win rate
        agent_eval = EvaluationResult(
            total_episodes=50,
            total_kills=75,
            total_deaths=50,
            total_wins=25,
            total_losses=25,
            kd_ratio=1.5,
            win_rate=0.5,  # Below 0.6 threshold
            average_episode_length=70.0,
            average_reward=20.0,
            kd_ratio_std=0.1,
            win_rate_std=0.3,
            confidence_interval_95=(1.4, 1.6),
        )

        comparison = manager.compare_to_baseline(agent_eval, opponent_baseline=None)

        assert comparison.criteria_details["min_win_rate"] is False
        assert comparison.meets_criteria is False


class TestEvaluationManagerConsecutivePasses:
    """Tests for consecutive passes tracking."""

    def test_consecutive_passes_incrementing(self) -> None:
        """Test that consecutive passes increment when criteria are met."""
        criteria = PromotionCriteria(
            min_kd_ratio=1.0,
            kd_improvement=0.0,  # No improvement required for this test
            min_eval_episodes=10,
            consecutive_passes=3,
        )
        manager = EvaluationManager(criteria)

        good_eval = EvaluationResult(
            total_episodes=50,
            total_kills=75,
            total_deaths=50,
            total_wins=30,
            total_losses=20,
            kd_ratio=1.5,
            win_rate=0.6,
            average_episode_length=70.0,
            average_reward=20.0,
            kd_ratio_std=0.1,
            win_rate_std=0.3,
            confidence_interval_95=(1.4, 1.6),
        )

        # First pass - should not meet consecutive_passes requirement
        comparison1 = manager.compare_to_baseline(good_eval, opponent_baseline=None)
        assert manager.consecutive_passes == 1
        assert comparison1.criteria_details["consecutive_passes"] is False
        assert comparison1.meets_criteria is False

        # Second pass
        comparison2 = manager.compare_to_baseline(good_eval, opponent_baseline=None)
        assert manager.consecutive_passes == 2
        assert comparison2.criteria_details["consecutive_passes"] is False

        # Third pass - should now meet requirement
        comparison3 = manager.compare_to_baseline(good_eval, opponent_baseline=None)
        assert manager.consecutive_passes == 3
        assert comparison3.criteria_details["consecutive_passes"] is True
        assert comparison3.meets_criteria is True

    def test_consecutive_passes_reset_on_failure(self) -> None:
        """Test that consecutive passes reset when criteria fail."""
        criteria = PromotionCriteria(
            min_kd_ratio=1.5,
            kd_improvement=0.0,
            min_eval_episodes=10,
            consecutive_passes=3,
        )
        manager = EvaluationManager(criteria)

        good_eval = EvaluationResult(
            total_episodes=50,
            total_kills=100,
            total_deaths=50,
            total_wins=35,
            total_losses=15,
            kd_ratio=2.0,
            win_rate=0.7,
            average_episode_length=70.0,
            average_reward=25.0,
            kd_ratio_std=0.1,
            win_rate_std=0.3,
            confidence_interval_95=(1.9, 2.1),
        )

        bad_eval = EvaluationResult(
            total_episodes=50,
            total_kills=50,
            total_deaths=50,
            total_wins=25,
            total_losses=25,
            kd_ratio=1.0,  # Below min_kd_ratio
            win_rate=0.5,
            average_episode_length=70.0,
            average_reward=15.0,
            kd_ratio_std=0.2,
            win_rate_std=0.3,
            confidence_interval_95=(0.9, 1.1),
        )

        # Build up consecutive passes
        manager.compare_to_baseline(good_eval, opponent_baseline=None)
        manager.compare_to_baseline(good_eval, opponent_baseline=None)
        assert manager.consecutive_passes == 2

        # Fail - should reset
        manager.compare_to_baseline(bad_eval, opponent_baseline=None)
        assert manager.consecutive_passes == 0

    def test_reset_consecutive_passes(self) -> None:
        """Test manual reset of consecutive passes."""
        criteria = PromotionCriteria(consecutive_passes=1)
        manager = EvaluationManager(criteria)

        good_eval = EvaluationResult(
            total_episodes=50,
            total_kills=75,
            total_deaths=50,
            total_wins=30,
            total_losses=20,
            kd_ratio=1.5,
            win_rate=0.6,
            average_episode_length=70.0,
            average_reward=20.0,
            kd_ratio_std=0.1,
            win_rate_std=0.3,
            confidence_interval_95=(1.4, 1.6),
        )

        manager.compare_to_baseline(good_eval, opponent_baseline=None)
        assert manager.consecutive_passes >= 1

        manager.reset_consecutive_passes()
        assert manager.consecutive_passes == 0
