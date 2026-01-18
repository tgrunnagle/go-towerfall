"""Evaluation and comparison logic for successive training.

This module provides classes for evaluating agent performance against
opponents and determining when agents should be promoted to the next
generation of training.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from bot.training.successive_config import PromotionCriteria


@dataclass
class EvaluationResult:
    """Result of evaluating an agent against an opponent.

    Captures raw statistics from evaluation episodes and computes
    aggregate metrics used for comparison and promotion decisions.

    Attributes:
        total_episodes: Number of episodes completed
        total_kills: Total kills across all episodes
        total_deaths: Total deaths across all episodes
        total_wins: Number of episodes won
        total_losses: Number of episodes lost
        kd_ratio: Aggregate kills/deaths ratio
        win_rate: Percentage of episodes won
        average_episode_length: Mean episode length in timesteps
        average_reward: Mean episode reward
        kd_ratio_std: Standard deviation of per-episode K/D
        win_rate_std: Standard deviation of win indicator
        confidence_interval_95: 95% CI for K/D ratio (low, high)
    """

    total_episodes: int
    total_kills: int
    total_deaths: int
    total_wins: int
    total_losses: int
    kd_ratio: float
    win_rate: float
    average_episode_length: float
    average_reward: float
    kd_ratio_std: float
    win_rate_std: float
    confidence_interval_95: tuple[float, float]

    @property
    def is_statistically_significant(self) -> bool:
        """Check if results have sufficient sample size.

        Returns:
            True if total_episodes >= 30 for statistical reliability.
        """
        return self.total_episodes >= 30

    @classmethod
    def from_episodes(
        cls,
        episode_kills: list[int],
        episode_deaths: list[int],
        episode_wins: list[bool],
        episode_lengths: list[int],
        episode_rewards: list[float],
    ) -> "EvaluationResult":
        """Create EvaluationResult from per-episode data.

        Args:
            episode_kills: Kills per episode
            episode_deaths: Deaths per episode
            episode_wins: Win indicator per episode
            episode_lengths: Length of each episode in timesteps
            episode_rewards: Total reward per episode

        Returns:
            EvaluationResult with computed statistics.
        """
        num_episodes = len(episode_kills)
        if num_episodes == 0:
            return cls(
                total_episodes=0,
                total_kills=0,
                total_deaths=0,
                total_wins=0,
                total_losses=0,
                kd_ratio=0.0,
                win_rate=0.0,
                average_episode_length=0.0,
                average_reward=0.0,
                kd_ratio_std=0.0,
                win_rate_std=0.0,
                confidence_interval_95=(0.0, 0.0),
            )

        total_kills = sum(episode_kills)
        total_deaths = sum(episode_deaths)
        total_wins = sum(1 for w in episode_wins if w)

        kd_ratio = total_kills / max(total_deaths, 1)
        win_rate = total_wins / num_episodes

        # Compute per-episode K/D for standard deviation
        per_episode_kd = [k / max(d, 1) for k, d in zip(episode_kills, episode_deaths)]
        kd_ratio_std = float(np.std(per_episode_kd)) if len(per_episode_kd) > 1 else 0.0

        # Win rate standard deviation
        win_indicators = [1 if w else 0 for w in episode_wins]
        win_rate_std = float(np.std(win_indicators)) if len(win_indicators) > 1 else 0.0

        # Compute 95% confidence interval for K/D ratio
        # Guard against zero standard error (when all episodes have identical K/D)
        if len(per_episode_kd) > 1:
            sem = stats.sem(per_episode_kd)
            mean_kd = float(np.mean(per_episode_kd))
            if sem > 0:
                ci = stats.t.interval(
                    confidence=0.95,
                    df=len(per_episode_kd) - 1,
                    loc=mean_kd,
                    scale=sem,
                )
                confidence_interval_95 = (float(ci[0]), float(ci[1]))
            else:
                # Zero variance - all episodes have same K/D
                confidence_interval_95 = (mean_kd, mean_kd)
        else:
            mean_kd = float(np.mean(per_episode_kd)) if per_episode_kd else 0.0
            confidence_interval_95 = (mean_kd, mean_kd)

        return cls(
            total_episodes=num_episodes,
            total_kills=total_kills,
            total_deaths=total_deaths,
            total_wins=total_wins,
            total_losses=num_episodes - total_wins,
            kd_ratio=kd_ratio,
            win_rate=win_rate,
            average_episode_length=float(np.mean(episode_lengths)),
            average_reward=float(np.mean(episode_rewards)),
            kd_ratio_std=kd_ratio_std,
            win_rate_std=win_rate_std,
            confidence_interval_95=confidence_interval_95,
        )


@dataclass
class ComparisonResult:
    """Result of comparing agent performance against opponent baseline.

    Encapsulates the comparison metrics and promotion decision based
    on the configured criteria.

    Attributes:
        agent_eval: Agent's evaluation result
        opponent_baseline: Opponent's baseline performance (None for rule-based)
        kd_improvement: Percentage improvement in K/D ratio
        win_rate_improvement: Absolute improvement in win rate
        kd_p_value: P-value for K/D difference significance test
        is_significantly_better: Whether improvement is statistically significant
        meets_criteria: Whether all promotion criteria are met
        criteria_details: Per-criterion pass/fail status
    """

    agent_eval: EvaluationResult
    opponent_baseline: EvaluationResult | None
    kd_improvement: float
    win_rate_improvement: float
    kd_p_value: float
    is_significantly_better: bool
    meets_criteria: bool
    criteria_details: dict[str, bool] = field(default_factory=dict)


class EvaluationManager:
    """Manages agent evaluation and comparison logic.

    Handles running evaluations, comparing results to baselines, and
    tracking consecutive criteria passes for promotion stability.

    Example:
        manager = EvaluationManager(criteria)
        result = await manager.evaluate_agent(agent, env, num_episodes=100)
        comparison = manager.compare_to_baseline(result, opponent_baseline)
        if comparison.meets_criteria:
            # Promote agent to next generation
            pass
    """

    def __init__(self, criteria: PromotionCriteria) -> None:
        """Initialize the evaluation manager.

        Args:
            criteria: Promotion criteria configuration.
        """
        self.criteria = criteria
        self._consecutive_passes = 0

    def compare_to_baseline(
        self,
        agent_eval: EvaluationResult,
        opponent_baseline: EvaluationResult | None,
    ) -> ComparisonResult:
        """Compare agent evaluation to opponent baseline.

        Calculates improvement metrics, performs statistical significance
        testing, and determines whether all promotion criteria are met.

        Args:
            agent_eval: Agent's evaluation result.
            opponent_baseline: Opponent's baseline performance (None for rule-based).

        Returns:
            ComparisonResult with comparison metrics and promotion decision.
        """
        # For rule-based bot, use fixed baseline (assumed K/D of 1.0)
        if opponent_baseline is None:
            opponent_kd = 1.0
            opponent_win_rate = 0.5
        else:
            opponent_kd = opponent_baseline.kd_ratio
            opponent_win_rate = opponent_baseline.win_rate

        # Calculate improvements
        # Note: We use max(opponent_kd, 0.01) to prevent division by zero when the
        # opponent has a K/D of 0.0 (e.g., opponent never gets kills). The 0.01 floor
        # means that if opponent_kd is 0, an agent with K/D of 1.0 would show 100x
        # improvement. This is intentional - any positive K/D is vastly better than
        # zero kills.
        kd_improvement = (agent_eval.kd_ratio - opponent_kd) / max(opponent_kd, 0.01)
        win_rate_improvement = agent_eval.win_rate - opponent_win_rate

        # Statistical significance test (one-sample t-test against opponent K/D)
        # H0: agent_kd <= opponent_kd, H1: agent_kd > opponent_kd
        if agent_eval.kd_ratio_std > 0 and agent_eval.total_episodes > 1:
            t_stat = (agent_eval.kd_ratio - opponent_kd) / (
                agent_eval.kd_ratio_std / np.sqrt(agent_eval.total_episodes)
            )
            p_value = float(1 - stats.t.cdf(t_stat, df=agent_eval.total_episodes - 1))
        else:
            # If no variance or single episode, can't compute significance
            p_value = 1.0 if agent_eval.kd_ratio <= opponent_kd else 0.0

        is_significantly_better = p_value < (1 - self.criteria.confidence_threshold)

        # Check individual criteria
        criteria_details: dict[str, bool] = {
            "min_kd_ratio": agent_eval.kd_ratio >= self.criteria.min_kd_ratio,
            "kd_improvement": kd_improvement >= self.criteria.kd_improvement,
            "min_eval_episodes": (
                agent_eval.total_episodes >= self.criteria.min_eval_episodes
            ),
            "statistical_significance": is_significantly_better,
        }

        if self.criteria.min_win_rate is not None:
            criteria_details["min_win_rate"] = (
                agent_eval.win_rate >= self.criteria.min_win_rate
            )

        # Update consecutive passes tracking
        all_criteria_met = all(criteria_details.values())
        if all_criteria_met:
            self._consecutive_passes += 1
        else:
            self._consecutive_passes = 0

        criteria_details["consecutive_passes"] = (
            self._consecutive_passes >= self.criteria.consecutive_passes
        )

        # Final promotion decision
        meets_criteria = all(criteria_details.values())

        return ComparisonResult(
            agent_eval=agent_eval,
            opponent_baseline=opponent_baseline,
            kd_improvement=kd_improvement,
            win_rate_improvement=win_rate_improvement,
            kd_p_value=p_value,
            is_significantly_better=is_significantly_better,
            meets_criteria=meets_criteria,
            criteria_details=criteria_details,
        )

    def reset_consecutive_passes(self) -> None:
        """Reset consecutive passes counter.

        Call this when starting a new generation to reset the promotion
        stability tracking.
        """
        self._consecutive_passes = 0

    @property
    def consecutive_passes(self) -> int:
        """Get current consecutive passes count."""
        return self._consecutive_passes
