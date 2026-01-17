"""Successive self-play training coordinator.

This module provides the SuccessiveTrainer class that coordinates
multi-generation training, where each generation trains against
the previous generation's model (or a rule-based bot for generation 0).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from bot.training.evaluation import (
    ComparisonResult,
    EvaluationManager,
    EvaluationResult,
)
from bot.training.orchestrator import TrainingOrchestrator
from bot.training.registry import ModelMetadata, ModelRegistry
from bot.training.successive_config import SuccessiveTrainingConfig

logger = logging.getLogger("bot.training.successive")


@dataclass
class GenerationResult:
    """Result of training one generation.

    Captures all information about a completed generation, including
    the trained model, evaluation metrics, and promotion status.

    Attributes:
        generation: Generation number (0-indexed).
        model_id: Registered model ID.
        model_metadata: Full model metadata from registry.
        final_evaluation: Final evaluation results for this generation.
        comparison: Comparison results against opponent (None for gen 0).
        timesteps_trained: Total timesteps trained in this generation.
        was_promoted: Whether this model was promoted to be the next opponent.
        promotion_reason: Human-readable explanation of promotion decision.
    """

    generation: int
    model_id: str
    model_metadata: ModelMetadata
    final_evaluation: EvaluationResult | None
    comparison: ComparisonResult | None
    timesteps_trained: int
    was_promoted: bool
    promotion_reason: str


class SuccessiveTrainer:
    """Coordinates successive self-play training across generations.

    Manages the progression from rule-based opponent through successive
    generations of trained models, evaluating each generation and
    promoting models that surpass their opponents.

    The training progression:
    1. Generation 0: Train PPO agent against rule-based bot
    2. Evaluation: Periodically evaluate agent performance (K/D ratio)
    3. Promotion: When agent achieves K/D > opponent's baseline, it becomes
       the new opponent
    4. Generation N: Train new agent against Generation N-1
    5. Termination: Stop when maximum generation reached or performance plateaus

    Example:
        config = SuccessiveTrainingConfig(
            base_config=OrchestratorConfig(num_envs=4),
            max_generations=5,
            timesteps_per_generation=500_000,
        )

        async with SuccessiveTrainer(config) as trainer:
            results = await trainer.train()
            print(f"Trained {len(results)} generations")

    Attributes:
        config: Successive training configuration.
        registry: Model registry for storing trained models.
        eval_manager: Evaluation manager for promotion decisions.
        current_generation: Current generation being trained.
        current_opponent_id: Model ID of current opponent (None = rule-based).
        generation_results: Results from all completed generations.
        is_running: Whether training is currently running.
        output_dir: Base output directory for all training artifacts.
    """

    def __init__(self, config: SuccessiveTrainingConfig) -> None:
        """Initialize the successive trainer.

        Args:
            config: Successive training configuration.
        """
        self.config = config
        self.output_dir = Path(config.output_dir)

        # Components (initialized in setup)
        self.registry: ModelRegistry | None = None
        self.eval_manager: EvaluationManager | None = None

        # State
        self.current_generation = 0
        self.current_opponent_id: str | None = None
        self.generation_results: list[GenerationResult] = []
        self.is_running = False

        # Callbacks for training events
        self._callbacks: list[Callable[[dict[str, Any]], None]] = []

    async def setup(self) -> None:
        """Initialize components for successive training.

        Creates output directories, initializes the model registry,
        and sets up the evaluation manager.
        """
        logger.info("Setting up successive trainer...")

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "checkpoints").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)

        # Initialize registry
        registry_path = self.output_dir / "model_registry"
        self.registry = ModelRegistry(registry_path=str(registry_path))

        # Initialize evaluation manager
        self.eval_manager = EvaluationManager(self.config.promotion_criteria)

        logger.info("Successive trainer setup complete. Output: %s", self.output_dir)

    async def train(self) -> list[GenerationResult]:
        """Run successive training through all generations.

        Trains each generation until max_generations is reached or
        early stopping criteria are met (stagnation).

        Returns:
            List of results for each generation trained.

        Raises:
            RuntimeError: If setup() was not called before train().
        """
        if self.registry is None or self.eval_manager is None:
            raise RuntimeError("Must call setup() before train()")

        self.is_running = True
        logger.info(
            "Starting successive training for up to %d generations",
            self.config.max_generations,
        )

        stagnant_count = 0

        while (
            self.current_generation < self.config.max_generations
            and self.is_running
            and stagnant_count < self.config.max_stagnant_evaluations
        ):
            logger.info("=" * 60)
            logger.info("Starting Generation %d", self.current_generation)
            logger.info("=" * 60)

            # Train this generation
            result = await self._train_generation()
            self.generation_results.append(result)

            # Log result
            self._log_generation_result(result)

            # Notify callbacks
            self._invoke_callbacks(
                {
                    "type": "generation_complete",
                    "generation": result.generation,
                    "result": result,
                }
            )

            # Check for stagnation
            if not result.was_promoted:
                stagnant_count += 1
                logger.warning(
                    "Generation %d not promoted. Stagnant count: %d/%d",
                    self.current_generation,
                    stagnant_count,
                    self.config.max_stagnant_evaluations,
                )
            else:
                stagnant_count = 0
                # Update opponent for next generation
                self.current_opponent_id = result.model_id

            self.current_generation += 1

        # Final summary
        self._log_training_summary()

        return self.generation_results

    async def _train_generation(self) -> GenerationResult:
        """Train a single generation.

        Creates an orchestrator for this generation, runs training,
        and evaluates the result for promotion.

        Returns:
            GenerationResult with training and evaluation data.
        """
        if self.registry is None or self.eval_manager is None:
            raise RuntimeError("Components not initialized")

        # Reset consecutive passes for new generation
        self.eval_manager.reset_consecutive_passes()

        # Create config for this generation
        gen_config = self.config.create_generation_config(
            generation=self.current_generation,
            opponent_model_id=self.current_opponent_id,
        )

        # Get opponent baseline if we have a model opponent
        opponent_baseline = await self._get_opponent_baseline()

        # Track promotion during training via callback
        promotion_achieved = False
        final_comparison: ComparisonResult | None = None

        def evaluation_callback(event: dict[str, Any]) -> None:
            nonlocal promotion_achieved, final_comparison
            if event.get("type") == "evaluation" and self.eval_manager is not None:
                metrics = event.get("metrics", {})

                # Build evaluation result from metrics
                eval_result = EvaluationResult(
                    total_episodes=self.config.evaluation_episodes,
                    total_kills=int(
                        metrics.get("eval_avg_kills", 0)
                        * self.config.evaluation_episodes
                    ),
                    total_deaths=int(
                        metrics.get("eval_avg_deaths", 1)
                        * self.config.evaluation_episodes
                    ),
                    total_wins=int(
                        metrics.get("eval_win_rate", 0)
                        * self.config.evaluation_episodes
                    ),
                    total_losses=self.config.evaluation_episodes
                    - int(
                        metrics.get("eval_win_rate", 0)
                        * self.config.evaluation_episodes
                    ),
                    kd_ratio=metrics.get("eval_kd_ratio", 0.0),
                    win_rate=metrics.get("eval_win_rate", 0.0),
                    average_episode_length=metrics.get("eval_avg_length", 0.0),
                    average_reward=metrics.get("eval_avg_reward", 0.0),
                    kd_ratio_std=0.0,  # Would need per-episode data
                    win_rate_std=0.0,
                    confidence_interval_95=(0.0, 0.0),
                )

                # Compare to baseline
                comparison = self.eval_manager.compare_to_baseline(
                    eval_result, opponent_baseline
                )
                final_comparison = comparison

                if comparison.meets_criteria:
                    promotion_achieved = True
                    logger.info(
                        "Generation %d meets promotion criteria!",
                        self.current_generation,
                    )

        # Train with orchestrator
        async with TrainingOrchestrator(gen_config) as orchestrator:
            orchestrator.register_callback(evaluation_callback)
            metadata = await orchestrator.train()

        # Build final evaluation result from metadata
        final_evaluation = EvaluationResult(
            total_episodes=metadata.training_metrics.total_episodes,
            total_kills=int(
                metadata.training_metrics.average_kills
                * metadata.training_metrics.total_episodes
            ),
            total_deaths=int(
                metadata.training_metrics.average_deaths
                * metadata.training_metrics.total_episodes
            ),
            total_wins=int(
                metadata.training_metrics.win_rate
                * metadata.training_metrics.total_episodes
            ),
            total_losses=int(
                (1 - metadata.training_metrics.win_rate)
                * metadata.training_metrics.total_episodes
            ),
            kd_ratio=metadata.training_metrics.kills_deaths_ratio,
            win_rate=metadata.training_metrics.win_rate,
            average_episode_length=metadata.training_metrics.average_episode_length,
            average_reward=metadata.training_metrics.average_reward,
            kd_ratio_std=0.0,
            win_rate_std=0.0,
            confidence_interval_95=(0.0, 0.0),
        )

        # Determine promotion reason
        if promotion_achieved:
            promotion_reason = "Met all promotion criteria during training"
        elif final_comparison is not None:
            failed_criteria = [
                k for k, v in final_comparison.criteria_details.items() if not v
            ]
            promotion_reason = f"Failed criteria: {', '.join(failed_criteria)}"
        else:
            promotion_reason = "No evaluation data available"

        return GenerationResult(
            generation=self.current_generation,
            model_id=metadata.model_id,
            model_metadata=metadata,
            final_evaluation=final_evaluation,
            comparison=final_comparison,
            timesteps_trained=metadata.training_metrics.total_timesteps,
            was_promoted=promotion_achieved,
            promotion_reason=promotion_reason,
        )

    async def _get_opponent_baseline(self) -> EvaluationResult | None:
        """Get baseline performance of current opponent.

        For rule-based bot (no opponent_id), returns None.
        For model opponents, retrieves performance from registry.

        Returns:
            EvaluationResult with opponent's baseline, or None for rule-based.
        """
        if self.current_opponent_id is None or self.registry is None:
            return None

        # Get opponent metadata from registry
        _, metadata = self.registry.get_model(self.current_opponent_id)

        # Build evaluation result from training metrics
        return EvaluationResult(
            total_episodes=metadata.training_metrics.total_episodes,
            total_kills=int(
                metadata.training_metrics.average_kills
                * metadata.training_metrics.total_episodes
            ),
            total_deaths=int(
                metadata.training_metrics.average_deaths
                * metadata.training_metrics.total_episodes
            ),
            total_wins=int(
                metadata.training_metrics.win_rate
                * metadata.training_metrics.total_episodes
            ),
            total_losses=int(
                (1 - metadata.training_metrics.win_rate)
                * metadata.training_metrics.total_episodes
            ),
            kd_ratio=metadata.training_metrics.kills_deaths_ratio,
            win_rate=metadata.training_metrics.win_rate,
            average_episode_length=metadata.training_metrics.average_episode_length,
            average_reward=metadata.training_metrics.average_reward,
            kd_ratio_std=0.0,
            win_rate_std=0.0,
            confidence_interval_95=(0.0, 0.0),
        )

    def _log_generation_result(self, result: GenerationResult) -> None:
        """Log the result of a generation.

        Args:
            result: GenerationResult to log.
        """
        logger.info("")
        logger.info("Generation %d Complete:", result.generation)
        logger.info("  Model ID: %s", result.model_id)
        logger.info("  Timesteps: %s", f"{result.timesteps_trained:,}")

        if result.final_evaluation:
            logger.info("  Final K/D: %.2f", result.final_evaluation.kd_ratio)
            logger.info("  Final Win Rate: %.1f%%", result.final_evaluation.win_rate * 100)

        if result.comparison:
            logger.info("  K/D Improvement: %.1f%%", result.comparison.kd_improvement * 100)

        logger.info("  Promoted: %s", result.was_promoted)
        logger.info("  Reason: %s", result.promotion_reason)

    def _log_training_summary(self) -> None:
        """Log summary of all generations."""
        logger.info("=" * 60)
        logger.info("SUCCESSIVE TRAINING COMPLETE")
        logger.info("=" * 60)
        logger.info("Total Generations: %d", len(self.generation_results))

        promotions = [r for r in self.generation_results if r.was_promoted]
        logger.info("Promotions: %d", len(promotions))

        if self.generation_results:
            best = max(
                self.generation_results,
                key=lambda r: (
                    r.final_evaluation.kd_ratio if r.final_evaluation else 0
                ),
            )
            logger.info(
                "Best Model: %s (Gen %d)", best.model_id, best.generation
            )

        total_timesteps = sum(r.timesteps_trained for r in self.generation_results)
        logger.info("Total Timesteps: %s", f"{total_timesteps:,}")

    def register_callback(self, callback: Callable[[dict[str, Any]], None]) -> None:
        """Register a callback for training events.

        Callbacks receive dictionaries with:
        - "type": Event type ("generation_complete", "training_complete")
        - Additional event-specific data

        Args:
            callback: Function to call on training events.
        """
        self._callbacks.append(callback)

    def _invoke_callbacks(self, event: dict[str, Any]) -> None:
        """Invoke all registered callbacks with an event.

        Args:
            event: Event dictionary to pass to callbacks.
        """
        for callback in self._callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.warning("Callback error: %s", e)

    def stop(self) -> None:
        """Signal to stop training after current generation.

        The current generation will complete, but no new generations
        will be started.
        """
        logger.info("Stopping successive training...")
        self.is_running = False

    async def cleanup(self) -> None:
        """Clean up resources.

        This should be called when training is complete or interrupted.
        """
        logger.info("Cleaning up successive trainer...")
        # No persistent resources to clean up

    async def __aenter__(self) -> "SuccessiveTrainer":
        """Async context manager entry."""
        await self.setup()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Async context manager exit."""
        await self.cleanup()

    def get_best_model_id(self) -> str | None:
        """Get the model ID of the best performing model.

        Returns:
            Model ID with highest K/D ratio, or None if no models trained.
        """
        if not self.generation_results:
            return None

        best = max(
            self.generation_results,
            key=lambda r: r.final_evaluation.kd_ratio if r.final_evaluation else 0,
        )
        return best.model_id

    def get_generation_lineage(self) -> list[tuple[str, str | None]]:
        """Get the training lineage (model_id, opponent_id) pairs.

        Returns:
            List of (model_id, opponent_model_id) tuples showing
            which model each generation trained against.
        """
        return [
            (r.model_id, r.model_metadata.opponent_model_id)
            for r in self.generation_results
        ]
