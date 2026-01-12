"""Training orchestrator for coordinating the RL training pipeline.

This module provides the TrainingOrchestrator class that coordinates all
components of the training pipeline: vectorized environments, PPO training,
checkpointing, and model registration.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import torch

from bot.agent.network import ActorCriticNetwork
from bot.agent.ppo_trainer import PPOTrainer
from bot.gym.vectorized_env import VectorizedTowerfallEnv
from bot.training.orchestrator_config import OrchestratorConfig
from bot.training.registry import ModelRegistry, TrainingMetrics

if TYPE_CHECKING:
    from bot.training.registry import ModelMetadata

logger = logging.getLogger(__name__)


class TrainingOrchestrator:
    """Main orchestrator for RL training pipeline.

    The TrainingOrchestrator coordinates all components of the training
    pipeline including:
    - Vectorized environments for parallel data collection
    - PPO trainer for policy updates
    - Model registry for storing trained models
    - Checkpointing for training recovery
    - Evaluation for monitoring performance

    Example:
        config = OrchestratorConfig(
            num_envs=4,
            total_timesteps=500_000,
        )

        async with TrainingOrchestrator(config) as orchestrator:
            final_metadata = await orchestrator.train()
            print(f"Training complete! Model: {final_metadata.model_id}")

    Attributes:
        config: Orchestrator configuration
        device: Torch device for computation
        env: Vectorized environment instance
        network: Actor-critic neural network
        trainer: PPO trainer instance
        registry: Model registry instance
        total_timesteps: Total timesteps collected so far
        num_updates: Number of PPO updates performed
        current_generation: Current model generation number
        is_running: Whether training is in progress
    """

    def __init__(self, config: OrchestratorConfig) -> None:
        """Initialize the training orchestrator.

        Args:
            config: Orchestrator configuration containing all settings
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Components (initialized in setup)
        self.env: VectorizedTowerfallEnv | None = None
        self.network: ActorCriticNetwork | None = None
        self.trainer: PPOTrainer | None = None
        self.registry: ModelRegistry | None = None

        # Training state
        self.total_timesteps = 0
        self.num_updates = 0
        self.current_generation = 0
        self.is_running = False
        self._training_start_time: float = 0.0

        # Callbacks for training events
        self._callbacks: list[Callable[[dict[str, Any]], None]] = []

    async def setup(self) -> None:
        """Initialize all training components.

        This must be called before train() to set up:
        - Vectorized environments
        - Neural network
        - PPO trainer
        - Model registry

        Raises:
            RuntimeError: If environment setup fails
        """
        logger.info("Setting up training orchestrator...")

        # Set random seeds if specified
        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)
            np.random.seed(self.config.seed)
            logger.info("Random seed set to %d", self.config.seed)

        # Initialize vectorized environments
        self.env = VectorizedTowerfallEnv(
            num_envs=self.config.num_envs,
            http_url=self.config.game_server_url,
            ws_url=self.config.game_server_url.replace("http", "ws") + "/ws",
            player_name="TrainingBot",
            room_name_prefix="Training",
            map_type=self.config.game_config.map_type,
            tick_rate_multiplier=self.config.game_config.tick_multiplier,
            max_episode_steps=self.config.game_config.max_game_duration_sec * 60,
        )

        # Get observation and action dimensions from environment
        obs_dim = self.env.single_observation_space.shape[0]
        action_dim = self.env.single_action_space.n

        # Initialize neural network
        self.network = ActorCriticNetwork(
            observation_size=obs_dim,
            action_size=action_dim,
        ).to(self.device)

        # Initialize PPO trainer
        self.trainer = PPOTrainer(
            network=self.network,
            config=self.config.ppo_config,
            device=self.device,
        )

        # Initialize model registry
        Path(self.config.registry_path).mkdir(parents=True, exist_ok=True)
        self.registry = ModelRegistry(registry_path=self.config.registry_path)

        # Load opponent model if specified
        if self.config.opponent_model_id:
            opponent_model, metadata = self.registry.get_model(
                self.config.opponent_model_id, device=self.device
            )
            self.current_generation = metadata.generation + 1
            logger.info(
                "Loaded opponent model: %s (gen %d)",
                self.config.opponent_model_id,
                metadata.generation,
            )
        else:
            self.current_generation = self.registry.get_next_generation()

        # Create checkpoint directory
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        logger.info(
            "Setup complete. Observation dim: %d, Action dim: %d",
            obs_dim,
            action_dim,
        )
        logger.info("Training on device: %s", self.device)
        logger.info("Current generation: %d", self.current_generation)

    async def train(self) -> "ModelMetadata":
        """Run the main training loop.

        Collects rollouts from vectorized environments and performs PPO updates
        until total_timesteps is reached. Handles checkpointing, evaluation,
        and logging at configured intervals.

        Returns:
            Metadata of the final trained model registered in the registry

        Raises:
            RuntimeError: If setup() was not called before train()
        """
        if self.trainer is None or self.env is None or self.registry is None:
            raise RuntimeError("Must call setup() before train()")
        if self.network is None:
            raise RuntimeError("Network not initialized")

        self.is_running = True
        self._training_start_time = time.time()
        logger.info(
            "Starting training for %d timesteps...", self.config.total_timesteps
        )

        # Reset environments and get initial observations
        obs_array, _ = self.env.reset(seed=self.config.seed)
        obs = torch.as_tensor(obs_array, dtype=torch.float32, device=self.device)

        # Track episode statistics
        episode_rewards: list[float] = []
        episode_lengths: list[int] = []
        episode_kills: list[float] = []
        episode_deaths: list[float] = []

        last_checkpoint_step = 0
        last_log_step = 0
        last_eval_step = 0

        try:
            while (
                self.total_timesteps < self.config.total_timesteps and self.is_running
            ):
                # Collect rollout and perform PPO update
                metrics, obs = self.trainer.train_step(self.env, obs)
                self.num_updates += 1
                self.total_timesteps = self.trainer.total_timesteps

                # Extract episode statistics from environment infos if available
                self._extract_episode_stats(
                    episode_rewards, episode_lengths, episode_kills, episode_deaths
                )

                # Logging
                if self.total_timesteps - last_log_step >= self.config.log_interval:
                    self._log_progress(
                        metrics,
                        episode_rewards,
                        episode_lengths,
                        episode_kills,
                        episode_deaths,
                    )
                    episode_rewards.clear()
                    episode_lengths.clear()
                    episode_kills.clear()
                    episode_deaths.clear()
                    last_log_step = self.total_timesteps

                # Checkpointing
                if (
                    self.total_timesteps - last_checkpoint_step
                    >= self.config.checkpoint_interval
                ):
                    self._save_checkpoint()
                    last_checkpoint_step = self.total_timesteps

                # Evaluation
                if self.total_timesteps - last_eval_step >= self.config.eval_interval:
                    eval_metrics = await self._run_evaluation()
                    self._invoke_callbacks({"type": "evaluation", "metrics": eval_metrics})
                    last_eval_step = self.total_timesteps

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error("Training error: %s", e)
            self._save_checkpoint()
            raise

        # Register final model
        return await self._register_final_model()

    def _extract_episode_stats(
        self,
        episode_rewards: list[float],
        episode_lengths: list[int],
        episode_kills: list[float],
        episode_deaths: list[float],
    ) -> None:
        """Extract episode statistics from stored data.

        Note: In the current implementation, episode statistics are tracked
        through the PPO trainer's rollout collection. This method is a
        placeholder for more sophisticated episode tracking.

        Args:
            episode_rewards: List to append episode rewards to
            episode_lengths: List to append episode lengths to
            episode_kills: List to append episode kills to
            episode_deaths: List to append episode deaths to
        """
        # The PPOTrainer handles rollout collection internally.
        # Episode statistics are derived from the number of updates.
        # In a more sophisticated implementation, we would track
        # individual episode completions from the environment's
        # info dict.
        pass

    def _log_progress(
        self,
        update_metrics: dict[str, Any],
        episode_rewards: list[float],
        episode_lengths: list[int],
        episode_kills: list[float],
        episode_deaths: list[float],
    ) -> None:
        """Log training progress.

        Args:
            update_metrics: Metrics from the last PPO update
            episode_rewards: List of episode rewards since last log
            episode_lengths: List of episode lengths since last log
            episode_kills: List of episode kills since last log
            episode_deaths: List of episode deaths since last log
        """
        avg_reward = float(np.mean(episode_rewards)) if episode_rewards else 0.0
        avg_length = float(np.mean(episode_lengths)) if episode_lengths else 0.0
        avg_kills = float(np.mean(episode_kills)) if episode_kills else 0.0
        avg_deaths = float(np.mean(episode_deaths)) if episode_deaths else 0.0

        elapsed = time.time() - self._training_start_time
        fps = self.total_timesteps / elapsed if elapsed > 0 else 0.0

        metrics = {
            "timesteps": self.total_timesteps,
            "updates": self.num_updates,
            "fps": fps,
            "avg_episode_reward": avg_reward,
            "avg_episode_length": avg_length,
            "avg_kills": avg_kills,
            "avg_deaths": avg_deaths,
            "num_episodes": len(episode_rewards),
            **update_metrics,
        }

        logger.info(
            "Timesteps: %s | Updates: %d | FPS: %.1f | Policy Loss: %.4f | "
            "Value Loss: %.4f | Entropy: %.4f",
            f"{self.total_timesteps:,}",
            self.num_updates,
            fps,
            update_metrics.get("policy_loss", 0),
            update_metrics.get("value_loss", 0),
            update_metrics.get("entropy", 0),
        )

        self._invoke_callbacks({"type": "progress", "metrics": metrics})

    def _save_checkpoint(self) -> None:
        """Save a training checkpoint."""
        if self.network is None or self.trainer is None:
            return

        checkpoint_path = (
            Path(self.config.checkpoint_dir) / f"checkpoint_{self.total_timesteps}.pt"
        )

        torch.save(
            {
                "network_state_dict": self.network.state_dict(),
                "optimizer_state_dict": self.trainer.optimizer.state_dict(),
                "total_timesteps": self.total_timesteps,
                "num_updates": self.num_updates,
                "generation": self.current_generation,
                "config": asdict(self.config.ppo_config),
                "training_start_time": self._training_start_time,
            },
            checkpoint_path,
        )

        logger.info("Saved checkpoint: %s", checkpoint_path)
        self._invoke_callbacks(
            {"type": "checkpoint", "path": str(checkpoint_path)}
        )

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load a training checkpoint to resume training.

        Args:
            checkpoint_path: Path to the checkpoint file

        Raises:
            RuntimeError: If network or trainer is not initialized
        """
        if self.network is None or self.trainer is None:
            raise RuntimeError("Must call setup() before load_checkpoint()")

        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )

        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.total_timesteps = checkpoint["total_timesteps"]
        self.num_updates = checkpoint["num_updates"]
        self.current_generation = checkpoint["generation"]

        # Update trainer's internal counters
        self.trainer.total_timesteps = self.total_timesteps
        self.trainer.num_updates = self.num_updates

        logger.info(
            "Loaded checkpoint from %s at timestep %d",
            checkpoint_path,
            self.total_timesteps,
        )

    async def _run_evaluation(self) -> dict[str, Any]:
        """Run evaluation episodes and compute metrics.

        Returns:
            Dictionary of evaluation metrics
        """
        if self.env is None or self.network is None:
            return {}

        logger.info("Running evaluation (%d episodes)...", self.config.eval_episodes)

        eval_rewards: list[float] = []
        eval_kills: list[float] = []
        eval_deaths: list[float] = []
        eval_lengths: list[int] = []

        for episode in range(self.config.eval_episodes):
            obs_array, _ = self.env.reset()
            obs = torch.as_tensor(obs_array, dtype=torch.float32, device=self.device)

            episode_reward = 0.0
            episode_length = 0
            done = False

            while not done and episode_length < 1000:
                with torch.no_grad():
                    action, _, _, _ = self.network.get_action_and_value(
                        obs, deterministic=True
                    )

                next_obs, reward, terminated, truncated, info = self.env.step(
                    action.cpu().numpy()
                )

                episode_reward += float(reward.sum())
                episode_length += 1
                done = bool(np.any(terminated | truncated))
                obs = torch.as_tensor(
                    next_obs, dtype=torch.float32, device=self.device
                )

                # Extract kills/deaths from info if available
                if "env_infos" in info:
                    for env_info in info["env_infos"]:
                        if "episode_kills" in env_info:
                            eval_kills.append(float(env_info["episode_kills"]))
                        if "episode_deaths" in env_info:
                            eval_deaths.append(float(env_info["episode_deaths"]))

            eval_rewards.append(episode_reward)
            eval_lengths.append(episode_length)

        metrics: dict[str, Any] = {
            "eval_avg_reward": float(np.mean(eval_rewards)),
            "eval_std_reward": float(np.std(eval_rewards)),
            "eval_avg_length": float(np.mean(eval_lengths)),
        }

        if eval_kills:
            metrics["eval_avg_kills"] = float(np.mean(eval_kills))
        if eval_deaths:
            metrics["eval_avg_deaths"] = float(np.mean(eval_deaths))
            if eval_kills:
                avg_deaths = float(np.mean(eval_deaths))
                metrics["eval_kd_ratio"] = (
                    float(np.mean(eval_kills)) / max(avg_deaths, 1.0)
                )

        logger.info(
            "Evaluation: Avg Reward = %.2f, Avg Length = %.1f",
            metrics["eval_avg_reward"],
            metrics["eval_avg_length"],
        )

        return metrics

    async def _register_final_model(self) -> "ModelMetadata":
        """Register the final trained model in the registry.

        Returns:
            Metadata of the registered model

        Raises:
            RuntimeError: If required components are not initialized
        """
        if self.network is None or self.registry is None or self.trainer is None:
            raise RuntimeError("Components not initialized")

        logger.info("Registering final model...")

        # Run final evaluation to get metrics
        eval_metrics = await self._run_evaluation()

        training_duration = time.time() - self._training_start_time

        # Create training metrics
        training_metrics = TrainingMetrics(
            total_episodes=self.num_updates * self.config.num_envs,
            total_timesteps=self.total_timesteps,
            average_reward=eval_metrics.get("eval_avg_reward", 0.0),
            average_episode_length=eval_metrics.get("eval_avg_length", 0.0),
            win_rate=0.0,  # Would need explicit win tracking
            average_kills=eval_metrics.get("eval_avg_kills", 0.0),
            average_deaths=eval_metrics.get("eval_avg_deaths", 0.0),
            kills_deaths_ratio=eval_metrics.get("eval_kd_ratio", 0.0),
        )

        # Register model
        model_id = self.registry.register_model(
            model=self.network,
            generation=self.current_generation,
            opponent_model_id=self.config.opponent_model_id,
            training_metrics=training_metrics,
            hyperparameters=asdict(self.config.ppo_config),
            training_duration_seconds=training_duration,
            optimizer=self.trainer.optimizer,
            training_step=self.num_updates,
        )

        logger.info("Training complete. Model registered as: %s", model_id)

        _, metadata = self.registry.get_model(model_id, device=self.device)
        return metadata

    def register_callback(
        self, callback: Callable[[dict[str, Any]], None]
    ) -> None:
        """Register a callback for training events.

        Callbacks receive dictionaries with:
        - "type": Event type ("progress", "evaluation", "checkpoint")
        - "metrics": Relevant metrics for the event

        Args:
            callback: Function to call on training events
        """
        self._callbacks.append(callback)

    def _invoke_callbacks(self, event: dict[str, Any]) -> None:
        """Invoke all registered callbacks with an event.

        Args:
            event: Event dictionary to pass to callbacks
        """
        for callback in self._callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.warning("Callback error: %s", e)

    def stop(self) -> None:
        """Signal the training loop to stop gracefully."""
        logger.info("Stopping training...")
        self.is_running = False

    async def cleanup(self) -> None:
        """Clean up all resources.

        This should be called when training is complete or interrupted.
        Closes all environments and releases resources.
        """
        logger.info("Cleaning up training resources...")

        if self.env is not None:
            self.env.close()
            self.env = None

        logger.info("Cleanup complete")

    async def __aenter__(self) -> "TrainingOrchestrator":
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
