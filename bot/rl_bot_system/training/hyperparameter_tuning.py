"""
Hyperparameter tuning system for RL bot training.

This module provides automated hyperparameter optimization using various strategies
including grid search, random search, and Bayesian optimization.
"""

import logging
import numpy as np
import json
from typing import Dict, Any, List, Tuple, Optional, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import itertools
import random
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class HyperparameterSpace:
    """Definition of hyperparameter search space."""
    name: str
    param_type: str  # 'continuous', 'discrete', 'categorical'
    bounds: Optional[Tuple[float, float]] = None  # For continuous
    choices: Optional[List[Any]] = None  # For discrete/categorical
    log_scale: bool = False  # Whether to use log scale for continuous params
    
    def sample(self) -> Any:
        """Sample a value from this parameter space."""
        if self.param_type == 'continuous':
            if self.bounds is None:
                raise ValueError("Bounds required for continuous parameters")
            
            low, high = self.bounds
            if self.log_scale:
                log_low, log_high = np.log10(low), np.log10(high)
                value = 10 ** np.random.uniform(log_low, log_high)
            else:
                value = np.random.uniform(low, high)
            return value
            
        elif self.param_type == 'discrete':
            if self.choices is None:
                raise ValueError("Choices required for discrete parameters")
            return random.choice(self.choices)
            
        elif self.param_type == 'categorical':
            if self.choices is None:
                raise ValueError("Choices required for categorical parameters")
            return random.choice(self.choices)
            
        else:
            raise ValueError(f"Unknown parameter type: {self.param_type}")


@dataclass
class HyperparameterConfig:
    """Complete hyperparameter configuration."""
    parameters: Dict[str, Any]
    score: Optional[float] = None
    training_time: Optional[float] = None
    evaluation_episodes: Optional[int] = None
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class GridSearchTuner:
    """
    Grid search hyperparameter tuner.
    
    Exhaustively searches through all combinations of specified parameter values.
    """
    
    def __init__(self, parameter_spaces: Dict[str, HyperparameterSpace]):
        """
        Initialize grid search tuner.
        
        Args:
            parameter_spaces: Dictionary of parameter names to search spaces
        """
        self.parameter_spaces = parameter_spaces
        self.results: List[HyperparameterConfig] = []
        
    def generate_configurations(self) -> List[Dict[str, Any]]:
        """
        Generate all possible parameter combinations for grid search.
        
        Returns:
            List of parameter dictionaries
        """
        # Get all parameter values to try
        param_values = {}
        for name, space in self.parameter_spaces.items():
            if space.param_type in ['discrete', 'categorical']:
                if space.choices is None:
                    raise ValueError(f"Choices required for parameter {name}")
                param_values[name] = space.choices
            else:
                # For continuous parameters, we need to discretize
                if space.bounds is None:
                    raise ValueError(f"Bounds required for continuous parameter {name}")
                
                low, high = space.bounds
                if space.log_scale:
                    # Create log-spaced values
                    values = np.logspace(np.log10(low), np.log10(high), num=5)
                else:
                    # Create linearly spaced values
                    values = np.linspace(low, high, num=5)
                param_values[name] = values.tolist()
        
        # Generate all combinations
        param_names = list(param_values.keys())
        param_combinations = list(itertools.product(*[param_values[name] for name in param_names]))
        
        configurations = []
        for combination in param_combinations:
            config = dict(zip(param_names, combination))
            configurations.append(config)
        
        return configurations


class RandomSearchTuner:
    """
    Random search hyperparameter tuner.
    
    Randomly samples parameter combinations from the search space.
    """
    
    def __init__(self, parameter_spaces: Dict[str, HyperparameterSpace], n_trials: int = 50):
        """
        Initialize random search tuner.
        
        Args:
            parameter_spaces: Dictionary of parameter names to search spaces
            n_trials: Number of random configurations to try
        """
        self.parameter_spaces = parameter_spaces
        self.n_trials = n_trials
        self.results: List[HyperparameterConfig] = []
        
    def generate_configurations(self) -> List[Dict[str, Any]]:
        """
        Generate random parameter combinations.
        
        Returns:
            List of parameter dictionaries
        """
        configurations = []
        
        for _ in range(self.n_trials):
            config = {}
            for name, space in self.parameter_spaces.items():
                config[name] = space.sample()
            configurations.append(config)
        
        return configurations


class BayesianOptimizationTuner:
    """
    Bayesian optimization hyperparameter tuner.
    
    Uses Gaussian Process to model the objective function and select
    promising parameter combinations.
    
    Note: This is a simplified implementation. For production use,
    consider using libraries like scikit-optimize or Optuna.
    """
    
    def __init__(
        self, 
        parameter_spaces: Dict[str, HyperparameterSpace], 
        n_trials: int = 50,
        n_initial_points: int = 10
    ):
        """
        Initialize Bayesian optimization tuner.
        
        Args:
            parameter_spaces: Dictionary of parameter names to search spaces
            n_trials: Total number of trials
            n_initial_points: Number of random initial points
        """
        self.parameter_spaces = parameter_spaces
        self.n_trials = n_trials
        self.n_initial_points = n_initial_points
        self.results: List[HyperparameterConfig] = []
        self.tried_configs: List[Dict[str, Any]] = []
        
    def generate_configurations(self) -> List[Dict[str, Any]]:
        """
        Generate parameter combinations using Bayesian optimization.
        
        Returns:
            List of parameter dictionaries
        """
        configurations = []
        
        # Start with random initial points
        for _ in range(min(self.n_initial_points, self.n_trials)):
            config = {}
            for name, space in self.parameter_spaces.items():
                config[name] = space.sample()
            configurations.append(config)
        
        # For remaining trials, use acquisition function (simplified)
        for _ in range(self.n_trials - len(configurations)):
            # Simplified acquisition: sample around best performing configs
            if self.results:
                # Find best configuration so far
                best_config = max(self.results, key=lambda x: x.score or -float('inf'))
                
                # Add noise to best configuration
                config = {}
                for name, space in self.parameter_spaces.items():
                    if name in best_config.parameters:
                        base_value = best_config.parameters[name]
                        
                        if space.param_type == 'continuous':
                            # Add Gaussian noise
                            if space.bounds:
                                noise_scale = (space.bounds[1] - space.bounds[0]) * 0.1
                                value = np.random.normal(base_value, noise_scale)
                                value = np.clip(value, space.bounds[0], space.bounds[1])
                            else:
                                value = base_value
                        else:
                            # For discrete/categorical, occasionally sample randomly
                            if np.random.random() < 0.3:
                                value = space.sample()
                            else:
                                value = base_value
                        
                        config[name] = value
                    else:
                        config[name] = space.sample()
            else:
                # No results yet, sample randomly
                config = {}
                for name, space in self.parameter_spaces.items():
                    config[name] = space.sample()
            
            configurations.append(config)
        
        return configurations
    
    def update_results(self, config: Dict[str, Any], score: float):
        """
        Update the tuner with new results.
        
        Args:
            config: Parameter configuration that was tried
            score: Performance score achieved
        """
        hyperconfig = HyperparameterConfig(parameters=config, score=score)
        self.results.append(hyperconfig)
        self.tried_configs.append(config)


class HyperparameterTuner:
    """
    Main hyperparameter tuning orchestrator.
    
    Supports multiple tuning strategies and provides a unified interface
    for hyperparameter optimization.
    """
    
    def __init__(
        self,
        algorithm: str,
        tuning_strategy: str = 'random',
        n_trials: int = 50,
        results_dir: str = "data/hyperparameter_tuning"
    ):
        """
        Initialize hyperparameter tuner.
        
        Args:
            algorithm: RL algorithm to tune ('DQN', 'PPO', 'A2C')
            tuning_strategy: Tuning strategy ('grid', 'random', 'bayesian')
            n_trials: Number of trials for random/bayesian search
            results_dir: Directory to save tuning results
        """
        self.algorithm = algorithm
        self.tuning_strategy = tuning_strategy
        self.n_trials = n_trials
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Define parameter spaces for each algorithm
        self.parameter_spaces = self._get_parameter_spaces(algorithm)
        
        # Initialize tuner
        if tuning_strategy == 'grid':
            self.tuner = GridSearchTuner(self.parameter_spaces)
        elif tuning_strategy == 'random':
            self.tuner = RandomSearchTuner(self.parameter_spaces, n_trials)
        elif tuning_strategy == 'bayesian':
            self.tuner = BayesianOptimizationTuner(self.parameter_spaces, n_trials)
        else:
            raise ValueError(f"Unknown tuning strategy: {tuning_strategy}")
        
        self.results: List[HyperparameterConfig] = []
        
    def _get_parameter_spaces(self, algorithm: str) -> Dict[str, HyperparameterSpace]:
        """
        Get parameter search spaces for the specified algorithm.
        
        Args:
            algorithm: RL algorithm name
            
        Returns:
            Dictionary of parameter spaces
        """
        if algorithm == 'DQN':
            return {
                'learning_rate': HyperparameterSpace(
                    'learning_rate', 'continuous', bounds=(1e-5, 1e-2), log_scale=True
                ),
                'gamma': HyperparameterSpace(
                    'gamma', 'continuous', bounds=(0.9, 0.999)
                ),
                'epsilon_decay': HyperparameterSpace(
                    'epsilon_decay', 'continuous', bounds=(0.99, 0.9999)
                ),
                'batch_size': HyperparameterSpace(
                    'batch_size', 'discrete', choices=[16, 32, 64, 128]
                ),
                'target_update_freq': HyperparameterSpace(
                    'target_update_freq', 'discrete', choices=[500, 1000, 2000, 5000]
                ),
                'double_dqn': HyperparameterSpace(
                    'double_dqn', 'categorical', choices=[True, False]
                ),
                'dueling': HyperparameterSpace(
                    'dueling', 'categorical', choices=[True, False]
                )
            }
            
        elif algorithm == 'PPO':
            return {
                'learning_rate': HyperparameterSpace(
                    'learning_rate', 'continuous', bounds=(1e-5, 1e-2), log_scale=True
                ),
                'gamma': HyperparameterSpace(
                    'gamma', 'continuous', bounds=(0.9, 0.999)
                ),
                'gae_lambda': HyperparameterSpace(
                    'gae_lambda', 'continuous', bounds=(0.9, 0.99)
                ),
                'clip_epsilon': HyperparameterSpace(
                    'clip_epsilon', 'continuous', bounds=(0.1, 0.3)
                ),
                'value_coef': HyperparameterSpace(
                    'value_coef', 'continuous', bounds=(0.1, 1.0)
                ),
                'entropy_coef': HyperparameterSpace(
                    'entropy_coef', 'continuous', bounds=(0.001, 0.1), log_scale=True
                ),
                'ppo_epochs': HyperparameterSpace(
                    'ppo_epochs', 'discrete', choices=[3, 4, 5, 8, 10]
                ),
                'batch_size': HyperparameterSpace(
                    'batch_size', 'discrete', choices=[32, 64, 128, 256]
                ),
                'buffer_size': HyperparameterSpace(
                    'buffer_size', 'discrete', choices=[1024, 2048, 4096]
                )
            }
            
        elif algorithm == 'A2C':
            return {
                'learning_rate': HyperparameterSpace(
                    'learning_rate', 'continuous', bounds=(1e-5, 1e-2), log_scale=True
                ),
                'gamma': HyperparameterSpace(
                    'gamma', 'continuous', bounds=(0.9, 0.999)
                ),
                'value_coef': HyperparameterSpace(
                    'value_coef', 'continuous', bounds=(0.1, 1.0)
                ),
                'entropy_coef': HyperparameterSpace(
                    'entropy_coef', 'continuous', bounds=(0.001, 0.1), log_scale=True
                ),
                'n_steps': HyperparameterSpace(
                    'n_steps', 'discrete', choices=[3, 5, 8, 10, 16]
                )
            }
            
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    
    def tune(
        self, 
        objective_function: Callable[[Dict[str, Any]], float],
        maximize: bool = True
    ) -> HyperparameterConfig:
        """
        Run hyperparameter tuning.
        
        Args:
            objective_function: Function that takes hyperparameters and returns score
            maximize: Whether to maximize or minimize the objective
            
        Returns:
            Best hyperparameter configuration found
        """
        logger.info(f"Starting {self.tuning_strategy} hyperparameter tuning for {self.algorithm}")
        logger.info(f"Number of trials: {self.n_trials}")
        
        # Generate configurations to try
        configurations = self.tuner.generate_configurations()
        
        best_config = None
        best_score = -float('inf') if maximize else float('inf')
        
        for i, config in enumerate(configurations):
            logger.info(f"Trial {i + 1}/{len(configurations)}: {config}")
            
            try:
                # Evaluate configuration
                score = objective_function(config)
                
                # Create result
                result = HyperparameterConfig(
                    parameters=config,
                    score=score
                )
                self.results.append(result)
                
                # Update tuner if it supports it (Bayesian optimization)
                if hasattr(self.tuner, 'update_results'):
                    self.tuner.update_results(config, score)
                
                # Check if this is the best so far
                is_better = (maximize and score > best_score) or (not maximize and score < best_score)
                if is_better:
                    best_score = score
                    best_config = result
                
                logger.info(f"Trial {i + 1} score: {score:.4f} (best so far: {best_score:.4f})")
                
            except Exception as e:
                logger.error(f"Trial {i + 1} failed: {e}")
                continue
        
        # Save results
        self.save_results()
        
        logger.info(f"Hyperparameter tuning completed. Best score: {best_score:.4f}")
        logger.info(f"Best configuration: {best_config.parameters}")
        
        return best_config
    
    def save_results(self):
        """Save tuning results to file."""
        results_file = self.results_dir / f"{self.algorithm}_{self.tuning_strategy}_results.json"
        
        results_data = {
            'algorithm': self.algorithm,
            'tuning_strategy': self.tuning_strategy,
            'n_trials': self.n_trials,
            'results': [asdict(result) for result in self.results],
            'parameter_spaces': {
                name: {
                    'param_type': space.param_type,
                    'bounds': space.bounds,
                    'choices': space.choices,
                    'log_scale': space.log_scale
                }
                for name, space in self.parameter_spaces.items()
            }
        }
        
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        logger.info(f"Results saved to {results_file}")
    
    def load_results(self, results_file: str):
        """Load previous tuning results."""
        with open(results_file, 'r') as f:
            results_data = json.load(f)
        
        self.results = []
        for result_data in results_data.get('results', []):
            result_data['timestamp'] = datetime.fromisoformat(result_data['timestamp'])
            self.results.append(HyperparameterConfig(**result_data))
        
        logger.info(f"Loaded {len(self.results)} previous results")
    
    def get_best_config(self, maximize: bool = True) -> Optional[HyperparameterConfig]:
        """
        Get the best configuration from tuning results.
        
        Args:
            maximize: Whether to maximize or minimize the score
            
        Returns:
            Best configuration or None if no results
        """
        if not self.results:
            return None
        
        valid_results = [r for r in self.results if r.score is not None]
        if not valid_results:
            return None
        
        if maximize:
            return max(valid_results, key=lambda x: x.score)
        else:
            return min(valid_results, key=lambda x: x.score)
    
    def get_top_configs(self, n: int = 5, maximize: bool = True) -> List[HyperparameterConfig]:
        """
        Get the top N configurations from tuning results.
        
        Args:
            n: Number of top configurations to return
            maximize: Whether to maximize or minimize the score
            
        Returns:
            List of top configurations
        """
        valid_results = [r for r in self.results if r.score is not None]
        if not valid_results:
            return []
        
        sorted_results = sorted(valid_results, key=lambda x: x.score, reverse=maximize)
        return sorted_results[:n]