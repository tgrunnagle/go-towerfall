"""
Configuration manager for the RL bot system.
Handles loading, validation, and management of configuration files.
"""
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
from .base_config import RLBotSystemConfig


class ConfigManager:
    """Manages configuration loading and validation for the RL bot system."""
    
    def __init__(self, config_dir: Union[str, Path] = "config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self._configs: Dict[str, RLBotSystemConfig] = {}
        self._default_config: Optional[RLBotSystemConfig] = None
    
    def load_config(self, config_name: str = "default") -> RLBotSystemConfig:
        """Load a configuration by name."""
        if config_name in self._configs:
            return self._configs[config_name]
        
        config_path = self.config_dir / f"{config_name}.yaml"
        
        if not config_path.exists():
            if config_name == "default":
                # Create default configuration if it doesn't exist
                config = RLBotSystemConfig()
                self.save_config(config, config_name)
                self._configs[config_name] = config
                return config
            else:
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        config = RLBotSystemConfig.from_yaml(config_path)
        config.validate()
        self._configs[config_name] = config
        return config
    
    def save_config(self, config: RLBotSystemConfig, config_name: str = "default") -> None:
        """Save a configuration with the given name."""
        config_path = self.config_dir / f"{config_name}.yaml"
        config.save_yaml(config_path)
        self._configs[config_name] = config
    
    def get_default_config(self) -> RLBotSystemConfig:
        """Get the default configuration."""
        if self._default_config is None:
            self._default_config = self.load_config("default")
        return self._default_config
    
    def create_config_from_template(self, template_name: str, new_config_name: str, 
                                  overrides: Optional[Dict[str, Any]] = None) -> RLBotSystemConfig:
        """Create a new configuration based on a template with optional overrides."""
        template_config = self.load_config(template_name)
        config_dict = template_config.to_dict()
        
        if overrides:
            config_dict = self._deep_update(config_dict, overrides)
        
        new_config = RLBotSystemConfig.from_dict(config_dict)
        new_config.validate()
        self.save_config(new_config, new_config_name)
        return new_config
    
    def list_configs(self) -> list[str]:
        """List all available configuration files."""
        config_files = list(self.config_dir.glob("*.yaml"))
        return [f.stem for f in config_files]
    
    def delete_config(self, config_name: str) -> None:
        """Delete a configuration file."""
        if config_name == "default":
            raise ValueError("Cannot delete the default configuration")
        
        config_path = self.config_dir / f"{config_name}.yaml"
        if config_path.exists():
            config_path.unlink()
        
        if config_name in self._configs:
            del self._configs[config_name]
    
    def validate_all_configs(self) -> Dict[str, bool]:
        """Validate all configuration files and return results."""
        results = {}
        for config_name in self.list_configs():
            try:
                config = self.load_config(config_name)
                config.validate()
                results[config_name] = True
            except Exception as e:
                results[config_name] = False
                print(f"Configuration '{config_name}' validation failed: {e}")
        return results
    
    def get_config_for_algorithm(self, algorithm: str) -> RLBotSystemConfig:
        """Get or create a configuration optimized for a specific algorithm."""
        config_name = f"{algorithm.lower()}_optimized"
        
        if config_name not in self.list_configs():
            # Create algorithm-specific configuration
            base_config = self.get_default_config()
            overrides = self._get_algorithm_overrides(algorithm)
            return self.create_config_from_template("default", config_name, overrides)
        
        return self.load_config(config_name)
    
    def _deep_update(self, base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively update a nested dictionary."""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                base_dict[key] = self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
        return base_dict
    
    def _get_algorithm_overrides(self, algorithm: str) -> Dict[str, Any]:
        """Get algorithm-specific configuration overrides."""
        overrides = {
            'training': {
                'algorithm': algorithm
            }
        }
        
        if algorithm == "DQN":
            overrides['training'].update({
                'batch_size': 32,
                'buffer_size': 100000,
                'learning_rate': 1e-4,
                'exploration_fraction': 0.1,
                'exploration_initial_eps': 1.0,
                'exploration_final_eps': 0.05,
                'train_freq': 4,
                'target_update_interval': 1000
            })
        elif algorithm == "PPO":
            overrides['training'].update({
                'batch_size': 64,
                'learning_rate': 3e-4,
                'n_steps': 2048,
                'n_epochs': 10,
                'clip_range': 0.2,
                'ent_coef': 0.0,
                'vf_coef': 0.5
            })
        elif algorithm == "A3C":
            overrides['training'].update({
                'learning_rate': 1e-4,
                'n_steps': 5,
                'gamma': 0.99,
                'ent_coef': 0.01,
                'vf_coef': 0.25
            })
        elif algorithm == "SAC":
            overrides['training'].update({
                'learning_rate': 3e-4,
                'buffer_size': 1000000,
                'batch_size': 256,
                'tau': 0.005,
                'gamma': 0.99,
                'train_freq': 1
            })
        
        return overrides


# Global configuration manager instance
config_manager = ConfigManager()