"""
Configuration management for training metrics server.

This module provides configuration loading, validation, and management
for the training metrics server and its integration components.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict, field

logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Database configuration for metrics storage."""
    enabled: bool = False
    url: str = "sqlite:///data/training_metrics.db"
    pool_size: int = 5
    max_overflow: int = 10
    echo: bool = False


@dataclass
class RedisConfig:
    """Redis configuration for caching and pub/sub."""
    enabled: bool = False
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    max_connections: int = 10


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    max_file_size_mb: int = 100
    backup_count: int = 5


@dataclass
class SecurityConfig:
    """Security configuration."""
    enable_auth: bool = False
    jwt_secret_key: Optional[str] = None
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24
    cors_origins: List[str] = field(default_factory=lambda: ["http://localhost:3000", "http://localhost:4000"])
    max_connections_per_ip: int = 100


@dataclass
class IntegrationConfig:
    """Integration configuration for external services."""
    game_server_url: str = "http://localhost:4000"
    enable_spectator_integration: bool = True
    enable_training_engine_integration: bool = True
    
    # Webhook configurations
    webhook_endpoints: List[str] = field(default_factory=list)
    webhook_timeout_seconds: int = 30
    
    # External API configurations
    external_apis: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class PerformanceConfig:
    """Performance and resource configuration."""
    max_connections_per_session: int = 50
    max_total_connections: int = 1000
    metrics_history_size: int = 10000
    cleanup_interval_seconds: int = 300
    
    # Memory management
    max_memory_usage_mb: int = 1024
    enable_memory_monitoring: bool = True
    
    # Rate limiting
    enable_rate_limiting: bool = True
    requests_per_minute: int = 1000
    
    # Caching
    enable_caching: bool = True
    cache_ttl_seconds: int = 300


@dataclass
class ServerConfiguration:
    """Complete server configuration."""
    # Basic server settings
    host: str = "localhost"
    port: int = 8000
    debug: bool = False
    
    # Component configurations
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    integration: IntegrationConfig = field(default_factory=IntegrationConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    
    # Data storage
    data_storage_path: str = "data/training_metrics"
    enable_data_persistence: bool = True
    
    @classmethod
    def from_file(cls, config_path: str) -> 'ServerConfiguration':
        """
        Load configuration from a JSON file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            ServerConfiguration instance
        """
        config_file = Path(config_path)
        
        if not config_file.exists():
            logger.warning(f"Configuration file {config_path} not found, using defaults")
            return cls()
        
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            return cls.from_dict(config_data)
        
        except Exception as e:
            logger.error(f"Failed to load configuration from {config_path}: {e}")
            logger.info("Using default configuration")
            return cls()
    
    @classmethod
    def from_dict(cls, config_data: Dict[str, Any]) -> 'ServerConfiguration':
        """
        Create configuration from a dictionary.
        
        Args:
            config_data: Configuration dictionary
            
        Returns:
            ServerConfiguration instance
        """
        # Extract nested configurations
        database_config = DatabaseConfig(**config_data.get('database', {}))
        redis_config = RedisConfig(**config_data.get('redis', {}))
        logging_config = LoggingConfig(**config_data.get('logging', {}))
        security_config = SecurityConfig(**config_data.get('security', {}))
        integration_config = IntegrationConfig(**config_data.get('integration', {}))
        performance_config = PerformanceConfig(**config_data.get('performance', {}))
        
        # Create main configuration
        main_config = {k: v for k, v in config_data.items() 
                      if k not in ['database', 'redis', 'logging', 'security', 'integration', 'performance']}
        
        return cls(
            database=database_config,
            redis=redis_config,
            logging=logging_config,
            security=security_config,
            integration=integration_config,
            performance=performance_config,
            **main_config
        )
    
    @classmethod
    def from_environment(cls) -> 'ServerConfiguration':
        """
        Load configuration from environment variables.
        
        Returns:
            ServerConfiguration instance
        """
        config = cls()
        
        # Basic server settings
        config.host = os.getenv('TRAINING_METRICS_HOST', config.host)
        config.port = int(os.getenv('TRAINING_METRICS_PORT', str(config.port)))
        config.debug = os.getenv('TRAINING_METRICS_DEBUG', 'false').lower() == 'true'
        
        # Database settings
        if os.getenv('DATABASE_URL'):
            config.database.enabled = True
            config.database.url = os.getenv('DATABASE_URL')
        
        # Redis settings
        if os.getenv('REDIS_URL'):
            config.redis.enabled = True
            redis_url = os.getenv('REDIS_URL')
            # Parse Redis URL if needed
            config.redis.host = os.getenv('REDIS_HOST', config.redis.host)
            config.redis.port = int(os.getenv('REDIS_PORT', str(config.redis.port)))
            config.redis.password = os.getenv('REDIS_PASSWORD')
        
        # Security settings
        config.security.jwt_secret_key = os.getenv('JWT_SECRET_KEY')
        if os.getenv('CORS_ORIGINS'):
            config.security.cors_origins = os.getenv('CORS_ORIGINS').split(',')
        
        # Integration settings
        config.integration.game_server_url = os.getenv('GAME_SERVER_URL', config.integration.game_server_url)
        
        # Performance settings
        config.performance.max_connections_per_session = int(
            os.getenv('MAX_CONNECTIONS_PER_SESSION', str(config.performance.max_connections_per_session))
        )
        
        # Data storage
        config.data_storage_path = os.getenv('DATA_STORAGE_PATH', config.data_storage_path)
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Configuration dictionary
        """
        return asdict(self)
    
    def save_to_file(self, config_path: str) -> None:
        """
        Save configuration to a JSON file.
        
        Args:
            config_path: Path to save the configuration file
        """
        config_file = Path(config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(config_file, 'w') as f:
                json.dump(self.to_dict(), f, indent=2, default=str)
            
            logger.info(f"Configuration saved to {config_path}")
        
        except Exception as e:
            logger.error(f"Failed to save configuration to {config_path}: {e}")
    
    def validate(self) -> List[str]:
        """
        Validate configuration settings.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Validate port range
        if not (1 <= self.port <= 65535):
            errors.append(f"Invalid port number: {self.port}")
        
        # Validate paths
        try:
            Path(self.data_storage_path).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            errors.append(f"Invalid data storage path: {e}")
        
        # Validate security settings
        if self.security.enable_auth and not self.security.jwt_secret_key:
            errors.append("JWT secret key is required when authentication is enabled")
        
        # Validate performance settings
        if self.performance.max_connections_per_session <= 0:
            errors.append("Max connections per session must be positive")
        
        if self.performance.metrics_history_size <= 0:
            errors.append("Metrics history size must be positive")
        
        # Validate integration settings
        if not self.integration.game_server_url:
            errors.append("Game server URL is required")
        
        return errors
    
    def setup_logging(self) -> None:
        """Setup logging based on configuration."""
        log_config = self.logging
        
        # Configure logging level
        logging.basicConfig(
            level=getattr(logging, log_config.level.upper()),
            format=log_config.format
        )
        
        # Add file handler if specified
        if log_config.file_path:
            from logging.handlers import RotatingFileHandler
            
            log_file = Path(log_config.file_path)
            log_file.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=log_config.max_file_size_mb * 1024 * 1024,
                backupCount=log_config.backup_count
            )
            file_handler.setFormatter(logging.Formatter(log_config.format))
            
            root_logger = logging.getLogger()
            root_logger.addHandler(file_handler)


class ConfigurationManager:
    """
    Manages configuration loading and validation.
    
    Provides utilities for loading configuration from multiple sources
    and managing configuration updates.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config/server.json"
        self._config: Optional[ServerConfiguration] = None
    
    def load_config(self) -> ServerConfiguration:
        """
        Load configuration from file, environment, or defaults.
        
        Returns:
            ServerConfiguration instance
        """
        if self._config is not None:
            return self._config
        
        # Try to load from file first
        if Path(self.config_path).exists():
            config = ServerConfiguration.from_file(self.config_path)
        else:
            # Try environment variables
            config = ServerConfiguration.from_environment()
        
        # Validate configuration
        errors = config.validate()
        if errors:
            logger.error("Configuration validation errors:")
            for error in errors:
                logger.error(f"  - {error}")
            raise ValueError("Invalid configuration")
        
        # Setup logging
        config.setup_logging()
        
        self._config = config
        return config
    
    def reload_config(self) -> ServerConfiguration:
        """
        Reload configuration from file.
        
        Returns:
            Updated ServerConfiguration instance
        """
        self._config = None
        return self.load_config()
    
    def get_config(self) -> ServerConfiguration:
        """
        Get current configuration (load if not already loaded).
        
        Returns:
            ServerConfiguration instance
        """
        if self._config is None:
            return self.load_config()
        return self._config
    
    def update_config(self, updates: Dict[str, Any]) -> ServerConfiguration:
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary of configuration updates
            
        Returns:
            Updated ServerConfiguration instance
        """
        current_config = self.get_config()
        config_dict = current_config.to_dict()
        
        # Apply updates (nested dictionary merge)
        self._deep_update(config_dict, updates)
        
        # Create new configuration
        new_config = ServerConfiguration.from_dict(config_dict)
        
        # Validate
        errors = new_config.validate()
        if errors:
            raise ValueError(f"Invalid configuration updates: {errors}")
        
        self._config = new_config
        return new_config
    
    def save_config(self) -> None:
        """Save current configuration to file."""
        if self._config is None:
            raise ValueError("No configuration loaded")
        
        self._config.save_to_file(self.config_path)
    
    def _deep_update(self, base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> None:
        """
        Recursively update a nested dictionary.
        
        Args:
            base_dict: Base dictionary to update
            update_dict: Updates to apply
        """
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value


# Global configuration manager instance
config_manager = ConfigurationManager()


def get_config() -> ServerConfiguration:
    """
    Get the global server configuration.
    
    Returns:
        ServerConfiguration instance
    """
    return config_manager.get_config()


def load_config(config_path: Optional[str] = None) -> ServerConfiguration:
    """
    Load configuration from specified path or defaults.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        ServerConfiguration instance
    """
    if config_path:
        manager = ConfigurationManager(config_path)
        return manager.load_config()
    else:
        return config_manager.load_config()