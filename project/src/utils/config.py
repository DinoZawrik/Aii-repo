"""
Configuration management module.

This module handles loading and managing configuration from environment variables
and configuration files.
"""

import os
from typing import Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv
import yaml


class Config:
    """Configuration manager for the sentiment analysis project."""

    def __init__(self, env_file: Optional[str] = None):
        """
        Initialize configuration.

        Args:
            env_file: Path to .env file. If None, uses default .env in project root.
        """
        # Load environment variables
        if env_file:
            load_dotenv(env_file)
        else:
            # Try to load from project root
            project_root = Path(__file__).parent.parent.parent
            env_path = project_root / '.env'
            if env_path.exists():
                load_dotenv(env_path)

        # Model configuration
        self.model_type = os.getenv('MODEL_TYPE', 'baseline')
        self.model_path = os.getenv('MODEL_PATH', 'artifacts/models/baseline/')

        # API configuration
        self.api_host = os.getenv('API_HOST', '0.0.0.0')
        self.api_port = int(os.getenv('API_PORT', '8000'))

        # Logging configuration
        self.log_level = os.getenv('LOG_LEVEL', 'INFO')
        self.log_format = os.getenv('LOG_FORMAT', 'console')

        # Training configuration
        self.batch_size = int(os.getenv('BATCH_SIZE', '16'))
        self.learning_rate = float(os.getenv('LEARNING_RATE', '2e-5'))
        self.epochs = int(os.getenv('EPOCHS', '3'))
        self.max_length = int(os.getenv('MAX_LENGTH', '512'))

    @staticmethod
    def load_yaml(config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to YAML configuration file.

        Returns:
            Dictionary with configuration parameters.
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    @staticmethod
    def save_yaml(config: Dict[str, Any], config_path: str) -> None:
        """
        Save configuration to YAML file.

        Args:
            config: Dictionary with configuration parameters.
            config_path: Path to save YAML file.
        """
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns:
            Dictionary with all configuration parameters.
        """
        return {
            'model_type': self.model_type,
            'model_path': self.model_path,
            'api_host': self.api_host,
            'api_port': self.api_port,
            'log_level': self.log_level,
            'log_format': self.log_format,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'max_length': self.max_length,
        }

    def __repr__(self) -> str:
        """String representation of configuration."""
        return f"Config({self.to_dict()})"


# Global configuration instance
_config: Optional[Config] = None


def get_config() -> Config:
    """
    Get global configuration instance (singleton pattern).

    Returns:
        Global Config instance.
    """
    global _config
    if _config is None:
        _config = Config()
    return _config
