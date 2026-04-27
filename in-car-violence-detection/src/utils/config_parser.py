"""
Configuration parser for YAML configs.
"""

import yaml
from pathlib import Path
from typing import Dict, Any


class Config:
    """Simple configuration container with dot notation access."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        self._config = config_dict
        
    def __getattr__(self, name: str) -> Any:
        if name.startswith('_'):
            return object.__getattribute__(self, name)
        
        value = self._config.get(name)
        if isinstance(value, dict):
            return Config(value)
        return value
    
    def __getitem__(self, key: str) -> Any:
        return self._config[key]
    
    def get(self, key: str, default: Any = None) -> Any:
        return self._config.get(key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        return self._config
    
    def __repr__(self) -> str:
        return f"Config({self._config})"


def load_config(config_path: str) -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Config object with dot notation access
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return Config(config_dict)


def merge_configs(base_config: Config, override_config: Config) -> Config:
    """
    Merge two configurations, with override taking precedence.
    """
    merged = {**base_config.to_dict(), **override_config.to_dict()}
    return Config(merged)
