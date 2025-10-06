"""
Configuration Package
Agent configuration and presets
"""

from .settings import AgentConfig
from .presets import (
    get_development_config,
    get_production_config,
    get_minimal_config,
    get_testing_config
)

__all__ = [
    'AgentConfig',
    'get_development_config',
    'get_production_config',
    'get_minimal_config',
    'get_testing_config',
]

