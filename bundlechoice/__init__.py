"""
BundleChoice: Modular bundle choice estimation framework.

This package provides a comprehensive framework for distributed bundle choice estimation
using MPI. It supports various subproblem algorithms and estimation methods for
parameter estimation in discrete choice models.

Main components:
- BundleChoice: Main orchestrator class
- DataManager: Handles data distribution across MPI ranks
- FeatureManager: Manages feature extraction
- SubproblemManager: Manages subproblem solving
- Various estimation solvers (row generation, ellipsoid)
"""

from .core import BundleChoice
from .data_manager import DataManager
from .feature_manager import FeatureManager
from .config import BundleChoiceConfig, DimensionsConfig
from .config_profiles import load_profile, list_profiles

__all__ = [
    'BundleChoice',
    'DataManager', 
    'FeatureManager',
    'BundleChoiceConfig',
    'DimensionsConfig',
    'load_profile',
    'list_profiles'
]
