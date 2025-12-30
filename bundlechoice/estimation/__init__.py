"""
Estimation module for modular bundle choice estimation (v2).

This module provides various estimation algorithms for parameter estimation
in bundle choice models, including row generation and ellipsoid methods.
"""

from .base import BaseEstimationManager
from .row_generation import RowGenerationManager
from .row_generation_1slack import RowGeneration1SlackManager
from .ellipsoid import EllipsoidManager
from .column_generation import ColumnGenerationManager
from .inequalities import InequalitiesManager
from .callbacks import adaptive_gurobi_timeout, constant_timeout
from .result import EstimationResult

__all__ = [
    'BaseEstimationManager',
    'RowGenerationManager', 
    'RowGeneration1SlackManager',
    'EllipsoidManager',
    'ColumnGenerationManager',
    'InequalitiesManager',
    'EstimationResult',
    'adaptive_gurobi_timeout',
    'constant_timeout',
]