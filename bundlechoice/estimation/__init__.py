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

# Backward compatibility aliases
BaseEstimationSolver = BaseEstimationManager
RowGenerationSolver = RowGenerationManager
RowGeneration1SlackSolver = RowGeneration1SlackManager
EllipsoidSolver = EllipsoidManager
ColumnGenerationSolver = ColumnGenerationManager
InequalitiesSolver = InequalitiesManager

__all__ = [
    'BaseEstimationManager',
    'RowGenerationManager', 
    'RowGeneration1SlackManager',
    'EllipsoidManager',
    'ColumnGenerationManager',
    'InequalitiesManager',
    # Backward compatibility
    'BaseEstimationSolver',
    'RowGenerationSolver',
    'RowGeneration1SlackSolver',
    'EllipsoidSolver',
    'ColumnGenerationSolver',
    'InequalitiesSolver',
    'adaptive_gurobi_timeout',
    'constant_timeout',
]