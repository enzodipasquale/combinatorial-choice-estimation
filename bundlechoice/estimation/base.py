"""
Base estimation solver for modular bundle choice estimation (v2).
This module provides a base class with common functionality for different estimation algorithms.
"""
import numpy as np
from typing import Optional, Any, Tuple
from bundlechoice.base import HasDimensions, HasData, HasComm
from bundlechoice.utils import get_logger

logger = get_logger(__name__)


class BaseEstimationSolver(HasDimensions, HasData, HasComm):
    """
    Base class for estimation solvers in modular bundle choice models.
    
    This class provides common functionality and helper methods that can be shared
    across different estimation algorithms (row generation, ellipsoid, etc.).
    
    Attributes:
        comm: MPI communicator
        rank: MPI rank of current process
        dimensions_cfg: DimensionsConfig instance
        data_manager: DataManager instance
        feature_manager: FeatureManager instance
        subproblem_manager: SubproblemManager instance
        errors: Error terms from input data
        obs_features: Observed features (computed on rank 0)
    """
    
    def __init__(
        self,
        comm_manager,
        dimensions_cfg,
        data_manager,
        feature_manager,
        subproblem_manager
    ):
        """
        Initialize the BaseEstimationSolver.

        Args:
            comm_manager: Communication manager for MPI operations
            dimensions_cfg: DimensionsConfig instance
            data_manager: DataManager instance
            feature_manager: FeatureManager instance
            subproblem_manager: SubproblemManager instance
        """
        self.comm_manager = comm_manager
        self.dimensions_cfg = dimensions_cfg
        self.data_manager = data_manager
        self.feature_manager = feature_manager
        self.subproblem_manager = subproblem_manager

        # Initialize common attributes
        self.agents_obs_features = self.get_agents_obs_features()
        self.obs_features = self.agents_obs_features.sum(0) if self.agents_obs_features is not None else None

    def get_obs_features(self) -> Optional[np.ndarray]:
        """
        Compute observed features from local data.
        
        This method computes the average observed features across all simulations.
        Only rank 0 returns the result, other ranks return None.
        
        Returns:
            np.ndarray or None: Average observed features (rank 0) or None (other ranks)
        """
        local_bundles = self.local_data.get("obs_bundles")
        agents_obs_features = self.feature_manager.compute_gathered_features(local_bundles)
        if self.is_root():
            obs_features = agents_obs_features.sum(0) 
            return obs_features
        else:
            return None

    def get_agents_obs_features(self) -> Optional[np.ndarray]:
        """
        Compute observed features from local data.
        
        This method computes the average observed features across all simulations.
        Only rank 0 returns the result, other ranks return None.
        """
        local_bundles = self.local_data.get("obs_bundles")
        agents_obs_features = self.feature_manager.compute_gathered_features(local_bundles)
        if self.is_root():
            return agents_obs_features
        else:
            return None

    def compute_obj_and_gradient(self, theta: np.ndarray) -> Optional[Tuple[float, np.ndarray]]:
        """
        Compute both objective function value and gradient efficiently.
        
        This method computes both values in one call to avoid duplicate
        expensive computations like subproblem solving and feature computation.
        
        Args:
            theta: Parameter vector
            
        Returns:
            Tuple[float, np.ndarray] or None: (objective_value, gradient) (rank 0) or None (other ranks)
        """
        # Solve subproblem and compute features (expensive operation)
        B_local = self.subproblem_manager.solve_local(theta)
        agents_features = self.feature_manager.compute_gathered_features(B_local)
        utilities = self.feature_manager.compute_gathered_utilities(B_local, theta)
        
        if self.is_root():
            # Compute utilities for objective
            obj_value = utilities.sum() - (self.obs_features @ theta).sum()
            
            # Compute (normalized) gradient
            gradient = (agents_features.sum(0) - self.obs_features) / self.num_agents
            
            return obj_value, gradient
        else:
            return None, None

    def objective(self, theta: np.ndarray) -> Optional[float]:
        """
        Compute the objective function value for given parameters.
        
        This method computes the difference between simulated and observed features
        weighted by the parameters, plus error terms.
        
        Args:
            theta: Parameter vector
            
        Returns:
            float or None: Objective function value (rank 0) or None (other ranks)
        """
        B_local = self.subproblem_manager.solve_local(theta)
        utilities = self.feature_manager.compute_gathered_utilities(B_local, theta)
        if self.is_root():
            return utilities.sum() - (self.obs_features @ theta).sum()
        else:
            return None
    
    def obj_gradient(self, theta: np.ndarray) -> Optional[np.ndarray]:
        """
        Compute the gradient of the objective function.
        
        This method computes the gradient with respect to the parameters.
        Note: Assumes subproblems are already initialized.
        
        Args:
            theta: Parameter vector
            
        Returns:
            np.ndarray or None: Gradient vector (rank 0) or None (other ranks)
        """
        B_local = self.subproblem_manager.solve_local(theta)
        agents_features = self.feature_manager.compute_gathered_features(B_local)
        if self.is_root():
            return (agents_features.sum(0) - self.obs_features) / self.num_agents
        else:
            return None



    def solve(self) -> np.ndarray:
        """
        Main solve method to be implemented by subclasses.
        
        This method should implement the specific estimation algorithm.
        
        Returns:
            np.ndarray: Estimated parameter vector
        """
        raise NotImplementedError("Subclasses must implement the solve method") 