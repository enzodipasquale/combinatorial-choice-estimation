"""
Base estimation solver for modular bundle choice estimation (v2).
This module provides a base class with common functionality for different estimation algorithms.
"""
import numpy as np
from typing import Optional, Any
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
        self.errors = self.input_data["errors"].reshape(-1, self.num_items) if self.input_data is not None else None
        self.obs_features = self.get_obs_features()

    def get_obs_features(self) -> Optional[np.ndarray]:
        """
        Compute observed features from local data.
        
        This method computes the average observed features across all simulations.
        Only rank 0 returns the result, other ranks return None.
        
        Returns:
            np.ndarray or None: Average observed features (rank 0) or None (other ranks)
        """
        local_bundles = self.local_data.get("obs_bundles")
        agents_obs_features = self.feature_manager.get_all_distributed(local_bundles)
        if self.is_root():
            obs_features = agents_obs_features.sum(0) / self.num_simuls
            return obs_features
        else:
            return None

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
        B_local_sim = self.subproblem_manager.solve_local(theta)
        features_sim = self.feature_manager.get_all_distributed(B_local_sim)
        B_sim = self.comm_manager.concatenate_at_root(B_local_sim, root=0)
        if self.is_root():
            errors_sim = (self.errors * B_sim).sum(1)
            with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                return (features_sim @ theta + errors_sim).sum() - (self.obs_features @ theta).sum()
        else:
            return None
    
    def obj_gradient(self, theta: np.ndarray) -> Optional[np.ndarray]:
        """
        Compute the gradient of the objective function.
        
        This method computes the gradient with respect to the parameters.
        
        Args:
            theta: Parameter vector
            
        Returns:
            np.ndarray or None: Gradient vector (rank 0) or None (other ranks)
        """
        B_local_sim = self.subproblem_manager.solve_local(theta)
        features_sim = self.feature_manager.get_all_distributed(B_local_sim)
        B_sim = self.comm_manager.concatenate_at_root(B_local_sim, root=0)
        if self.is_root():
            with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                return features_sim.sum(0) - self.obs_features 
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