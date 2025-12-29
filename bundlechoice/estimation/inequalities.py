import numpy as np
from datetime import datetime
from typing import Tuple, List, Optional, Any, Dict
from numpy.typing import NDArray
import logging
import gurobipy as gp
from gurobipy import GRB
from bundlechoice.utils import get_logger, suppress_output
from .base import BaseEstimationManager
logger = get_logger(__name__)


class InequalitiesManager(BaseEstimationManager):
    """
    Implements the inequalities method for parameter estimation in modular bundle choice models.

    This manager is designed for use with the v2 BundleChoice API and its managers. It supports distributed computation via MPI and Gurobi for solving the master problem.
    """
    def __init__(
                self,
                comm_manager: Any,
                dimensions_cfg: Any,
                data_manager: Any,
                feature_manager: Any,
                subproblem_manager: Optional[Any] = None
                ) -> None:
        """
        Initialize the InequalitiesManager.
        Note: subproblem_manager is optional (not needed for inequalities method).
        """
        super().__init__(
            comm_manager=comm_manager,
            dimensions_cfg=dimensions_cfg,
            data_manager=data_manager,
            feature_manager=feature_manager,
            subproblem_manager=subproblem_manager
        )

    def solve(self):
        """
        Solve the inequalities estimation problem.
        
        Returns:
            np.ndarray: Estimated parameter vector theta
        """
        logger.info("=== INEQUALITIES SOLVER ===")
        tic = datetime.now()
        
        model = gp.Model()
        model.setParam('OutputFlag', 0)
        theta = model.addMVar(self.num_features, obj=-self.obs_features, ub=100, name='parameter')
        u = model.addMVar(self.num_simulations * self.num_agents, obj=1, name='utility')
        features_alt, errors_alt = self.compute_features_alt_AddDrop()
        model.addConstr(u[:, None] >= gp.quicksum(features_alt[:,:,k] * theta[k] for k in range(self.num_features)) + errors_alt)
        
        model.optimize()
        
        elapsed = (datetime.now() - tic).total_seconds()
        logger.info(f"Elapsed time: {elapsed:.2f} seconds")
        
        return theta.X
        
    def compute_features_alt_AddDrop(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Compute alternative features for add/drop operations. Returns (features_alt, errors_alt)."""
        features_alt = np.zeros((self.num_agents * self.num_simulations, self.num_items, self.num_features))
        errors_alt = np.zeros((self.num_agents * self.num_simulations, self.num_items))
        
        for si in range(self.num_agents * self.num_simulations):
            i = si % self.num_agents
            s = si // self.num_agents
            bundle = self.input_data["obs_bundle"][i]
            
            for j in range(self.num_items):
                if bundle[j] == 1:
                    bundle_alt = bundle.copy()
                    bundle_alt[j] = 0
                    
                else:
                    bundle_alt = bundle.copy()
                    bundle_alt[j] = 1
                features_alt[si, j] = self.feature_manager.features_oracle(i, bundle_alt)
                errors_alt[si, j] = self.input_data["errors"][s, i, bundle_alt].sum()
                    
        return features_alt, errors_alt 

    def get_obs_features(self) -> Optional[np.ndarray]:
        """
        Compute observed features from local data.
        
        This method computes the average observed features across all simulations.
        Only rank 0 returns the result, other ranks return None.
        
        Returns:
            np.ndarray or None: Average observed features (rank 0) or None (other ranks)
        """
        obs_bundles = self.input_data.get("obs_bundle")
        agents_obs_features = np.array([self.feature_manager.features_oracle(i, obs_bundles[i]) for i in range(self.num_agents)])
        return agents_obs_features.sum(0) 
