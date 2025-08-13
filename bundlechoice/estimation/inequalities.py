import numpy as np
from datetime import datetime
from typing import Tuple, List, Optional, Any, Dict
import logging
import gurobipy as gp
from gurobipy import GRB
from bundlechoice.utils import get_logger, suppress_output
from .base import HasDimensions, HasData
logger = get_logger(__name__)


# Ensure root logger is configured for INFO level output
logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s][%(process)d][%(name)s] %(message)s')


class InequalitiesSolver(HasDimensions, HasData):
    """
    Implements the inequalities method for parameter estimation in modular bundle choice models.

    This solver is designed for use with the v2 BundleChoice API and its managers. It supports distributed computation via MPI and Gurobi for solving the master problem.
    """
    def __init__(
                self,
                dimensions_cfg,  
                data_manager, 
                feature_manager 
                ):
        """
        Initialize the InequalitiesSolver.
        """
        self.dimensions_cfg = dimensions_cfg
        self.data_manager = data_manager
        self.feature_manager = feature_manager
        
        # Initialize common attributes
        self.obs_features = self.get_obs_features()

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
        u = model.addMVar(self.num_simuls * self.num_agents, obj=1, name='utility')
        features_alt, errors_alt = self.compute_features_alt_AddDrop()
        model.addConstr(u[:, None] >= gp.quicksum(features_alt[:,:,k] * theta[k] for k in range(self.num_features)) + errors_alt)
        
        model.optimize()
        
        elapsed = (datetime.now() - tic).total_seconds()
        logger.info(f"Elapsed time: {elapsed:.2f} seconds")
        
        return theta.X
        
    def compute_features_alt_AddDrop(self):
        """
        Compute alternative features for add/drop operations.
        
        Returns:
            tuple: (features_alt, errors_alt) arrays
        """
        features_alt = np.zeros((self.num_agents * self.num_simuls, self.num_items, self.num_features))
        errors_alt = np.zeros((self.num_agents * self.num_simuls, self.num_items))
        
        for si in range(self.num_agents * self.num_simuls):
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
