import numpy as np
from datetime import datetime
from typing import Tuple, List, Optional, Any, Dict
import logging
import gurobipy as gp
from gurobipy import GRB
from bundlechoice.utils import get_logger, suppress_output
from .base import BaseEstimationSolver
logger = get_logger(__name__)


# Ensure root logger is configured for INFO level output
logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s][%(process)d][%(name)s] %(message)s')


class InequalitiesSolver(BaseEstimationSolver):
    """
    Implements the inequalities method for parameter estimation in modular bundle choice models.

    This solver is designed for use with the v2 BundleChoice API and its managers. It supports distributed computation via MPI and Gurobi for solving the master problem.
    """
    def __init__(
                self, 
                comm_manager, 
                dimensions_cfg,  
                data_manager, 
                feature_manager, 
                subproblem_manager):
        """
        Initialize the InequalitiesSolver.
        """
        super().__init__(comm_manager, dimensions_cfg, data_manager, feature_manager, subproblem_manager)
        
        # Initialize common attributes
        self.agents_obs_features = self.get_agents_obs_features()
        self.obs_features = self.agents_obs_features.sum(0) if self.agents_obs_features is not None else None

    def solve(self):
        """
        Solve the inequalities estimation problem.
        
        Returns:
            np.ndarray: Estimated parameter vector theta
        """
        logger.info("=== INEQUALITIES SOLVER ===")
        tic = datetime.now()
        
        model = gp.Model()
        theta = model.addMVar(self.num_features, obj=-self.obs_features, ub=100, name='parameter')
        u = model.addMVar(self.num_simuls * self.num_agents, obj=1, name='utility')
        features_alt, errors_alt = self.compute_features_alt_AddDrop()
        model.addConstrs((u[si] >= features_alt[si] @ theta + errors_alt[si] for si in range(self.num_agents * self.num_simuls)))
        
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
        features_alt = np.zeros((self.num_agents * self.num_simuls, self.num_features))
        errors_alt = np.zeros(self.num_agents * self.num_simuls)
        
        for si in range(self.num_agents * self.num_simuls):
            i = si % self.num_agents
            s = si // self.num_agents
            bundle = self.input_data["obs_bundle"][i]
            
            for j in range(self.num_items):
                if bundle[j] == 1:
                    bundle_alt = bundle.copy()
                    bundle_alt[j] = 0
                    features_alt[si] = self.feature_manager.features_oracle(i, bundle_alt)
                    errors_alt[si] = self.input_data["errors"][s, i, j]
                    break  # Only need one alternative per agent-simulation pair
                    
        return features_alt, errors_alt