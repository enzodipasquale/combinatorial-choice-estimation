"""
Row generation solver for modular bundle choice estimation (v2).
This module will be used by BundleChoice to estimate parameters using row generation.
Future solvers can be added to this folder as well.
"""
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


class RowGenerationSolver(BaseEstimationSolver):
    """
    Implements the row generation algorithm for parameter estimation in modular bundle choice models.

    This solver is designed for use with the v2 BundleChoice API and its managers. It supports distributed computation via MPI and Gurobi for solving the master problem.
    """
    def __init__(
                self, 
                comm_manager, 
                dimensions_cfg, 
                rowgen_cfg, 
                data_manager, 
                feature_manager, 
                subproblem_manager):
        """
        Initialize the RowGenerationSolver.

        Args:
            comm_manager: Communication manager for MPI operations
            dimensions_cfg: DimensionsConfig instance
            rowgen_cfg: RowGenerationConfig instance
            data_manager: DataManager instance
            feature_manager: FeatureManager instance
            subproblem_manager: SubproblemManager instance
        """
        super().__init__(
            comm_manager=comm_manager,
            dimensions_cfg=dimensions_cfg,
            data_manager=data_manager,
            feature_manager=feature_manager,
            subproblem_manager=subproblem_manager
        )
        
        self.rowgen_cfg = rowgen_cfg
        self.master_model = None
        self.master_variables = None
        self.theta_val = None
        self.theta_hat = None
        self.slack_counter = None

    def _setup_gurobi_model_params(self):
        """
        Create and set up Gurobi model with parameters from configuration.
        
        Returns:
            Gurobi model instance with configured parameters.
        """    
        with suppress_output():
            model = gp.Model()
            Method = self.rowgen_cfg.master_settings.get("Method", 0)
            model.setParam('Method', Method)
            # model.setParam('Threads', self.master_threads)
            LPWarmStart = self.rowgen_cfg.master_settings.get("LPWarmStart", 2)
            model.setParam('LPWarmStart', LPWarmStart)
            OutputFlag = self.rowgen_cfg.master_settings.get("OutputFlag", 0)
            model.setParam('OutputFlag', OutputFlag)
            return model

    def _initialize_master_problem(self):
        """
        Create and configure the master problem (Gurobi model).
        
        Returns:
            tuple: (master_model, master_variables, theta_val)
        """

        if self.is_root():
            self.master_model = self._setup_gurobi_model_params()          
            # obs_features = agents_obs_features.sum(0)
            theta = self.master_model.addMVar(self.num_features, obj= - self.obs_features, ub=100, name='parameter')
            u = self.master_model.addMVar(self.num_simuls * self.num_agents, obj=1, name='utility')
            
            # self.master_model.addConstrs((
            #     u[si] >=
            #     self.errors[si] @ obs_bundle[si % self.num_agents] +
            #     agents_obs_features[si % self.num_agents, :] @ theta
            #     for si in range(self.num_simuls * self.num_agents)
            # ))
            self.master_model.optimize()
            logger.info("Master Initialized. Parameter: %s", theta.X)
            
            self.master_variables = (theta, u)
            self.theta_val = theta.X
        self.theta_val = self.comm_manager.broadcast_from_root(self.theta_val, root=0)
    


    def _master_iteration(self, local_pricing_results):
        """
        Perform one iteration of the master problem in the row generation algorithm.

        Args:
            pricing_results (list of np.ndarray): List of bundle selection matrices from pricing subproblems.

        Returns:
            bool: Whether the stopping criterion is met.
        """
        x_sim = self.feature_manager.compute_gathered_features(local_pricing_results)
        errors_sim = self.feature_manager.compute_gathered_errors(local_pricing_results)
        stop = False
        if self.is_root():
            theta, u = self.master_variables
            with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                u_sim = x_sim @ theta.X + errors_sim
            u_master = u.X

            violations = np.where((u_master - u_sim) / (np.abs(u_master) + 1e-8) > 1e-6)[0]
            if len(violations) > 0:
                logger.warning("Possible failure of demand oracle: %d", len(violations))
                logger.warning("u_sim", u_sim[violations])
                logger.warning("u_master", u_master[violations])
             
            logger.info("Parameter: %s", np.round(self.theta_val, 2))
            logger.info(f"ObjVal: {self.master_model.ObjVal}")
            max_reduced_cost = np.max(u_sim - u_master)
            logger.info("Reduced cost: %s", max_reduced_cost)
            if max_reduced_cost < self.rowgen_cfg.tol_certificate:
                stop = True
            rows_to_add = np.where(u_sim > u_master * (1 + self.rowgen_cfg.tol_row_generation) + self.rowgen_cfg.tol_certificate)[0]
            logger.info("New constraints: %d", len(rows_to_add))
            self.master_model.addConstr(u[rows_to_add]  >= errors_sim[rows_to_add] + x_sim[rows_to_add] @ theta)
            num_removed = self._enforce_slack_counter()
            logger.info("Removed constraints: %d", num_removed)
            logger.info("Number of constraints: %d", self.master_model.NumConstrs)
            self.master_model.optimize()
            theta_val = theta.X
            self.rowgen_cfg.tol_row_generation *= self.rowgen_cfg.row_generation_decay
        else:
            theta_val = None
        self.theta_val = self.comm_manager.broadcast_from_root(theta_val, root=0)
        stop = self.comm_manager.broadcast_from_root(stop, root=0)
        return stop

    def solve(self):
        """
        Run the row generation algorithm to estimate model parameters.

        Returns:
            tuple: (theta_val)
                - theta_val (np.ndarray): Estimated parameter vector.
        """
        tic = datetime.now()
        self.subproblem_manager.initialize_local()
        self._initialize_master_problem()        
        self.slack_counter = {}
        logger.info("Starting row generation loop.")
        iteration = 0
        while iteration < self.rowgen_cfg.max_iters:
            logger.info(f"ITERATION {iteration + 1}")
            local_pricing_results = self.subproblem_manager.solve_local(self.theta_val)
            stop = self._master_iteration(local_pricing_results) 
            if stop and iteration >= self.rowgen_cfg.min_iters:
                if self.is_root():
                    elapsed = (datetime.now() - tic).total_seconds()
                    logger.info("Row generation ended after %d iterations in %.2f seconds.", iteration + 1, elapsed)
                    logger.info(f"ObjVal: {self.master_model.ObjVal}")
                break
            iteration += 1
        self.theta_hat = self.theta_val
        return self.theta_hat




    def _enforce_slack_counter(self):
        """
        Update the slack counter for master problem constraints and remove those that have been slack for too long.

        Returns:
            int: Number of constraints removed.
        """
        to_remove = []
        for constr in self.master_model.getConstrs():
            if constr.Slack > 0:
                # Only add to counter when constraint is actually slack
                if constr not in self.slack_counter:
                    self.slack_counter[constr] = 0
                self.slack_counter[constr] += 1
                if self.slack_counter[constr] >= self.rowgen_cfg.max_slack_counter:
                    to_remove.append(constr)
        
        # Remove all constraints that exceeded the slack counter limit
        for constr in to_remove:
            self.master_model.remove(constr)
            self.slack_counter.pop(constr, None)
        
        return len(to_remove)
