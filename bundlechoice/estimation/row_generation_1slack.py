"""
Row generation solver with 1slack formulation for modular bundle choice estimation (v2).
This module implements a simplified row generation approach with a single scalar utility variable.
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


class RowGeneration1SlackSolver(BaseEstimationSolver):
    """
    Implements the row generation algorithm with 1slack formulation for parameter estimation in modular bundle choice models.

    This solver uses a single scalar utility variable instead of one per simulation/agent pair,
    with constraints of the form: u >= sum(si) errors_si + sum(si) sum(k) x_si,k * theta_k

    This solver is designed for use with the v2 BundleChoice API and its managers. It supports distributed computation via MPI and Gurobi for solving the master problem.
    """
    def __init__(
                self, 
                comm_manager, 
                dimensions_cfg, 
                row_generation_cfg, 
                data_manager, 
                feature_manager, 
                subproblem_manager):
        """
        Initialize the RowGeneration1SlackSolver.

        Args:
            comm_manager: Communication manager for MPI operations
            dimensions_cfg: DimensionsConfig instance
            row_generation_cfg: RowGenerationConfig instance
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
        
        self.row_generation_cfg = row_generation_cfg
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
            Method = self.row_generation_cfg.gurobi_settings.get("Method", 0)
            model.setParam('Method', Method)
            Threads = self.row_generation_cfg.gurobi_settings.get("Threads")
            if Threads is not None:
                model.setParam('Threads', Threads)
            LPWarmStart = self.row_generation_cfg.gurobi_settings.get("LPWarmStart", 2)
            model.setParam('LPWarmStart', LPWarmStart)
            OutputFlag = self.row_generation_cfg.gurobi_settings.get("OutputFlag", 0)
            model.setParam('OutputFlag', OutputFlag)
        return model

    def _initialize_master_problem(self):
        """
        Create and configure the master problem (Gurobi model) with 1slack formulation.
        
        Returns:
            tuple: (master_model, master_variables, theta_val)
        """
        obs_features = self.get_obs_features()
        if self.is_root():
            self.master_model = self._setup_gurobi_model_params()    
            theta = self.master_model.addMVar(self.num_features, obj=-obs_features, ub=self.row_generation_cfg.theta_ubs, name='parameter')
            if self.row_generation_cfg.theta_lbs is not None:
                theta.lb = self.row_generation_cfg.theta_lbs
            u_bar = self.master_model.addVar(obj=1, name='utility')  # Single scalar utility variable
            
            self.master_model.optimize()
            logger.info("Master Initialized (1slack formulation)")
            self.master_variables = (theta, u_bar)
            self.theta_val = theta.X
            self.log_parameter()

        self.theta_val = self.comm_manager.broadcast_from_root(self.theta_val, root=0)

    def _master_iteration(self, optimal_bundles):
        """
        Perform one iteration of the master problem in the row generation algorithm with 1slack formulation.

        Args:
            pricing_results (list of np.ndarray): List of bundle selection matrices from pricing subproblems.

        Returns:
            bool: Whether the stopping criterion is met.
        """
        x_sim = self.feature_manager.compute_gathered_features(optimal_bundles)
        errors_sim = self.feature_manager.compute_gathered_errors(optimal_bundles)
        stop = False
        
        if self.is_root():
            theta, u_bar = self.master_variables
            u_sim = (x_sim @ theta.X).sum() + errors_sim.sum()
            u_master = u_bar.X  # Single scalar value


            self.log_parameter()
            logger.info(f"ObjVal: {self.master_model.ObjVal}")
            reduced_cost = u_sim - u_master
            logger.info("Reduced cost: %s", reduced_cost)
            
            if reduced_cost < self.row_generation_cfg.tolerance_optimality:
                stop = True
            else:          
                # Only add constraint if there's a violation (like standard formulation)
                if u_sim > u_master * (1 + self.row_generation_cfg.tol_row_generation) + self.row_generation_cfg.tolerance_optimality:
                    agents_utilities = (x_sim @ theta).sum() + errors_sim.sum()
                    self.master_model.addConstr(u_bar >= agents_utilities)
                    self._enforce_slack_counter()
                    logger.info("Number of constraints: %d", self.master_model.NumConstrs)
                    self.master_model.optimize()
                    self.row_generation_cfg.tol_row_generation *= self.row_generation_cfg.row_generation_decay
            
            # Get theta values for broadcasting
            theta_val = theta.X
        else:
            stop = False
            theta_val = None  # Will be overwritten by broadcast
            
        self.theta_val = self.comm_manager.broadcast_from_root(theta_val, root=0)
        stop = self.comm_manager.broadcast_from_root(stop, root=0)
        return stop

    def solve(self):
        """
        Run the row generation algorithm with 1slack formulation to estimate model parameters.

        Returns:
            tuple: (theta_val)
                - theta_val (np.ndarray): Estimated parameter vector.
        """
        logger.info("=== ROW GENERATION 1SLACK ===")
        tic = datetime.now()
        self.subproblem_manager.initialize_local()
        self._initialize_master_problem()        
        self.slack_counter = {}
        logger.info("Starting row generation loop (1slack formulation).")
        iteration = 0
        pricing_times, master_times = [], []
        
        while iteration < self.row_generation_cfg.max_iters:
            logger.info(f"ITERATION {iteration + 1}")
            t1 = datetime.now()
            optimal_bundles = self.subproblem_manager.solve_local(self.theta_val)
            pricing_times.append((datetime.now() - t1).total_seconds())
            t2 = datetime.now()
            stop = self._master_iteration(optimal_bundles) 
            master_times.append((datetime.now() - t2).total_seconds())
            
            if stop and iteration >= self.row_generation_cfg.min_iters:
                if self.is_root():
                    elapsed = (datetime.now() - tic).total_seconds()
                    logger.info("Row generation ended after %d iterations in %.2f seconds.", iteration + 1, elapsed)
                    logger.info(f"ObjVal: {self.master_model.ObjVal}")
                    logger.info(f"Avg pricing time: {np.mean(pricing_times):.3f}s, Avg master time: {np.mean(master_times):.3f}s")
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
        if self.row_generation_cfg.max_slack_counter < float('inf'):
            to_remove = []
            for constr in self.master_model.getConstrs():
                if constr.Slack < -1e-6:
                    # Only add to counter when constraint is actually slack
                    if constr not in self.slack_counter:
                        self.slack_counter[constr] = 0
                    self.slack_counter[constr] += 1
                    if self.slack_counter[constr] >= self.row_generation_cfg.max_slack_counter:
                        to_remove.append(constr)
                if constr.Pi > 1e-6:
                    self.slack_counter.pop(constr, None)
            # Remove all constraints that exceeded the slack counter limit
            for constr in to_remove:
                self.master_model.remove(constr)
                self.slack_counter.pop(constr, None)
            num_removed = len(to_remove)
            logger.info("Removed constraints: %d", num_removed)
            return num_removed
        else:
            return 0

    def log_parameter(self):
        feature_ids = self.row_generation_cfg.parameters_to_log
        precision = 3
        if feature_ids is not None:
            logger.info("Parameters: %s", np.round(self.theta_val[feature_ids], precision))
        else:
            logger.info("Parameters: %s", np.round(self.theta_val, precision))
