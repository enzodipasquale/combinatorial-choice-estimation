"""
Row generation solver for modular bundle choice estimation (v2).
This module will be used by BundleChoice to estimate parameters using row generation.
Future solvers can be added to this folder as well.
"""
import numpy as np
from datetime import datetime
from typing import Tuple, List, Optional, Any, Dict
import logging
from ..utils import suppress_output
import gurobipy as gp
from gurobipy import GRB
from bundlechoice.utils import get_logger, suppress_output
from bundlechoice.base import HasDimensions, HasData
logger = get_logger(__name__)

# Ensure root logger is configured for INFO level output
logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s][%(process)d][%(name)s] %(message)s')


class RowGenerationSolver(HasDimensions, HasData):
    """
    Implements the row generation algorithm for parameter estimation in modular bundle choice models.

    This solver is designed for use with the v2 BundleChoice API and its managers. It supports distributed computation via MPI and Gurobi for solving the master problem.
    """
    def __init__(
                self, 
                comm, 
                dimensions_cfg, 
                rowgen_cfg, 
                data_manager, 
                feature_manager, 
                subproblem_manager):
        """
        Initialize the RowGenerationSolver.

        Args:
            comm: MPI communicator
            dimensions_cfg: DimensionsConfig instance
            rowgen_cfg: RowGenConfig instance
            data_manager: DataManager instance
            feature_manager: FeatureManager instance
            subproblem_manager: SubproblemManager instance
        """
        self.comm = comm
        self.rank = comm.Get_rank()
        self.dimensions_cfg = dimensions_cfg
        self.rowgen_cfg = rowgen_cfg
        self.data_manager = data_manager
        self.feature_manager = feature_manager
        self.subproblem_manager = subproblem_manager

        self.master_model = None
        self.master_variables = None
        self.parameter_iter = None


        self.errors = self.input_data["errors"].reshape(self.num_simuls * self.num_agents, self.num_items) if self.input_data is not None else None

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
            tuple: (master_model, master_variables, parameter_iter)
        """
        if self.rank == 0:
            self.master_model = self._setup_gurobi_model_params()
            # Get observed bundle features
            obs_bundle = self.input_data["obs_bundle"]
            agents_obs_features = self.feature_manager.get_all_agent_features(obs_bundle)
            obs_features = agents_obs_features.sum(0)
            theta = self.master_model.addMVar(self.num_features, obj= - self.num_simuls * obs_features, ub=1e4, name='parameter')
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
            self.parameter_iter = theta.X
        self.parameter_iter = self.comm.bcast(self.parameter_iter, root=0)
        


    def _master_iteration(self, pricing_results):
        """
        Perform one iteration of the master problem in the row generation algorithm.

        Args:
            pricing_results (list of np.ndarray): List of bundle selection matrices from pricing subproblems.

        Returns:
            bool: Whether the stopping criterion is met.
        """
        stop = False
        if self.rank == 0:
            tic = datetime.now()
            B_sim = np.concatenate(pricing_results).astype(bool)
            theta, u = self.master_variables
            error_sim = np.where(B_sim, self.errors, 0).sum(1)
            x_sim = self.feature_manager.get_all_simulated_agent_features(B_sim)
            with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                u_sim = x_sim @ theta.X + error_sim
            u_master = u.X

            violations = np.where(u_master - u_sim > 1e-6)[0]
            if len(violations) > 0:
                print("X"* 100)
                print("u_sim", u_sim[violations])
                print("u_master", u_master[violations])
                print("X"* 100)
                stop = True

            logger.info("Parameter: %s", np.round(self.parameter_iter, 2))
            max_reduced_cost = np.max(u_sim - u_master)
            logger.info("Reduced cost: %s", max_reduced_cost)
            if max_reduced_cost < self.rowgen_cfg.tol_certificate:
                stop = True
            rows_to_add = np.where(u_sim > u_master * (1 + self.rowgen_cfg.tol_row_generation) + self.rowgen_cfg.tol_certificate)[0]
            logger.info("New constraints: %d", len(rows_to_add))
            self.master_model.addConstr(u[rows_to_add]  >= error_sim[rows_to_add] + x_sim[rows_to_add] @ theta)
            # self.master_model.addConstr(u  >= error_sim + x_sim @ theta)
            self.master_model.optimize()
            parameter_iter = theta.X
            self.rowgen_cfg.tol_row_generation *= self.rowgen_cfg.row_generation_decay
        else:
            parameter_iter = None
        # self.comm.Barrier()
        parameter_iter = self.comm.bcast(parameter_iter, root=0)
        self.parameter_iter = parameter_iter
        stop = self.comm.bcast(stop, root=0)
        return stop

    def solve(self):
        """
        Run the row generation algorithm to estimate model parameters.

        Returns:
            tuple: (parameter_iter)
                - parameter_iter (np.ndarray): Estimated parameter vector.
        """
        tic = datetime.now()
        self.subproblem_manager.initialize_local()
        self._initialize_master_problem()        

        logger.info("Starting row generation loop.")
        for iteration in range(int(self.rowgen_cfg.max_iters)):
            logger.info(f"ITERATION {iteration + 1}")
            local_pricing_results = self.subproblem_manager.solve_local(self.parameter_iter)
            pricing_results = self.comm.gather(local_pricing_results, root=0)
            stop = self._master_iteration(pricing_results) 
            if stop and iteration >= self.rowgen_cfg.min_iters:
                if self.rank == 0:
                    elapsed = (datetime.now() - tic).total_seconds()
                    logger.info("Row generation ended after %d iterations in %.2f seconds.", iteration + 1, elapsed)
                    logger.info(f"ObjVal: {self.master_model.ObjVal}")
                break
        return self.parameter_iter


    # Helper functions
    def ObjVal(self, theta):
        B_local_obs = self.local_data.get("obs_bundles")
        features_obs = self.feature_manager.get_all_simulated_agent_features_MPI(B_local_obs)
    
        B_local_sim = self.subproblem_manager.solve_local(theta)
        features_sim = self.feature_manager.get_all_simulated_agent_features_MPI(B_local_sim)
        B_sim = self.comm.gather(B_local_sim, root=0)
        if self.rank == 0:
            B_sim = np.concatenate(B_sim)
            errors = self.data_manager.input_data.get("errors").reshape(-1, self.num_items)
            errors_sim = (errors * B_sim).sum(1)
            with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                return  (features_sim @ theta+ errors_sim).sum() - (features_obs @ theta).sum()
        else:
            return None

    def _update_slack_counter(self, master_model, slack_counter):
        """
        Update the slack counter for master problem constraints and remove those that have been slack for too long.

        Args:
            master_model (gurobipy.Model): The master Gurobi model.
            slack_counter (dict): Dictionary mapping constraint names to their slack count.

        Returns:
            tuple: (updated slack_counter, int)
                - slack_counter (dict): Updated slack counter.
                - int: Number of constraints removed.
        """
        to_remove = []
        for constr in master_model.getConstrs():
            constr_name = constr.ConstrName
            if constr_name not in slack_counter:
                slack_counter[constr_name] = 0
            if constr.Slack < 0:
                slack_counter[constr_name] += 1
            else:
                slack_counter[constr_name] = 0
            if slack_counter[constr_name] >= self.rowgen_cfg.max_slack_counter:
                to_remove.append((constr_name, constr))
        for constr_name, constr in to_remove:
            master_model.remove(constr)
            slack_counter.pop(constr_name, None)
        return slack_counter, len(to_remove)
