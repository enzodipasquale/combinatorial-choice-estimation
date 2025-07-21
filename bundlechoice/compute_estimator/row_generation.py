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
from bundlechoice.subproblems.subproblem_manager import SubproblemManager
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
     

    @property
    def error_si_j(self):
        return self.input_data["errors"].reshape(self.num_simuls * self.num_agents, self.num_items)

    def _initialize_master_problem(self):
        """
        Initialize the master Gurobi model for row generation (rank 0 only).

        Sets self.master_model and self.master_variables as instance attributes.

        Returns:
            tuple: (lambda_k.X, p_j.X or None)
                - lambda_k.X (np.ndarray): Initial parameter vector.
                - p_j.X (np.ndarray or None): Initial price vector if item fixed effects are enabled, else None.
        """
        if self.rank == 0:
            with suppress_output():
                self.master_model = gp.Model()
                self.master_model.setAttr('ModelSense', gp.GRB.MAXIMIZE)
                Method = self.rowgen_cfg.master_settings.get("Method", 0)
                self.master_model.setParam('Method', Method)
                # self.master_model.setParam('Threads', self.master_threads)
                LPWarmStart = self.rowgen_cfg.master_settings.get("LPWarmStart", 2)
                self.master_model.setParam('LPWarmStart', LPWarmStart)
                OutputFlag = self.rowgen_cfg.master_settings.get("OutputFlag", 0)
                self.master_model.setParam('OutputFlag', OutputFlag)
            
            # Get observed bundle features
            obs_bundle = self.input_data["obs_bundle"]
            x_hat_i_k = self.feature_manager.get_all_agent_features(obs_bundle)
            x_hat_k = x_hat_i_k.sum(0)

            lambda_k = self.master_model.addMVar(self.num_features, obj=self.num_simuls * x_hat_k, ub=1e4, name='parameter')
            u_si = self.master_model.addMVar(self.num_simuls * self.num_agents, obj=-1, name='utility')

            # Add item fixed effects if enabled
            if self.rowgen_cfg.item_fixed_effects:
                p_j = self.master_model.addMVar(self.num_items, obj=-self.num_simuls, name='price')
            else:
                p_j = None

            self.master_model.update()

            # Revert to original, robust constraint addition
            self.master_model.addConstrs((
                u_si[si] >=
                self.error_si_j[si] @ obs_bundle[si % self.num_agents] +
                x_hat_i_k[si % self.num_agents, :] @ lambda_k
                for si in range(self.num_simuls * self.num_agents)
            ))
            self.master_model.optimize()
            logger.info("Master Initialized. Parameter: %s", lambda_k.X)

            self.master_variables = (lambda_k, u_si, p_j)
            return lambda_k.X, p_j.X if p_j is not None else None
        else:
            self.master_model = None
            self.master_variables = None
            return None, None

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

    def _master_iteration(self, pricing_results, slack_counter=None):
        """
        Perform one iteration of the master problem in the row generation algorithm.

        Args:
            pricing_results (list of np.ndarray): List of bundle selection matrices from pricing subproblems.
            slack_counter (dict, optional): Slack counter for constraints.

        Returns:
            tuple: (stop, lambda_k.X, p_j.X or None)
                - stop (bool): Whether the stopping criterion is met.
                - lambda_k.X (np.ndarray): Current parameter vector.
                - p_j.X (np.ndarray or None): Current price vector if item fixed effects are enabled, else None.
        """
        if self.rank == 0:
            tic = datetime.now()
            B_star_si_j = np.concatenate(pricing_results).astype(bool)
            lambda_k, u_si, p_j = self.master_variables
            eps_si_star = np.where(B_star_si_j, self.error_si_j, 0).sum(1)
            x_star_si_k = self.feature_manager.get_all_simulated_agent_features(B_star_si_j)
            u_si_star = x_star_si_k @ lambda_k.X + eps_si_star

            if p_j is not None:
                u_si_star -= B_star_si_j @ p_j.X
            u_si_master = u_si.X
            max_reduced_cost = np.max(u_si_star - u_si_master)
            logger.info("Reduced cost: %s", max_reduced_cost)
            if max_reduced_cost < self.rowgen_cfg.tol_certificate:
                return True, lambda_k.X, p_j.X if p_j is not None else None
            rows_id = np.where(u_si_star > u_si_master * (1 + self.rowgen_cfg.tol_row_generation))[0]
            logger.info("New constraints: %d", len(rows_id))

            # self.master_model.addConstrs((
            #     u_si[si]  >= eps_si_star[si] + x_star_si_k[si] @ lambda_k
            #     for si in rows_id
            # ))
            self.master_model.addConstr(u_si[rows_id]  >= eps_si_star[rows_id] + x_star_si_k[rows_id] @ lambda_k)

            slack_counter, num_constrs_removed = self._update_slack_counter(self.master_model, slack_counter)
            self.master_model.optimize()
            self.rowgen_cfg.tol_row_generation *= self.rowgen_cfg.row_generation_decay
            return False, lambda_k.X, p_j.X if p_j is not None else None
        else:
            # On non-root ranks, self.master_model and self.master_variables are None
            return None, None, None

    def solve(self):
        """
        Run the row generation algorithm to estimate model parameters.

        Returns:
            tuple: (lambda_k_iter, p_j_iter)
                - lambda_k_iter (np.ndarray): Estimated parameter vector.
                - p_j_iter (np.ndarray or None): Estimated price vector if item fixed effects are enabled, else None.
        """
        tic = datetime.now()
        
        # Initialize local subproblems
        self.subproblem_manager.initialize_local()
        
        # Initialize master problem
        lambda_k_iter, p_j_iter = self._initialize_master_problem()
        slack_counter = {}
        lambda_k_iter, p_j_iter = self.comm.bcast((lambda_k_iter, p_j_iter), root=0)
        
        logger.info("Starting row generation loop.")
        # Main row generation loop
        for iteration in range(int(self.rowgen_cfg.max_iters)):
            logger.info(f"ITERATION {iteration + 1}")
            # Solve pricing problems
            local_pricing_results = self.subproblem_manager.solve_local(lambda_k_iter)
            pricing_results = self.comm.gather(local_pricing_results, root=0)
            
            # Update master problem
            stop, lambda_k_iter, p_j_iter = self._master_iteration(pricing_results, slack_counter)
            stop, lambda_k_iter, p_j_iter = self.comm.bcast((stop, lambda_k_iter, p_j_iter), root=0)
            
            if stop and iteration >= self.rowgen_cfg.min_iters:
                elapsed = (datetime.now() - tic).total_seconds()
                logger.info("Row generation completed after %d iterations in %.2f seconds.", iteration + 1, elapsed)
                break
                
        return lambda_k_iter, p_j_iter 