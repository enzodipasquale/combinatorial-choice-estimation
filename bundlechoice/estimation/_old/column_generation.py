from __future__ import annotations
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING
import gurobipy as gp
import numpy as np
from gurobipy import GRB
from numpy.typing import NDArray
from bundlechoice.utils import get_logger, suppress_output, make_timing_stats
from .base import BaseEstimationManager
from .result import EstimationResult
logger = get_logger(__name__)

class ColumnGenerationManager(BaseEstimationManager):

    def __init__(self, comm_manager, dimensions_cfg, row_generation_cfg, data_manager, oracles_manager, subproblem_manager, theta_init=None):
        super().__init__(comm_manager=comm_manager, dimensions_cfg=dimensions_cfg, data_manager=data_manager, oracles_manager=oracles_manager, subproblem_manager=subproblem_manager)
        self.row_generation_cfg = row_generation_cfg
        self.theta_init = theta_init
        self.master_model: Optional[gp.Model] = None
        self.theta_val: Optional[NDArray[np.float64]] = None
        self.theta_hat: Optional[NDArray[np.float64]] = None
        self.timing_stats: Optional[Dict[str, float]] = None
        self.feature_constrs: List[gp.Constr] = []
        self.agent_constrs: Dict[int, gp.Constr] = {}
        self.column_vars: List[gp.Var] = []
        self.active_columns: List[Dict[str, Any]] = []
        self.has_columns: bool = False
        self._pricing_cache: Tuple[Optional[NDArray[np.bool_]], Optional[NDArray[np.float64]], Optional[NDArray[np.float64]], Optional[NDArray[np.float64]]] = (None, None, None, None)
        self.theta_upper = self._expand_bounds(getattr(row_generation_cfg, 'theta_ubs', None), np.inf)
        self.theta_lower = self._expand_bounds(getattr(row_generation_cfg, 'theta_lbs', None), -np.inf)
        self.alpha_vars: List[Optional[gp.Var]] = []
        self.beta_vars: List[Optional[gp.Var]] = []

    def _setup_colgen_model(self):
        model = self._setup_gurobi_model(self.row_generation_cfg.gurobi_settings)
        model.ModelSense = GRB.MAXIMIZE
        return model

    def _initialize_master_problem(self):
        obs_features = self.get_obs_features()
        if self.comm_manager._is_root():
            self.master_model = self._setup_colgen_model()
            self.feature_constrs = []
            self.agent_constrs = {}
            self.column_vars = []
            self.active_columns = []
            self.has_columns = False
            self.alpha_vars = []
            self.beta_vars = []
            for k in range(self.dimensions_cfg.num_features):
                expr = gp.LinExpr()
                alpha_var: Optional[gp.Var] = None
                upper = self.theta_upper[k]
                if np.isfinite(upper):
                    alpha_var = self.master_model.addVar(lb=0.0, obj=-float(upper), name=f'alpha[{k}]')
                    expr.addTerms(1.0, alpha_var)
                self.alpha_vars.append(alpha_var)
                beta_var: Optional[gp.Var] = None
                lower = self.theta_lower[k]
                if np.isfinite(lower):
                    beta_var = self.master_model.addVar(lb=0.0, obj=float(lower), name=f'beta[{k}]')
                    expr.addTerms(-1.0, beta_var)
                self.beta_vars.append(beta_var)
                rhs = float(obs_features[k] * max(1, self.dimensions_cfg.num_simulations))
                if np.isfinite(self.theta_lower[k]) and self.theta_lower[k] >= 0:
                    constr = self.master_model.addConstr(expr >= rhs, name=f'feature_match[{k}]')
                else:
                    constr = self.master_model.addConstr(expr == rhs, name=f'feature_match[{k}]')
                self.feature_constrs.append(constr)
            if self.theta_init is not None:
                self.theta_val = self.theta_init.astype(np.float64).copy()
            else:
                self.theta_val = np.zeros(self.dimensions_cfg.num_features, dtype=np.float64)
            self.master_model.update()
            self._add_initial_columns()
            logger.info('Column generation master initialised (dual).')
        else:
            self.theta_val = np.empty(self.dimensions_cfg.num_features, dtype=np.float64)
        self.theta_val = self.comm_manager.Bcast(self.theta_val, root=0)

    def _compute_theta_from_duals(self, dual_prices):
        has_nonneg_lower = np.isfinite(self.theta_lower) & (self.theta_lower >= 0)
        theta = np.where(has_nonneg_lower, np.maximum(self.theta_lower, -dual_prices), -dual_prices)
        return np.clip(theta, self.theta_lower, self.theta_upper)

    def _solve_pricing_problem(self, dual_prices, agent_penalties):
        modified_theta = self._compute_theta_from_duals(dual_prices)
        try:
            local_bundles = self.subproblem_manager.solve_local(modified_theta)
        except Exception as e:
            logger.error('Pricing problem failed with theta=%s, lower=%s, upper=%s: %s', modified_theta, self.theta_lower, self.theta_upper, e, exc_info=True)
            raise
        bundles_all = self.comm_manager.Gatherv_by_row(local_bundles, root=0)
        features_all = self.oracles_manager.compute_gathered_features(local_bundles)
        errors_all = self.oracles_manager.compute_gathered_errors(local_bundles)
        max_reduced_cost = 0.0
        if self.comm_manager._is_root() and features_all is not None and (errors_all is not None):
            penalties = agent_penalties[:len(errors_all)]
            scaled_errors = errors_all / max(1, self.dimensions_cfg.num_simulations)
            reduced_costs = scaled_errors - features_all @ dual_prices - penalties
            max_reduced_cost = float(np.max(reduced_costs))
            self._pricing_cache = (bundles_all, features_all, scaled_errors, reduced_costs)
        else:
            self._pricing_cache = (None, None, None, None)
        max_reduced_cost = self.comm_manager.bcast(max_reduced_cost, root=0)
        return (*self._pricing_cache, max_reduced_cost)

    def _ensure_agent_constraint(self, idx):
        if idx not in self.agent_constrs:
            constr = self.master_model.addConstr(gp.LinExpr() == 1.0, name=f'agent_balance[{idx}]')
            self.agent_constrs[idx] = constr
        return self.agent_constrs[idx]

    def _add_columns_to_master(self, bundles, features, errors, reduced_costs):
        if not self.comm_manager._is_root() or bundles is None or features is None or (errors is None) or (reduced_costs is None):
            return 0
        tolerance = self.row_generation_cfg.tolerance_optimality
        num_added = 0
        num_rows = len(errors)
        for idx in range(num_rows):
            rc = reduced_costs[idx]
            if rc <= tolerance:
                continue
            feature_coeffs = features[idx]
            agent_constr = self._ensure_agent_constraint(idx)
            column = gp.Column()
            for k, coeff in enumerate(feature_coeffs):
                if coeff != 0.0:
                    column.addTerms(float(coeff), self.feature_constrs[k])
            column.addTerms(1.0, agent_constr)
            var = self.master_model.addVar(obj=float(errors[idx] / max(1, self.dimensions_cfg.num_simulations)), lb=0.0, column=column, name=f'mu_{idx}_{len(self.column_vars)}')
            self.column_vars.append(var)
            self.active_columns.append({'bundle': bundles[idx].copy(), 'features': features[idx].copy(), 'errors': float(errors[idx]), 'agent_index': idx})
            num_added += 1
        if num_added > 0:
            self.master_model.update()
            self.has_columns = True
            logger.info('Added %d new columns (violated bundles).', num_added)
        else:
            logger.info('No violated bundles detected.')
        return num_added

    def _add_initial_columns(self):
        if not self.comm_manager._is_root() or self.master_model is None:
            return
        input_data = getattr(self.data_manager, 'input_data', None)
        obs_bundles = None if input_data is None else input_data.get('obs_bundle')
        if obs_bundles is None or self._features_at_obs_bundles_at_root is None:
            return
        obs_bundles = np.asarray(obs_bundles, dtype=bool)
        if obs_bundles.ndim == 2:
            obs_bundles = np.expand_dims(obs_bundles, axis=0)
        if obs_bundles.ndim != 3:
            raise ValueError(f'Unexpected obs_bundles dimensionality: {obs_bundles.shape}')
        errors_tensor = input_data.get('errors') if input_data else None
        if errors_tensor is None:
            errors_tensor = np.zeros((self.dimensions_cfg.num_simulations, self.dimensions_cfg.num_obs, self.dimensions_cfg.num_items), dtype=np.float64)
        else:
            errors_tensor = np.asarray(errors_tensor, dtype=np.float64)
            if errors_tensor.ndim == 2:
                errors_tensor = np.expand_dims(errors_tensor, axis=0)
            if errors_tensor.ndim != 3:
                raise ValueError(f'Unexpected errors dimensionality: {errors_tensor.shape}')
        num_sims = self.dimensions_cfg.num_simulations
        num_obs = self.dimensions_cfg.num_obs
        sim_indices = np.minimum(np.arange(num_sims), obs_bundles.shape[0] - 1)
        bundles_arr = obs_bundles[sim_indices]
        features_arr = np.broadcast_to(self._features_at_obs_bundles_at_root[None, :, :], (num_sims, num_obs, self.dimensions_cfg.num_features)).copy()
        err_indices = np.minimum(np.arange(num_sims), errors_tensor.shape[0] - 1)
        errors_arr = np.einsum('sij,sij->si', errors_tensor[err_indices], bundles_arr.astype(np.float64))
        flat_bundles = bundles_arr.reshape(-1, self.dimensions_cfg.num_items)
        flat_features = features_arr.reshape(-1, self.dimensions_cfg.num_features)
        flat_errors = errors_arr.reshape(-1)
        reduced_costs = np.full(flat_bundles.shape[0], self.row_generation_cfg.tolerance_optimality + 1.0, dtype=np.float64)
        self._add_columns_to_master(flat_bundles, flat_features, flat_errors, reduced_costs)
        if self.has_columns:
            self.master_model.optimize()

    def _master_iteration(self):
        stop = False
        num_si = self.dimensions_cfg.num_simulations * self.dimensions_cfg.num_obs
        if self.comm_manager._is_root():
            if self.has_columns:
                assert self.master_model is not None
                if self.master_model.Status != GRB.OPTIMAL:
                    self.master_model.optimize()
                if self.master_model.Status != GRB.OPTIMAL:
                    status = self.master_model.Status
                    logger.error('Master status=%s before pricing', status)
                    raise RuntimeError('Column generation master problem is not optimal before pricing.')
                dual_prices = np.array(self.master_model.getAttr(GRB.Attr.Pi, self.feature_constrs), dtype=np.float64)
                agent_penalties = np.zeros(num_si, dtype=np.float64)
                if self.agent_constrs:
                    penalties = self.master_model.getAttr(GRB.Attr.Pi, list(self.agent_constrs.values()))
                    for (idx, _), pi_val in zip(self.agent_constrs.items(), penalties):
                        if idx < num_si:
                            agent_penalties[idx] = pi_val
                self.theta_val = self._compute_theta_from_duals(dual_prices)
            else:
                dual_prices = self.theta_val if self.theta_val is not None else np.zeros(self.dimensions_cfg.num_features, dtype=np.float64)
                agent_penalties = np.zeros(num_si, dtype=np.float64)
        else:
            dual_prices = np.empty(self.dimensions_cfg.num_features, dtype=np.float64)
            agent_penalties = np.empty(num_si, dtype=np.float64)
        dual_prices = self.comm_manager.Bcast(dual_prices, root=0)
        agent_penalties = self.comm_manager.Bcast(agent_penalties, root=0)
        t_pricing = time.perf_counter()
        bundles, features, errors, reduced_costs, max_rc = self._solve_pricing_problem(dual_prices, agent_penalties)
        pricing_time = time.perf_counter() - t_pricing
        if self.comm_manager._is_root():
            logger.info('Max reduced cost: %.6e', max_rc)
            if max_rc <= self.row_generation_cfg.tolerance_optimality:
                stop = True
            else:
                added = self._add_columns_to_master(bundles, features, errors, reduced_costs)
                if added > 0:
                    self.master_model.optimize()
        if self.comm_manager._is_root():
            theta_to_send = self.theta_val.copy() if self.theta_val is not None else np.empty(self.dimensions_cfg.num_features, dtype=np.float64)
        else:
            theta_to_send = np.empty(self.dimensions_cfg.num_features, dtype=np.float64)
        self.theta_val, stop = self.comm_manager.Bcast(theta_to_send, root=0)
        return (bool(stop), pricing_time)

    def solve(self, callback=None):
        logger.info('=== COLUMN GENERATION (DUAL) ===')
        tic = time.perf_counter()
        self.subproblem_manager.initialize_subproblems()
        self._initialize_master_problem()
        total_pricing = 0.0
        for iteration in range(int(self.row_generation_cfg.max_iters)):
            logger.info('ITERATION %d', iteration + 1)
            stop, pricing_time = self._master_iteration()
            total_pricing += pricing_time
            if callback and self.comm_manager._is_root():
                callback({'iteration': iteration + 1, 'theta': None if self.theta_val is None else self.theta_val.copy(), 'objective': None if self.master_model is None else getattr(self.master_model, 'ObjVal', None)})
            if stop and iteration + 1 >= self.row_generation_cfg.min_iters:
                break
        elapsed = time.perf_counter() - tic
        num_iters = iteration + 1
        converged = num_iters < self.row_generation_cfg.max_iters
        if self.comm_manager._is_root():
            logger.info('Column generation completed in %.2fs after %d iterations.', elapsed, num_iters)
            obj_val = self.master_model.ObjVal if hasattr(self.master_model, 'ObjVal') else None
            self.timing_stats = make_timing_stats(elapsed, num_iters, total_pricing)
            self._log_timing_summary(self.timing_stats, obj_val, self.theta_val, header='COLUMN GENERATION SUMMARY')
        else:
            obj_val = None
            self.timing_stats = None
        self.theta_hat = self.theta_val.copy()
        return self._create_result(self.theta_hat, converged, num_iters, obj_val)