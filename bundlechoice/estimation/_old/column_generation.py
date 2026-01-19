import time
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from bundlechoice.utils import get_logger, suppress_output
from .base import BaseEstimationManager
logger = get_logger(__name__)

class ColumnGenerationManager(BaseEstimationManager):

    def __init__(self, comm_manager, config, data_manager, oracles_manager, subproblem_manager):
        super().__init__(comm_manager, config, data_manager, oracles_manager, subproblem_manager)
        self.master_model = None
        self.theta_val = None
        self.theta_hat = None
        self.timing_stats = None
        self.feature_constrs = []
        self.agent_constrs = {}
        self.column_vars = []
        self.active_columns = []
        self.has_columns = False
        self._pricing_cache = (None, None, None, None)
        self.alpha_vars = []
        self.beta_vars = []
        self.cfg = self.config.row_generation
        self.dim = self.config.dimensions

    def _setup_colgen_model(self):
        model = self._setup_gurobi_model(self.cfg.gurobi_settings)
        model.ModelSense = GRB.MAXIMIZE
        return model

    def _initialize_master_problem(self):
        obs_features = self._compute_theta_obj_coef()
        theta_ubs = self.cfg.theta_ubs if np.isscalar(self.cfg.theta_ubs) else np.array(self.cfg.theta_ubs)
        theta_lbs = self.cfg.theta_lbs if np.isscalar(self.cfg.theta_lbs) else np.array(self.cfg.theta_lbs)
        
        if self.comm_manager._is_root():
            self.master_model = self._setup_colgen_model()
            self.feature_constrs = []
            self.agent_constrs = {}
            self.column_vars = []
            self.active_columns = []
            self.has_columns = False
            self.alpha_vars = []
            self.beta_vars = []
            
            for k in range(self.dim.n_features):
                expr = gp.LinExpr()
                upper = theta_ubs[k] if hasattr(theta_ubs, '__getitem__') else theta_ubs
                if np.isfinite(upper):
                    alpha_var = self.master_model.addVar(lb=0.0, obj=-float(upper), name=f'alpha[{k}]')
                    expr.addTerms(1.0, alpha_var)
                    self.alpha_vars.append(alpha_var)
                else:
                    self.alpha_vars.append(None)
                    
                lower = theta_lbs[k] if hasattr(theta_lbs, '__getitem__') else theta_lbs
                if np.isfinite(lower):
                    beta_var = self.master_model.addVar(lb=0.0, obj=float(lower), name=f'beta[{k}]')
                    expr.addTerms(-1.0, beta_var)
                    self.beta_vars.append(beta_var)
                else:
                    self.beta_vars.append(None)
                    
                rhs = float(obs_features[k] * max(1, self.dim.n_simulations))
                if np.isfinite(lower) and lower >= 0:
                    constr = self.master_model.addConstr(expr >= rhs, name=f'feature_match[{k}]')
                else:
                    constr = self.master_model.addConstr(expr == rhs, name=f'feature_match[{k}]')
                self.feature_constrs.append(constr)
                
            self.theta_val = np.zeros(self.dim.n_features, dtype=np.float64)
            self.master_model.update()
            self._add_initial_columns()
            logger.info('Column generation master initialised (dual).')
        else:
            self.theta_val = np.empty(self.dim.n_features, dtype=np.float64)
        self.comm_manager.Bcast(self.theta_val)

    def _compute_theta_from_duals(self, dual_prices):
        theta_lbs = self.cfg.theta_lbs
        theta_ubs = self.cfg.theta_ubs
        lbs = np.full(self.dim.n_features, theta_lbs) if np.isscalar(theta_lbs) else np.array(theta_lbs)
        ubs = np.full(self.dim.n_features, theta_ubs) if np.isscalar(theta_ubs) else np.array(theta_ubs)
        has_nonneg_lower = np.isfinite(lbs) & (lbs >= 0)
        theta = np.where(has_nonneg_lower, np.maximum(lbs, -dual_prices), -dual_prices)
        return np.clip(theta, lbs, ubs)

    def _solve_pricing_problem(self, dual_prices, agent_penalties):
        modified_theta = self._compute_theta_from_duals(dual_prices)
        local_bundles = self.subproblem_manager.solve_subproblems(modified_theta)
        bundles_all = self.comm_manager.Gatherv_by_row(local_bundles, row_counts=self.data_manager.agent_counts)
        features_local = self.oracles_manager.features_oracle(local_bundles)
        errors_local = self.oracles_manager.error_oracle(local_bundles)
        features_all = self.comm_manager.Gatherv_by_row(features_local, row_counts=self.data_manager.agent_counts)
        errors_all = self.comm_manager.Gatherv_by_row(errors_local, row_counts=self.data_manager.agent_counts)
        
        max_reduced_cost = 0.0
        if self.comm_manager._is_root() and features_all is not None and errors_all is not None:
            penalties = agent_penalties[:len(errors_all)]
            scaled_errors = errors_all / max(1, self.dim.n_simulations)
            reduced_costs = scaled_errors - features_all @ dual_prices - penalties
            max_reduced_cost = float(np.max(reduced_costs))
            self._pricing_cache = (bundles_all, features_all, scaled_errors, reduced_costs)
        else:
            self._pricing_cache = (None, None, None, None)
        max_reduced_cost = self.comm_manager.bcast(max_reduced_cost)
        return (*self._pricing_cache, max_reduced_cost)

    def _ensure_agent_constraint(self, idx):
        if idx not in self.agent_constrs:
            constr = self.master_model.addConstr(gp.LinExpr() == 1.0, name=f'agent_balance[{idx}]')
            self.agent_constrs[idx] = constr
        return self.agent_constrs[idx]

    def _add_columns_to_master(self, bundles, features, errors, reduced_costs):
        if not self.comm_manager._is_root() or bundles is None:
            return 0
        tolerance = self.cfg.tolerance_optimality
        num_added = 0
        for idx in range(len(errors)):
            if reduced_costs[idx] <= tolerance:
                continue
            agent_constr = self._ensure_agent_constraint(idx)
            column = gp.Column()
            for k, coeff in enumerate(features[idx]):
                if coeff != 0.0:
                    column.addTerms(float(coeff), self.feature_constrs[k])
            column.addTerms(1.0, agent_constr)
            var = self.master_model.addVar(obj=float(errors[idx]), lb=0.0, column=column, 
                                           name=f'mu_{idx}_{len(self.column_vars)}')
            self.column_vars.append(var)
            self.active_columns.append({'bundle': bundles[idx].copy(), 'features': features[idx].copy(), 
                                        'errors': float(errors[idx]), 'agent_index': idx})
            num_added += 1
        if num_added > 0:
            self.master_model.update()
            self.has_columns = True
            logger.info('Added %d new columns', num_added)
        return num_added

    def _add_initial_columns(self):
        if not self.comm_manager._is_root() or self.master_model is None:
            return
        # Stashed: initial column logic depends on data_manager structure

    def _master_iteration(self):
        stop = False
        num_si = self.dim.n_simulations * self.dim.n_obs
        if self.comm_manager._is_root():
            if self.has_columns:
                if self.master_model.Status != GRB.OPTIMAL:
                    self.master_model.optimize()
                if self.master_model.Status != GRB.OPTIMAL:
                    raise RuntimeError('Column generation master not optimal')
                dual_prices = np.array(self.master_model.getAttr(GRB.Attr.Pi, self.feature_constrs), dtype=np.float64)
                agent_penalties = np.zeros(num_si, dtype=np.float64)
                if self.agent_constrs:
                    penalties = self.master_model.getAttr(GRB.Attr.Pi, list(self.agent_constrs.values()))
                    for (idx, _), pi_val in zip(self.agent_constrs.items(), penalties):
                        if idx < num_si:
                            agent_penalties[idx] = pi_val
                self.theta_val = self._compute_theta_from_duals(dual_prices)
            else:
                dual_prices = self.theta_val if self.theta_val is not None else np.zeros(self.dim.n_features, dtype=np.float64)
                agent_penalties = np.zeros(num_si, dtype=np.float64)
        else:
            dual_prices = np.empty(self.dim.n_features, dtype=np.float64)
            agent_penalties = np.empty(num_si, dtype=np.float64)
            
        self.comm_manager.Bcast(dual_prices)
        self.comm_manager.Bcast(agent_penalties)
        
        bundles, features, errors, reduced_costs, max_rc = self._solve_pricing_problem(dual_prices, agent_penalties)
        
        if self.comm_manager._is_root():
            logger.info('Max reduced cost: %.6e', max_rc)
            if max_rc <= self.cfg.tolerance_optimality:
                stop = True
            else:
                added = self._add_columns_to_master(bundles, features, errors, reduced_costs)
                if added > 0:
                    self.master_model.optimize()
                    
        if self.comm_manager._is_root():
            theta_to_send = self.theta_val.copy() if self.theta_val is not None else np.empty(self.dim.n_features, dtype=np.float64)
        else:
            theta_to_send = np.empty(self.dim.n_features, dtype=np.float64)
        self.comm_manager.Bcast(theta_to_send)
        self.theta_val = theta_to_send
        stop = self.comm_manager.bcast(stop)
        return stop

    def solve(self, callback=None):
        logger.info('=== COLUMN GENERATION (DUAL) ===')
        t0 = time.perf_counter()
        self.subproblem_manager.initialize_subproblems()
        self._initialize_master_problem()
        
        iteration = 0
        while iteration < self.cfg.max_iters:
            logger.info('ITERATION %d', iteration + 1)
            stop = self._master_iteration()
            if callback and self.comm_manager._is_root():
                callback({'iteration': iteration + 1, 'theta': self.theta_val.copy() if self.theta_val is not None else None,
                          'objective': getattr(self.master_model, 'ObjVal', None) if self.master_model else None})
            if stop and iteration + 1 >= self.cfg.min_iters:
                break
            iteration += 1
            
        elapsed = time.perf_counter() - t0
        num_iters = iteration + 1
        converged = num_iters < self.cfg.max_iters
        
        if self.comm_manager._is_root():
            logger.info('Column generation completed in %.2fs after %d iterations.', elapsed, num_iters)
            obj_val = self.master_model.ObjVal if hasattr(self.master_model, 'ObjVal') else None
        else:
            obj_val = None
            
        self.theta_hat = self.theta_val.copy() if self.theta_val is not None else None
        return self._create_result(num_iters, self.master_model, self.theta_hat, self.cfg)
