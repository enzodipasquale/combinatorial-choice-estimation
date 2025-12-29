"""
Column generation solver for modular bundle choice estimation.

Solves the dual of the row generation master problem via column generation.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import gurobipy as gp
import numpy as np
from gurobipy import GRB
from numpy.typing import NDArray

from bundlechoice.utils import get_logger, suppress_output
from .base import BaseEstimationManager

logger = get_logger(__name__)


class ColumnGenerationManager(BaseEstimationManager):
    """
    Column generation algorithm for the dual formulation of bundle choice estimation.

    Master (dual) problem:
        maximise      Σ_{s,i,b} μ_{s,i,b} * errors_{s,i,b}
        subject to    Σ_{s,i,b} μ_{s,i,b} * x_{s,i,b,k} = obs_features_k        ∀ k
                      Σ_{b} μ_{s,i,b} ≤ 1                                        ∀ (s,i)
                      μ_{s,i,b} ≥ 0

    Pricing problem (per (s,i)):
        max_b  errors_{s,i,b} - π^T x_{s,i,b} - σ_{s,i}
    where π, σ are dual prices of the feature-matching constraints and
    agent-balance constraints respectively.
    """

    def __init__(
        self,
        comm_manager: Any,
        dimensions_cfg: Any,
        row_generation_cfg: Any,
        data_manager: Any,
        feature_manager: Any,
        subproblem_manager: Any,
        theta_init: Optional[NDArray[np.float64]] = None,
    ) -> None:
        super().__init__(
            comm_manager=comm_manager,
            dimensions_cfg=dimensions_cfg,
            data_manager=data_manager,
            feature_manager=feature_manager,
            subproblem_manager=subproblem_manager,
        )

        self.row_generation_cfg = row_generation_cfg
        self.theta_init = theta_init

        self.master_model: Optional[gp.Model] = None
        self.theta_val: Optional[NDArray[np.float64]] = None
        self.theta_hat: Optional[NDArray[np.float64]] = None
        self.timing_stats: Optional[Dict[str, float]] = None

        # Master problem structures (root only)
        self.feature_constrs: List[gp.Constr] = []
        self.agent_constrs: Dict[int, gp.Constr] = {}
        self.column_vars: List[gp.Var] = []
        self.active_columns: List[Dict[str, Any]] = []
        self.has_columns: bool = False
        self._pricing_cache: Tuple[
            Optional[NDArray[np.bool_]],
            Optional[NDArray[np.float64]],
            Optional[NDArray[np.float64]],
            Optional[NDArray[np.float64]],
        ] = (None, None, None, None)
        self.theta_upper = self._expand_bounds(getattr(row_generation_cfg, "theta_ubs", None), np.inf)
        self.theta_lower = self._expand_bounds(getattr(row_generation_cfg, "theta_lbs", None), -np.inf)
        self.alpha_vars: List[Optional[gp.Var]] = []
        self.beta_vars: List[Optional[gp.Var]] = []

    # ------------------------------------------------------------------ #
    # Master initialisation
    # ------------------------------------------------------------------ #
    def _setup_gurobi_model(self) -> gp.Model:
        defaults = {"Method": 0, "LPWarmStart": 2, "OutputFlag": 0}
        params = {**defaults, **self.row_generation_cfg.gurobi_settings}

        with suppress_output():
            model = gp.Model()
            for param, value in params.items():
                if value is not None:
                    model.setParam(param, value)
        model.ModelSense = GRB.MAXIMIZE
        return model

    def _initialize_master_problem(self) -> None:
        obs_features = self.get_obs_features()

        if self.is_root():
            self.master_model = self._setup_gurobi_model()
            self.feature_constrs = []
            self.agent_constrs = {}
            self.column_vars = []
            self.active_columns = []
            self.has_columns = False
            self.alpha_vars = []
            self.beta_vars = []

            for k in range(self.num_features):
                expr = gp.LinExpr()

                alpha_var: Optional[gp.Var] = None
                upper = self.theta_upper[k]
                if np.isfinite(upper):
                    alpha_var = self.master_model.addVar(
                        lb=0.0,
                        obj=-float(upper),
                        name=f"alpha[{k}]",
                    )
                    expr.addTerms(1.0, alpha_var)
                self.alpha_vars.append(alpha_var)

                beta_var: Optional[gp.Var] = None
                lower = self.theta_lower[k]
                if np.isfinite(lower):
                    beta_var = self.master_model.addVar(
                        lb=0.0,
                        obj=float(lower),
                        name=f"beta[{k}]",
                    )
                    expr.addTerms(-1.0, beta_var)
                self.beta_vars.append(beta_var)

                rhs = float(obs_features[k] * max(1, self.num_simuls))
                if np.isfinite(self.theta_lower[k]) and self.theta_lower[k] >= 0:
                    constr = self.master_model.addConstr(expr >= rhs, name=f"feature_match[{k}]")
                else:
                    constr = self.master_model.addConstr(expr == rhs, name=f"feature_match[{k}]")
                self.feature_constrs.append(constr)

            if self.theta_init is not None:
                self.theta_val = self.theta_init.astype(np.float64).copy()
            else:
                self.theta_val = np.zeros(self.num_features, dtype=np.float64)

            self.master_model.update()
            self._add_initial_columns()
            logger.info("Column generation master initialised (dual).")
        else:
            self.theta_val = np.empty(self.num_features, dtype=np.float64)

        self.theta_val = self.comm_manager.broadcast_array(self.theta_val, root=0)

    # ------------------------------------------------------------------ #
    # Pricing problem
    # ------------------------------------------------------------------ #
    def _solve_pricing_problem(
        self,
        dual_prices: NDArray[np.float64],
        agent_penalties: NDArray[np.float64],
    ) -> Tuple[Optional[NDArray[np.bool_]], Optional[NDArray[np.float64]], Optional[NDArray[np.float64]], float]:
        """Solve pricing problems on local ranks and gather results."""
        modified_theta = np.zeros(len(dual_prices), dtype=np.float64)
        for k in range(len(dual_prices)):
            if np.isfinite(self.theta_lower[k]) and self.theta_lower[k] >= 0:
                modified_theta[k] = max(self.theta_lower[k], -dual_prices[k])
            else:
                modified_theta[k] = -dual_prices[k]
        
        modified_theta = np.clip(modified_theta, self.theta_lower, self.theta_upper)
        
        try:
            local_bundles = self.subproblem_manager.solve_local(modified_theta)
        except Exception as e:
            logger.error("Pricing problem failed with theta=%s, lower=%s, upper=%s: %s", 
                        modified_theta, self.theta_lower, self.theta_upper, e, exc_info=True)
            raise

        bundles_all = self.comm_manager.concatenate_array_at_root_fast(local_bundles, root=0)
        features_all = self.feature_manager.compute_gathered_features(local_bundles)
        errors_all = self.feature_manager.compute_gathered_errors(local_bundles)

        max_reduced_cost = 0.0
        if self.is_root() and features_all is not None and errors_all is not None:
            penalties = agent_penalties[: len(errors_all)]
            scaled_errors = errors_all / max(1, self.num_simuls)
            reduced_costs = scaled_errors - features_all @ dual_prices - penalties
            max_reduced_cost = float(np.max(reduced_costs))
            self._pricing_cache = (bundles_all, features_all, scaled_errors, reduced_costs)
        else:
            self._pricing_cache = (None, None, None, None)

        max_reduced_cost = self.comm_manager.broadcast_from_root(max_reduced_cost, root=0)
        return (*self._pricing_cache, max_reduced_cost)

    # ------------------------------------------------------------------ #
    # Column management
    # ------------------------------------------------------------------ #
    def _ensure_agent_constraint(self, idx: int) -> gp.Constr:
        if idx not in self.agent_constrs:
            constr = self.master_model.addConstr(gp.LinExpr() == 1.0, name=f"agent_balance[{idx}]")
            self.agent_constrs[idx] = constr
        return self.agent_constrs[idx]

    def _add_columns_to_master(
        self,
        bundles: Optional[NDArray[np.bool_]],
        features: Optional[NDArray[np.float64]],
        errors: Optional[NDArray[np.float64]],
        reduced_costs: Optional[NDArray[np.float64]],
    ) -> int:
        if not self.is_root() or bundles is None or features is None or errors is None or reduced_costs is None:
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

            var = self.master_model.addVar(
                obj=float(errors[idx] / max(1, self.num_simuls)),
                lb=0.0,
                column=column,
                name=f"mu_{idx}_{len(self.column_vars)}",
            )
            self.column_vars.append(var)
            self.active_columns.append(
                {
                    "bundle": bundles[idx].copy(),
                    "features": features[idx].copy(),
                    "errors": float(errors[idx]),
                    "agent_index": idx,
                }
            )
            num_added += 1

        if num_added > 0:
            self.master_model.update()
            self.has_columns = True
            logger.info("Added %d new columns (violated bundles).", num_added)
        else:
            logger.info("No violated bundles detected.")

        return num_added

    def _add_initial_columns(self) -> None:
        """Add observed bundles as initial columns to ensure master feasibility."""
        if not self.is_root() or self.master_model is None:
            return

        input_data = getattr(self.data_manager, "input_data", None)
        obs_bundles = None if input_data is None else input_data.get("obs_bundle")
        if obs_bundles is None or self.agents_obs_features is None:
            return

        obs_bundles = np.asarray(obs_bundles, dtype=bool)
        if obs_bundles.ndim == 2:
            obs_bundles = np.expand_dims(obs_bundles, axis=0)
        if obs_bundles.ndim != 3:
            raise ValueError(f"Unexpected obs_bundles dimensionality: {obs_bundles.shape}")

        errors_tensor = input_data.get("errors") if input_data else None
        if errors_tensor is None:
            errors_tensor = np.zeros((self.num_simuls, self.num_agents, self.num_items), dtype=np.float64)
        else:
            errors_tensor = np.asarray(errors_tensor, dtype=np.float64)
            if errors_tensor.ndim == 2:
                errors_tensor = np.expand_dims(errors_tensor, axis=0)
            if errors_tensor.ndim != 3:
                raise ValueError(f"Unexpected errors dimensionality: {errors_tensor.shape}")

        bundles_arr = np.zeros((self.num_simuls, self.num_agents, self.num_items), dtype=bool)
        features_arr = np.zeros((self.num_simuls, self.num_agents, self.num_features), dtype=np.float64)
        errors_arr = np.zeros((self.num_simuls, self.num_agents), dtype=np.float64)

        for s in range(self.num_simuls):
            bundle_slice = obs_bundles[min(s, obs_bundles.shape[0] - 1)]
            error_slice = errors_tensor[min(s, errors_tensor.shape[0] - 1)]
            for i in range(self.num_agents):
                bundle = bundle_slice[i]
                bundles_arr[s, i] = bundle
                features_arr[s, i] = self.agents_obs_features[i]
                errors_arr[s, i] = error_slice[i] @ bundle

        flat_bundles = bundles_arr.reshape(-1, self.num_items)
        flat_features = features_arr.reshape(-1, self.num_features)
        flat_errors = errors_arr.reshape(-1)

        reduced_costs = np.full(
            flat_bundles.shape[0],
            self.row_generation_cfg.tolerance_optimality + 1.0,
            dtype=np.float64,
        )
        self._add_columns_to_master(flat_bundles, flat_features, flat_errors, reduced_costs)
        if self.has_columns:
            self.master_model.optimize()

    # ------------------------------------------------------------------ #
    # Master iteration
    # ------------------------------------------------------------------ #
    def _master_iteration(self, timing: Dict[str, float]) -> bool:
        stop = False
        num_si = self.num_simuls * self.num_agents

        if self.is_root():
            t_prep = datetime.now()
            if self.has_columns:
                assert self.master_model is not None
                if self.master_model.Status != GRB.OPTIMAL:
                    self.master_model.optimize()
                if self.master_model.Status != GRB.OPTIMAL:
                    status = self.master_model.Status
                    logger.error("Master status=%s before pricing", status)
                    raise RuntimeError("Column generation master problem is not optimal before pricing.")
                dual_prices = np.array(
                    self.master_model.getAttr(GRB.Attr.Pi, self.feature_constrs),
                    dtype=np.float64,
                )
                agent_penalties = np.zeros(num_si, dtype=np.float64)
                if self.agent_constrs:
                    penalties = self.master_model.getAttr(
                        GRB.Attr.Pi, list(self.agent_constrs.values())
                    )
                    for (idx, _), pi_val in zip(self.agent_constrs.items(), penalties):
                        if idx < num_si:
                            agent_penalties[idx] = pi_val
                
                self.theta_val = np.zeros(self.num_features, dtype=np.float64)
                for k in range(self.num_features):
                    if np.isfinite(self.theta_lower[k]) and self.theta_lower[k] >= 0:
                        self.theta_val[k] = max(self.theta_lower[k], -dual_prices[k])
                    else:
                        self.theta_val[k] = -dual_prices[k]
                self.theta_val = np.clip(self.theta_val, self.theta_lower, self.theta_upper)
            else:
                dual_prices = self.theta_val if self.theta_val is not None else np.zeros(self.num_features, dtype=np.float64)
                agent_penalties = np.zeros(num_si, dtype=np.float64)

            timing["master_prep"] = (datetime.now() - t_prep).total_seconds()
        else:
            dual_prices = np.empty(self.num_features, dtype=np.float64)
            agent_penalties = np.empty(num_si, dtype=np.float64)

        dual_prices = self.comm_manager.broadcast_array(dual_prices, root=0)
        agent_penalties = self.comm_manager.broadcast_array(agent_penalties, root=0)

        t_pricing = datetime.now()
        bundles, features, errors, reduced_costs, max_rc = self._solve_pricing_problem(dual_prices, agent_penalties)
        timing["pricing"] = (datetime.now() - t_pricing).total_seconds()

        if self.is_root():
            logger.info("Max reduced cost: %.6e", max_rc)
            if max_rc <= self.row_generation_cfg.tolerance_optimality:
                stop = True
            else:
                t_update = datetime.now()
                added = self._add_columns_to_master(bundles, features, errors, reduced_costs)
                timing["master_update"] = (datetime.now() - t_update).total_seconds()

                if added > 0:
                    t_opt = datetime.now()
                    self.master_model.optimize()
                    timing["master_optimize"] = (datetime.now() - t_opt).total_seconds()
                else:
                    timing["master_optimize"] = 0.0
        else:
            timing["master_update"] = 0.0
            timing["master_optimize"] = 0.0

        if self.is_root():
            theta_to_send = self.theta_val.copy() if self.theta_val is not None else np.empty(self.num_features, dtype=np.float64)
        else:
            # Pre-allocate array for buffer-based broadcast (must match root's shape/dtype)
            theta_to_send = np.empty(self.num_features, dtype=np.float64)

        t_broadcast = datetime.now()
        self.theta_val, stop = self.comm_manager.broadcast_array_with_flag(theta_to_send, stop, root=0)
        timing["mpi_broadcast"] = (datetime.now() - t_broadcast).total_seconds()

        return bool(stop)

    # ------------------------------------------------------------------ #
    # Public solve
    # ------------------------------------------------------------------ #
    def solve(self, callback: Optional[Callable[[Dict[str, Any]], None]] = None) -> NDArray[np.float64]:
        logger.info("=== COLUMN GENERATION (DUAL) ===")
        tic = datetime.now()

        self.subproblem_manager.initialize_local()
        self._initialize_master_problem()

        timing_breakdown = {
            "pricing": [],
            "master_prep": [],
            "master_update": [],
            "master_optimize": [],
            "mpi_broadcast": [],
            "callback": [],
        }

        for iteration in range(int(self.row_generation_cfg.max_iters)):
            logger.info("ITERATION %d", iteration + 1)
            iter_timing: Dict[str, float] = {}

            stop = self._master_iteration(iter_timing)

            for key in timing_breakdown:
                if key in iter_timing:
                    timing_breakdown[key].append(iter_timing[key])

            if callback and self.is_root():
                t_cb = datetime.now()
                callback(
                    {
                        "iteration": iteration + 1,
                        "theta": None if self.theta_val is None else self.theta_val.copy(),
                        "objective": None if self.master_model is None else getattr(self.master_model, "ObjVal", None),
                    }
                )
                timing_breakdown["callback"].append((datetime.now() - t_cb).total_seconds())

            if stop and iteration + 1 >= self.row_generation_cfg.min_iters:
                break

        elapsed = (datetime.now() - tic).total_seconds()
        if self.is_root():
            logger.info("Column generation completed in %.2fs after %d iterations.", elapsed, iteration + 1)

        self.theta_hat = self.theta_val.copy()
        return self.theta_hat

    def _expand_bounds(self, bound, fill_value: float) -> NDArray[np.float64]:
        arr = np.full(self.num_features, fill_value, dtype=np.float64)
        if bound is None:
            return arr
        if np.isscalar(bound):
            arr[:] = float(bound)
            return arr
        bound_list = list(bound)
        if len(bound_list) != self.num_features:
            raise ValueError("Length of theta bounds does not match number of features.")
        for idx, val in enumerate(bound_list):
            if val is None:
                arr[idx] = fill_value
            else:
                arr[idx] = float(val)
        return arr

