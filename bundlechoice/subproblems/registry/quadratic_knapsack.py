"""
Quadratic knapsack subproblem solver using Gurobi.

Solves max utility (with quadratic terms) subject to weight constraint.
"""

import numpy as np
import gurobipy as gp
import logging
from typing import Any, Optional
from contextlib import nullcontext
from numpy.typing import NDArray
from ..base import SerialSubproblemBase
from bundlechoice.utils import suppress_output, get_logger

logger = get_logger(__name__)


class QuadraticKnapsackSubproblem(SerialSubproblemBase):
    """Quadratic knapsack solver: max utility with quadratic terms, weight constraint."""
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._last_bundles: Optional[NDArray[np.bool_]] = None  # Cache last solutions for MIP starts

    def initialize(self, local_id: int) -> Any:
        """Initialize Gurobi model with weight constraint."""
        output_flag = self.config.settings.get("OutputFlag", 0)
        context = suppress_output() if output_flag == 0 else nullcontext()
        gurobi_logger = logging.getLogger("gurobipy")
        old_gurobi_level = gurobi_logger.level if output_flag == 1 else None
        try:
            if output_flag == 1:
                gurobi_logger.setLevel(logging.WARNING)
            with context:
                subproblem = gp.Model()
                subproblem.setParam('OutputFlag', output_flag)
                subproblem.setParam('Threads', 1)
                time_limit = self.config.settings.get("TimeLimit")
                if time_limit is not None:
                    subproblem.setParam("TimeLimit", time_limit)
                subproblem.setAttr('ModelSense', gp.GRB.MAXIMIZE)
                B_j = subproblem.addVars(self.num_items, vtype=gp.GRB.BINARY)
                weights = self.local_data["item_data"]["weights"]
                capacity = self.local_data["agent_data"]["capacity"][local_id]
                subproblem.addConstr(gp.quicksum(weights[j] * B_j[j] for j in range(self.num_items)) <= capacity)
                subproblem.update()        
        finally:
            if output_flag == 1 and old_gurobi_level is not None:
                gurobi_logger.setLevel(old_gurobi_level)
        return subproblem

    def solve(self, local_id: int, theta: NDArray[np.float64], pb: Any) -> NDArray[np.bool_]:
        """Solve quadratic knapsack: set linear + quadratic objective, optimize."""
        output_flag = self.config.settings.get("OutputFlag", 0)
        gurobi_logger = logging.getLogger("gurobipy")
        old_gurobi_level = None
        if output_flag == 1:
            old_gurobi_level = gurobi_logger.level
            gurobi_logger.setLevel(logging.WARNING)
        
        try:
            L_j = self._build_L_j(local_id, theta)
            Q_j_j = self._build_Q_j_j(local_id, theta)
            
            # Use setMObjective for faster quadratic objective setting (matrix-based)
            # This is much faster than building QuadExpr term-by-term for large problems
            # setMObjective(Q, c, constant, sense) where Q is quadratic matrix, c is linear coefficients
            pb.setMObjective(Q_j_j, L_j, 0.0, sense=gp.GRB.MAXIMIZE)
            
            # Set MIP start from cached solution if available
            B_j = pb.getVars()  # Need vars for MIP start
            if self._last_bundles is not None and local_id < len(self._last_bundles):
                last_bundle = self._last_bundles[local_id]
                for j in range(len(B_j)):
                    B_j[j].Start = float(last_bundle[j])
                pb.update()
            
            pb.optimize()
            
            # Handle case where Gurobi times out or has no solution
            # Use MIP start if available, otherwise use current solution
            try:
                if pb.status in (gp.GRB.OPTIMAL, gp.GRB.TIME_LIMIT, gp.GRB.SUBOPTIMAL):
                    optimal_bundle = np.array(pb.x, dtype=bool)
                else:
                    # No solution available, use MIP start if available
                    if self._last_bundles is not None and local_id < len(self._last_bundles):
                        optimal_bundle = self._last_bundles[local_id].copy()
                    else:
                        optimal_bundle = np.zeros(self.num_items, dtype=bool)
            except (gp.GurobiError, AttributeError):
                # Gurobi has no solution (e.g., timeout before any solution found)
                if self._last_bundles is not None and local_id < len(self._last_bundles):
                    optimal_bundle = self._last_bundles[local_id].copy()
                else:
                    optimal_bundle = np.zeros(self.num_items, dtype=bool)
            
            self._check_mip_gap(pb, local_id)
            
            # Cache solution for next solve
            if self._last_bundles is None:
                self._last_bundles = np.zeros((self.num_local_agents, self.num_items), dtype=bool)
            self._last_bundles[local_id] = optimal_bundle
        finally:
            if output_flag == 1 and old_gurobi_level is not None:
                gurobi_logger.setLevel(old_gurobi_level)
        
        return optimal_bundle

    def _build_Q_j_j(self, local_id: int, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        """Build quadratic matrix Q_j_j from agent/item quadratic features."""
        info = self.data_manager.get_data_info()
        agent_data = self.local_data.get("agent_data", {})
        item_data = self.local_data.get("item_data", {})
        
        Q_j_j = np.zeros((self.num_items, self.num_items))
        offset = 0
        
        if info["has_modular_agent"]:
            offset += info["num_modular_agent"]
        
        if info["has_quadratic_agent"]:
            quadratic_agent = agent_data["quadratic"]
            Q_j_j += (quadratic_agent[local_id] @ theta[offset:offset + info["num_quadratic_agent"]])
            offset += info["num_quadratic_agent"]
        
        if info["has_modular_item"]:
            offset += info["num_modular_item"]
        
        if info["has_quadratic_item"]:
            quadratic_item = item_data["quadratic"]
            Q_j_j += (quadratic_item @ theta[offset:offset + info["num_quadratic_item"]])
            offset += info["num_quadratic_item"]
        
        return Q_j_j

    def _build_L_j(self, local_id: int, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        """Build linear coefficients L_j from agent/item modular features."""
        info = self.data_manager.get_data_info()
        error_j = self.local_data["errors"][local_id]
        agent_data = self.local_data.get("agent_data", {})
        item_data = self.local_data.get("item_data", {})
        
        L_j = error_j.copy()
        offset = 0
        
        if info["has_modular_agent"]:
            modular_agent = agent_data["modular"]
            L_j += (modular_agent[local_id] @ theta[offset:offset + info["num_modular_agent"]])
            offset += info["num_modular_agent"]
        
        if info["has_modular_item"]:
            modular_item = item_data["modular"]
            L_j += (modular_item @ theta[offset:offset + info["num_modular_item"]])
            offset += info["num_modular_item"]
        
        return L_j

    def _check_mip_gap(self, subproblem: Any, local_id: int) -> None:
        """Check MIP gap and warn if exceeds tolerance."""
        MIPGap_tol = self.config.settings.get("MIPGap_tol")
        if MIPGap_tol is not None:
            if subproblem.MIPGap > float(MIPGap_tol):
                logger.warning(
                    f"Subproblem {local_id} (rank {getattr(self.data_manager, 'rank', '?')}): "
                    f"MIPGap {subproblem.MIPGap:.4g} > tol {MIPGap_tol}, value: {subproblem.objVal}"
                )
