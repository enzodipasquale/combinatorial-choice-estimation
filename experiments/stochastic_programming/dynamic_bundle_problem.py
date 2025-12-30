#!/usr/bin/env python3
"""
Dynamic bundle choice problem with additions and removals using scenario tree.

Solves the stochastic programming formulation with explicit scenario tree and
linearized quadratic terms.
"""

import numpy as np
import gurobipy as gp
from typing import Dict, Any, Optional, Tuple, List
from contextlib import nullcontext
import logging

logger = logging.getLogger(__name__)


class ScenarioTreeNode:
    """Represents a node in the scenario tree."""
    def __init__(self, node_id: int, time: int, parent: Optional['ScenarioTreeNode'] = None, probability: float = 1.0):
        self.node_id = node_id
        self.time = time
        self.parent = parent
        self.probability = probability
        self.children: List['ScenarioTreeNode'] = []


class DynamicBundleProblem:
    """
    Solves dynamic bundle choice problem with scenario tree formulation.
    
    Uses explicit scenario tree with linearized quadratic terms via auxiliary variables.
    """
    
    def __init__(
        self,
        num_actual_items: int,
        num_periods: int,
        beta: float = 0.95,
        branching_factor: int = 2,
        output_flag: int = 0,
        time_limit: Optional[float] = None,
    ):
        """
        Initialize dynamic bundle problem.
        
        Args:
            num_actual_items: Number of actual items (J)
            num_periods: Number of time periods (T_bar)
            beta: Discount factor
            branching_factor: Number of branches at each node (for simple tree)
            output_flag: Gurobi output flag (0=silent, 1=verbose)
            time_limit: Time limit for Gurobi solver
        """
        self.num_actual_items = num_actual_items
        self.num_periods = num_periods
        self.beta = beta
        self.branching_factor = branching_factor
        self.output_flag = output_flag
        self.time_limit = time_limit
        
        # Scenario tree structure
        self.nodes: List[ScenarioTreeNode] = []
        self.node_by_id: Dict[int, ScenarioTreeNode] = {}
        self.nodes_by_time: Dict[int, List[ScenarioTreeNode]] = {}
        
        # Problem data (to be set)
        self.r: Optional[np.ndarray] = None  # Revenue: (node_id, num_actual_items)
        self.c: Optional[np.ndarray] = None  # Operating cost: (num_actual_items,)
        self.F: Optional[np.ndarray] = None  # Fixed cost: (num_actual_items,)
        self.S: Optional[np.ndarray] = None  # Savings matrix: (num_actual_items, num_actual_items)
        self.eta: Optional[np.ndarray] = None  # Stochastic shocks: (node_id, num_actual_items)
        self.b0: Optional[np.ndarray] = None  # Initial bundle: (num_actual_items,)
        
    def build_scenario_tree(self):
        """Build a simple scenario tree with branching_factor branches per node."""
        self.nodes = []
        self.node_by_id = {}
        self.nodes_by_time = {t: [] for t in range(self.num_periods + 1)}
        
        # Root node (n_0) at time 0
        root = ScenarioTreeNode(node_id=0, time=0, parent=None, probability=1.0)
        self.nodes.append(root)
        self.node_by_id[0] = root
        self.nodes_by_time[0].append(root)
        
        next_id = 1
        
        # Build tree level by level
        for t in range(1, self.num_periods + 1):
            for parent in self.nodes_by_time[t - 1]:
                # Create branching_factor children for each parent
                parent_prob = parent.probability
                child_prob = parent_prob / self.branching_factor
                
                for _ in range(self.branching_factor):
                    child = ScenarioTreeNode(
                        node_id=next_id,
                        time=t,
                        parent=parent,
                        probability=child_prob
                    )
                    self.nodes.append(child)
                    self.node_by_id[next_id] = child
                    self.nodes_by_time[t].append(child)
                    parent.children.append(child)
                    next_id += 1
        
        logger.info(f"Built scenario tree with {len(self.nodes)} nodes")
        for t in range(self.num_periods + 1):
            logger.info(f"  Time {t}: {len(self.nodes_by_time[t])} nodes")
        
    def set_data(
        self,
        r: np.ndarray,
        c: np.ndarray,
        F: np.ndarray,
        S: np.ndarray,
        eta: np.ndarray,
        b0: np.ndarray,
    ):
        """
        Set problem data.
        
        Args:
            r: Revenue per node and item, shape (num_nodes, num_actual_items)
            c: Operating cost per item, shape (num_actual_items,)
            F: Fixed cost per item, shape (num_actual_items,)
            S: Savings matrix, shape (num_actual_items, num_actual_items)
            eta: Stochastic shocks, shape (num_nodes, num_actual_items)
            b0: Initial bundle, shape (num_actual_items,)
        """
        num_nodes = len(self.nodes)
        assert r.shape == (num_nodes, self.num_actual_items), f"r shape {r.shape} != ({num_nodes}, {self.num_actual_items})"
        assert c.shape == (self.num_actual_items,), f"c shape {c.shape} != ({self.num_actual_items},)"
        assert F.shape == (self.num_actual_items,), f"F shape {F.shape} != ({self.num_actual_items},)"
        assert S.shape == (self.num_actual_items, self.num_actual_items), f"S shape {S.shape} != ({self.num_actual_items}, {self.num_actual_items})"
        assert eta.shape == (num_nodes, self.num_actual_items), f"eta shape {eta.shape} != ({num_nodes}, {self.num_actual_items})"
        assert b0.shape == (self.num_actual_items,), f"b0 shape {b0.shape} != ({self.num_actual_items},)"
        assert np.all(b0 >= 0) and np.all(b0 <= 1), "b0 must be binary"
        
        self.r = r
        self.c = c
        self.F = F
        self.S = S
        self.eta = eta
        self.b0 = b0.astype(bool)
        
    def solve(self) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], float]:
        """
        Solve the dynamic bundle problem.
        
        Returns:
            d_optimal: Optimal additions by node, dict[node_id] -> array
            e_optimal: Optimal removals by node, dict[node_id] -> array
            obj_value: Optimal objective value
        """
        if self.r is None:
            raise ValueError("Problem data not set. Call set_data() first.")
        
        if len(self.nodes) == 0:
            self.build_scenario_tree()
        
        # Create Gurobi model
        model = gp.Model("DynamicBundle")
        model.setParam('OutputFlag', self.output_flag)
        model.setParam('Threads', 1)
        if self.time_limit is not None:
            model.setParam('TimeLimit', self.time_limit)
        model.setAttr('ModelSense', gp.GRB.MAXIMIZE)
        
        # Enable non-convex MIQP (REQUIRED for quadratic terms in binary variables)
        model.Params.NonConvex = 2
        
        # Parameters optimized for non-convex MIQP
        model.Params.MIPFocus = 1  # Focus on feasible solutions
        model.Params.MIQCPMethod = 1  # Method for MIQCP (try 1 or 2)
        model.Params.Heuristics = 0.2  # Reasonable heuristic effort
        model.Params.MIPGap = 0.02  # 2% optimality gap
        model.Params.Presolve = 2  # Aggressive presolve
        
        # Decision variables for each node (binary only, no auxiliary variables)
        b = {}  # b[n][j]: bundle state at node n, item j
        d = {}  # d[n][j]: addition at node n, item j
        e = {}  # e[n][j]: removal at node n, item j
        
        for node in self.nodes:
            n = node.node_id
            b[n] = model.addVars(self.num_actual_items, vtype=gp.GRB.BINARY, name=f"b_{n}")
            d[n] = model.addVars(self.num_actual_items, vtype=gp.GRB.BINARY, name=f"d_{n}")
            e[n] = model.addVars(self.num_actual_items, vtype=gp.GRB.BINARY, name=f"e_{n}")
        
        # Initial state constraint: b_j^{n_0} = b_j^0
        root = self.node_by_id[0]
        for j in range(self.num_actual_items):
            model.addConstr(
                b[root.node_id][j] == (1 if self.b0[j] else 0),
                name=f"init_{j}"
            )
        
        # State transition constraints: b_j^n = b_j^{pa(n)} + d_j^{pa(n)} - e_j^{pa(n)}
        for node in self.nodes:
            if node.parent is not None:
                n = node.node_id
                pa_n = node.parent.node_id
                for j in range(self.num_actual_items):
                    model.addConstr(
                        b[n][j] == b[pa_n][j] + d[pa_n][j] - e[pa_n][j],
                        name=f"transition_{n}_{j}"
                    )
        
        # Feasibility constraints: e_j^n <= b_j^n, d_j^n <= 1 - b_j^n
        for node in self.nodes:
            n = node.node_id
            for j in range(self.num_actual_items):
                model.addConstr(e[n][j] <= b[n][j], name=f"e_le_b_{n}_{j}")
                model.addConstr(d[n][j] <= 1 - b[n][j], name=f"d_le_1minusb_{n}_{j}")
        
        # Build objective: sum_{t=1}^{T_bar} sum_{n in N_t} π_n β^t [ ... ]
        obj_terms = []
        
        for t in range(1, self.num_periods + 1):
            for node in self.nodes_by_time[t]:
                n = node.node_id
                pi_n = node.probability
                beta_t = self.beta ** t
                
                # Linear terms in b^n: sum_j b_j^n (r_j^n - c_j)
                b_linear = gp.quicksum(
                    b[n][j] * (self.r[n, j] - self.c[j])
                    for j in range(self.num_actual_items)
                )
                
                # Quadratic terms in b^n: sum_{i<j} (c_i + c_j) S_{ij} b_i^n * b_j^n
                # NOTE: Coefficients (c, S) are constant across all nodes and periods
                b_quad = gp.QuadExpr()
                for i in range(self.num_actual_items):
                    for j in range(i + 1, self.num_actual_items):
                        if abs(self.S[i, j]) > 1e-10:
                            # Constant coefficient: (c[i] + c[j]) * S[i,j] same for all nodes
                            coeff = (self.c[i] + self.c[j]) * self.S[i, j]
                            b_quad.add(b[n][i] * b[n][j], coeff)
                
                # Linear terms in d^n: -sum_j d_j^n F_j
                d_linear = gp.quicksum(
                    d[n][j] * (-self.F[j])
                    for j in range(self.num_actual_items)
                )
                
                # Quadratic terms in d^n: sum_{i<j} (F_i + F_j) S_{ij} d_i^n * d_j^n
                # NOTE: Coefficients (F, S) are constant across all nodes and periods
                d_quad = gp.QuadExpr()
                for i in range(self.num_actual_items):
                    for j in range(i + 1, self.num_actual_items):
                        if abs(self.S[i, j]) > 1e-10:
                            # Constant coefficient: (F[i] + F[j]) * S[i,j] same for all nodes
                            coeff = (self.F[i] + self.F[j]) * self.S[i, j]
                            d_quad.add(d[n][i] * d[n][j], coeff)
                
                # Linear terms in e^n: sum_j e_j^n r_j^n
                e_linear = gp.quicksum(
                    e[n][j] * self.r[n, j]
                    for j in range(self.num_actual_items)
                )
                
                # Stochastic shocks: sum_j b[n,j] * η_j^n (only affects chosen items)
                eta_term = gp.quicksum(
                    b[n][j] * self.eta[n, j]
                    for j in range(self.num_actual_items)
                )
                
                period_obj = b_linear + b_quad + d_linear + d_quad + e_linear + eta_term
                obj_terms.append(pi_n * beta_t * period_obj)
        
        model.setObjective(gp.quicksum(obj_terms))
        
        # Optimize
        model.optimize()
        
        if model.status != gp.GRB.OPTIMAL:
            raise RuntimeError(f"Optimization failed with status {model.status}")
        
        # Extract solution
        d_optimal = {}
        e_optimal = {}
        
        for node in self.nodes:
            n = node.node_id
            d_optimal[n] = np.array([d[n][j].x > 0.5 for j in range(self.num_actual_items)], dtype=bool)
            e_optimal[n] = np.array([e[n][j].x > 0.5 for j in range(self.num_actual_items)], dtype=bool)
        
        obj_value = model.ObjVal
        
        return d_optimal, e_optimal, obj_value


def generate_test_data(
    num_nodes: int,
    num_actual_items: int,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Generate test data for dynamic bundle problem.
    
    Args:
        num_nodes: Number of nodes in scenario tree
        num_actual_items: Number of actual items
        seed: Random seed
    
    Returns:
        Dictionary with keys: r, c, F, S, eta, b0
    """
    rng = np.random.default_rng(seed)
    
    # Revenue: positive, varies by node and item
    r = rng.uniform(1.0, 5.0, size=(num_nodes, num_actual_items))
    
    # Operating cost: negative (cost)
    c = rng.uniform(-2.0, -0.5, size=num_actual_items)
    
    # Fixed cost: negative (cost)
    F = rng.uniform(-3.0, -1.0, size=num_actual_items)
    
    # Savings matrix: non-negative, symmetric
    # Increased density for more quadratic interactions
    S = np.zeros((num_actual_items, num_actual_items))
    sparsity = 0.3  # 30% of pairs have interactions (increased from 10%)
    for i in range(num_actual_items):
        for j in range(i + 1, num_actual_items):
            if rng.random() < sparsity:
                S[i, j] = rng.uniform(0.0, 0.5)
    S = S + S.T  # Make symmetric
    np.fill_diagonal(S, 0)  # No self-interactions
    
    # Stochastic shocks: normal distribution
    eta = rng.normal(0.0, 0.5, size=(num_nodes, num_actual_items))
    
    # Initial bundle: random binary
    b0 = rng.integers(0, 2, size=num_actual_items, dtype=bool)
    
    return {
        'r': r,
        'c': c,
        'F': F,
        'S': S,
        'eta': eta,
        'b0': b0,
    }


def main():
    """Test the dynamic bundle problem solver."""
    # Problem parameters
    num_actual_items = 40
    num_periods = 3
    branching_factor = 3
    beta = 0.95
    seed = 42
    
    print("=" * 80)
    print("Dynamic Bundle Choice Problem (Scenario Tree)")
    print("=" * 80)
    print(f"Number of actual items: {num_actual_items}")
    print(f"Number of periods: {num_periods}")
    print(f"Branching factor: {branching_factor}")
    print(f"Discount factor: {beta}")
    print()
    
    # Create problem and build tree
    problem = DynamicBundleProblem(
        num_actual_items=num_actual_items,
        num_periods=num_periods,
        beta=beta,
        branching_factor=branching_factor,
        output_flag=1,
    )
    
    problem.build_scenario_tree()
    num_nodes = len(problem.nodes)
    
    # Generate test data
    print("Generating test data...")
    data = generate_test_data(num_nodes, num_actual_items, seed)
    
    problem.set_data(**data)
    
    print("Solving problem...")
    d_optimal, e_optimal, obj_value = problem.solve()
    
    print()
    print("=" * 80)
    print("Solution")
    print("=" * 80)
    print(f"Optimal objective value: {obj_value:.6f}")
    print()
    
    # Analyze solution quality - check for trivial cases
    print("=" * 80)
    print("Solution Quality Analysis")
    print("=" * 80)
    
    # Compute bundle states from decisions
    b_states = {}
    root = problem.node_by_id[0]
    b_states[0] = data['b0'].copy()
    
    for node in problem.nodes:
        if node.parent is not None:
            n = node.node_id
            pa_n = node.parent.node_id
            b_states[n] = (b_states[pa_n].astype(int) + 
                          d_optimal[pa_n].astype(int) - 
                          e_optimal[pa_n].astype(int)).astype(bool)
    
    # Check for trivial solutions
    all_zero_d = True
    all_one_d = True
    all_zero_e = True
    all_one_e = True
    all_zero_b = True
    all_one_b = True
    
    d_counts = []
    e_counts = []
    b_counts = []
    
    for node in problem.nodes:
        n = node.node_id
        if n > 0:  # Skip root
            d_count = d_optimal[n].sum()
            e_count = e_optimal[n].sum()
            d_counts.append(d_count)
            e_counts.append(e_count)
            
            if d_count > 0:
                all_zero_d = False
            if d_count < num_actual_items:
                all_one_d = False
            if e_count > 0:
                all_zero_e = False
            if e_count < num_actual_items:
                all_one_e = False
        
        if n in b_states:
            b_count = b_states[n].sum()
            b_counts.append(b_count)
            if b_count > 0:
                all_zero_b = False
            if b_count < num_actual_items:
                all_one_b = False
    
    print(f"Bundle states (b):")
    print(f"  All zero: {all_zero_b}")
    print(f"  All one:  {all_one_b}")
    print(f"  Count range: [{min(b_counts) if b_counts else 0}, {max(b_counts) if b_counts else 0}]")
    print(f"  Mean count: {np.mean(b_counts) if b_counts else 0:.2f}")
    print()
    
    print(f"Additions (d):")
    print(f"  All zero: {all_zero_d}")
    print(f"  All one:  {all_one_d}")
    print(f"  Count range: [{min(d_counts) if d_counts else 0}, {max(d_counts) if d_counts else 0}]")
    print(f"  Mean count: {np.mean(d_counts) if d_counts else 0:.2f}")
    print()
    
    print(f"Removals (e):")
    print(f"  All zero: {all_zero_e}")
    print(f"  All one:  {all_one_e}")
    print(f"  Count range: [{min(e_counts) if e_counts else 0}, {max(e_counts) if e_counts else 0}]")
    print(f"  Mean count: {np.mean(e_counts) if e_counts else 0:.2f}")
    print()
    
    # Check if solution is non-trivial
    is_trivial = (all_zero_d and all_zero_e) or (all_one_d and all_one_e) or all_zero_b or all_one_b
    print(f"Solution is {'TRIVIAL' if is_trivial else 'NON-TRIVIAL'}")
    print()
    
    # Show bundle state transitions
    print("=" * 80)
    print("Bundle State Transitions")
    print("=" * 80)
    for t in range(num_periods + 1):
        print(f"Period {t}:")
        for node in problem.nodes_by_time[t]:
            n = node.node_id
            if n in b_states:
                b_count = b_states[n].sum()
                print(f"  Node {n}: {b_count}/{num_actual_items} items")
        print()
    
    # Display solution for each time period
    print("=" * 80)
    print("Detailed Solution")
    print("=" * 80)
    for t in range(1, num_periods + 1):
        print(f"Period {t} nodes:")
        for node in problem.nodes_by_time[t]:
            n = node.node_id
            d_count = d_optimal[n].sum()
            e_count = e_optimal[n].sum()
            b_count = b_states[n].sum() if n in b_states else 0
            print(f"  Node {n} (prob={node.probability:.4f}, parent={node.parent.node_id if node.parent else None}):")
            print(f"    b^n: {b_count}/{num_actual_items} items")
            print(f"    d^n: {d_count} additions")
            print(f"    e^n: {e_count} removals")
            # Show first 10 items as example
            print(f"    d^n (first 10): {d_optimal[n][:10].astype(int)}")
            print(f"    e^n (first 10): {e_optimal[n][:10].astype(int)}")
        print()
    
    print("=" * 80)


if __name__ == "__main__":
    main()
