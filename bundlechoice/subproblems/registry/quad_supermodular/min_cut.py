"""
Quadratic Supermodular Network Subproblem Solver
------------------------------------------------
Implements quadratic supermodular minimization for combinatorial choice estimation using min-cut reduction.
Handles modular and quadratic agent/item features, supports missing data, and is designed for MPI batch solving.
"""
import numpy as np
import networkx as nx
from typing import Any, Optional
from .quadratic_supermodular_base import QuadraticSupermodular


class QuadraticSOptNetwork(QuadraticSupermodular):
    """
    Subproblem for quadratic supermodular minimization via min-cut reduction.
    Handles modular/quadratic agent/item features, missing data, and batch MPI solving.
    """
    def solve(self, theta: np.ndarray, pb: Optional[Any] = None) -> np.ndarray:
        agent_data = self.data_manager.local_data.get("agent_data") or {}
        constraint_mask = agent_data.get("constraint_mask") if self.has_constraint_mask else None
        linear, quadratic = self.build_quadratic_matrix(theta)
        optimal_bundles =  np.zeros((self.data_manager.num_local_agents, self.dimensions_cfg.num_items), dtype=bool)
        for i in range(self.data_manager.num_local_agents):
            agent_mask = constraint_mask[i] if constraint_mask is not None else None
            solver = MinCutSubmodularSolver(-linear[i], -quadratic[i], agent_mask)
            optimal_bundle = solver.solve_QSM()
            optimal_bundles[i] = optimal_bundle
        return optimal_bundles

def get_scale_factor(arr, digits=12):
    """
    Calculate scale factor to preserve specified number of significant digits.
    Args:
        arr: Input array
        digits: Number of significant digits to preserve (default: 12)
    Returns:
        Scale factor for converting to integers
    """
    max_val = np.max(np.abs(arr))
    if max_val == 0:
        return 1
    
    return 10**(digits - 1 - int(np.floor(np.log10(max_val))))

class MinCutSubmodularSolver:
    """
    Encapsulates the quadratic supermodular minimization via min-cut reduction.
    Given a quadratic matrix and a constraint mask, builds the posiform, constructs the graph,
    and solves for the optimal bundle using a min-cut algorithm.
    """
    def __init__(self, b_j: np.ndarray, b_j_j: np.ndarray, constraint_mask: Optional[np.ndarray] = None):
        """
        Initialize the minimization problem.
            Args:   
                b_j: Linear matrix (num_items,)
                b_j_j: Upper Triangular matrix (num_items x num_items)
                constraint_mask: Boolean mask, True for feasible items, False for infeasible
        """
        self.b_j = b_j
        self.b_j_j = b_j_j
        self.num_items = b_j_j.shape[0]
 
        self.constraint_mask = constraint_mask
        if constraint_mask is not None:
            # Convert to indices (works for both boolean masks and index arrays)
            if isinstance(constraint_mask, np.ndarray) and constraint_mask.dtype == bool:
                self.choice_set = np.where(constraint_mask)[0].tolist()
            else:
                self.choice_set = constraint_mask.tolist() if isinstance(constraint_mask, np.ndarray) else constraint_mask
        else:
            self.choice_set = list(range(self.num_items))


    @staticmethod
    def build_graph(a_j_j, a_j, nodes):
        """
        Build the undirected graph for the min-cut problem.
        Args:
            a_j_j: Upper-triangular off-diagonal matrix
            a_j: Node capacities
            nodes: List of feasible node indices
        Returns:
            G: networkx.Graph for min-cut
        """
        G = nx.DiGraph()
        G.add_node('s')
        G.add_node('t')
        G.add_nodes_from(nodes)
        # Use the same scale for both arrays since they represent the same problem
        combined_array = np.concatenate([a_j.flatten(), a_j_j.flatten()])
        scale = get_scale_factor(combined_array)
        a_j = np.round(a_j * scale).astype(np.int64)
        a_j_j = np.round(a_j_j * scale).astype(np.int64)
        for i in nodes:
            if a_j[i] >= 0:
                G.add_edge(i, 't', capacity= a_j[i])
            else:
                G.add_edge('s', i, capacity= -a_j[i])
            for j in nodes:
                if j > i: 
                    G.add_edge(i, j, capacity=a_j_j[i, j])
        G.add_edge('s', 't', capacity=0)
        return G
        
    @staticmethod
    def solve_mincut(G):
        """
        Solve the min-cut problem on the given graph.
        Args:
            G: networkx.Graph with source 's' and sink 't'
        Returns:
            S: List of node indices on the source side of the min-cut (excluding 's')
        """
        # Use push_relabel instead of edmonds_karp for better performance on dense graphs
        cut_value, partition = nx.minimum_cut(G, 's', 't', flow_func=nx.algorithms.flow.preflow_push)
        S,T = partition
        S = list(S - {'s'})
        
        return S, cut_value

    def solve_QSM(self):
        """
        Solve the quadratic supermodular minimization problem for the given matrix and constraints.
        Returns:
            optimal_bundle: Boolean array (num_items,) indicating selected items
        """
        # Build posiform
        a_j_j = - self.b_j_j
        a_j = self.b_j - a_j_j.sum(axis=1)
        # Build graph
        G = self.build_graph(a_j_j, a_j, self.choice_set )
        # Solve min-cut
        S, _ = self.solve_mincut(G)
        # Return optimal bundle
        optimal_bundle = np.zeros(self.num_items, dtype=bool)
        optimal_bundle[S] = True
    
        return optimal_bundle
