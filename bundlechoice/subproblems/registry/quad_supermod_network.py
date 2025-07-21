"""
Quadratic Supermodular Network Subproblem Solver
------------------------------------------------
Implements quadratic supermodular minimization for combinatorial choice estimation using min-cut reduction.
Handles modular and quadratic agent/item features, supports missing data, and is designed for MPI batch solving.
"""
import numpy as np
import networkx as nx
from typing import Any, Optional
from ..base import BatchSubproblemBase

def assert_zero_diagonal(arr: np.ndarray, axis1: int, axis2: int, context: str = ""):
    """Utility to assert that the diagonal along given axes is zero for all slices."""
    diags = np.diagonal(arr, axis1=axis1, axis2=axis2)
    if not np.all(diags == 0):
        raise ValueError(f"Nonzero diagonal detected in {context} (shape {arr.shape})")

class QuadSupermodularSubproblem(BatchSubproblemBase):
    """
    Subproblem for quadratic supermodular minimization via min-cut reduction.
    Handles modular/quadratic agent/item features, missing data, and batch MPI solving.
    """
    def initialize(self) -> None:
        """
        Prepare all agent/item data and precompute slices for efficient solving.
        Handles missing modular/quadratic agent/item data gracefully.
        """
        agent_data = self.local_data.get("agent_data", {})
        item_data = self.local_data.get("item_data", {})

        # Modular and quadratic features (None if missing)
        self.modular_agent = agent_data.get("modular")
        self.quadratic_agent = agent_data.get("quadratic")
        self.modular_item = item_data.get("modular")
        self.quadratic_item = item_data.get("quadratic")

        # Assert that the diagonal of quadratic terms are all zero (if present)
        if self.quadratic_agent is not None:
            for k in range(self.quadratic_agent.shape[-1]):
                assert_zero_diagonal(self.quadratic_agent[:, :, :, k], axis1=1, axis2=2, context="agent quadratic")
        if self.quadratic_item is not None:
            for k in range(self.quadratic_item.shape[-1]):
                assert_zero_diagonal(self.quadratic_item[:, :, k], axis1=0, axis2=1, context="item quadratic")

        self.errors = self.local_data.get("errors")
        self.constraint_mask = self.local_data.get("constraint_mask") or np.ones((self.num_local_agents, self.num_items), dtype=bool)

        # Feature counts
        self.num_mod_agent = self.modular_agent.shape[-1] if self.modular_agent is not None else 0
        self.num_quad_agent = self.quadratic_agent.shape[-1] if self.quadratic_agent is not None else 0
        self.num_mod_item = self.modular_item.shape[-1] if self.modular_item is not None else 0
        self.num_quad_item = self.quadratic_item.shape[-1] if self.quadratic_item is not None else 0

        # Lambda slices
        offset = 0
        self.lambda_mod_agent_slice = slice(offset, offset + self.num_mod_agent); offset += self.num_mod_agent
        self.lambda_quad_agent_slice = slice(offset, offset + self.num_quad_agent); offset += self.num_quad_agent
        self.lambda_mod_item_slice = slice(offset, offset + self.num_mod_item); offset += self.num_mod_item
        self.lambda_quad_item_slice = slice(offset, offset + self.num_quad_item); offset += self.num_quad_item

        # Precompute zeros matrix template and diagonal indices using self.num_items
        self.P_i_j_j_template = np.zeros((self.num_local_agents, self.num_items, self.num_items))
        self.diag_indices = np.diag_indices(self.num_items)  # For advanced diagonal indexing

        self.optimal_bundles = np.zeros((self.num_local_agents, self.num_items), dtype=bool)

    def solve(self, lambda_k: np.ndarray, pb: Optional[Any] = None) -> np.ndarray:
        P_i_j_j = self.build_quadratic_matrix(lambda_k)
        optimal_bundles =  self.optimal_bundles.copy()
        for i in range(self.num_local_agents):
            solver = QuadSubmodularMinimization(-P_i_j_j[i], self.constraint_mask[i])
            optimal_bundle = solver.solve_QSM()
            optimal_bundles[i] = optimal_bundle
        return optimal_bundles

    def build_quadratic_matrix(self, lambda_k: np.ndarray) -> np.ndarray:
        P_i_j_j = self.P_i_j_j_template.copy()

        # Add quadratic agent/item terms if present
        if self.quadratic_agent is not None:
            P_i_j_j += (self.quadratic_agent @ lambda_k[self.lambda_quad_agent_slice])
        if self.quadratic_item is not None:
            P_i_j_j += (self.quadratic_item @ lambda_k[self.lambda_quad_item_slice])
        # Symmetrize each agent's matrix (i.e., ensure P_i_j_j is symmetric in last two dims)
        P_i_j_j = P_i_j_j + np.transpose(P_i_j_j, (0, 2, 1))
        assert np.all(P_i_j_j >= 0)
        # Correct diagonal check for each agent
        diags = np.diagonal(P_i_j_j, axis1=1, axis2=2)
        if not np.all(diags == 0):
            raise ValueError("Nonzero diagonal detected in quadratic matrix after symmetrization.")

        # Add modular agent/item terms to diagonal
        if self.modular_agent is not None:
            P_i_j_j[:, self.diag_indices[0], self.diag_indices[1]] += (self.modular_agent @ lambda_k[self.lambda_mod_agent_slice])
        if self.modular_item is not None:
            P_i_j_j[:, self.diag_indices[0], self.diag_indices[1]] += (self.modular_item @ lambda_k[self.lambda_mod_item_slice])

        # Add errors to diagonal if present
        if self.errors is not None:
            P_i_j_j[:, self.diag_indices[0], self.diag_indices[1]] += self.errors

        return P_i_j_j

class QuadSubmodularMinimization:
    """
    Encapsulates the quadratic supermodular minimization via min-cut reduction.
    Given a quadratic matrix and a constraint mask, builds the posiform, constructs the graph,
    and solves for the optimal bundle using a min-cut algorithm.
    """
    def __init__(self, P_j_j: np.ndarray, constraint_mask: Optional[np.ndarray] = None):
        """
        Initialize the minimization problem.
        Args:
            P_j_j: Quadratic matrix (num_items x num_items)
            constraint_mask: Boolean mask, True for feasible items, False for infeasible
        """
        self.P_j_j = P_j_j
        self.constraint_mask = constraint_mask
        self.num_items = P_j_j.shape[0]
        if constraint_mask is not None:
            self.choice_set = np.where(constraint_mask)[0].tolist()
        else:
            self.choice_set = list(range(self.num_items))

    @staticmethod
    def build_posiform(P_j_j: np.ndarray):
        """
        Convert the quadratic matrix to posiform representation for min-cut.
        Args:
            P_j_j: Quadratic matrix (num_items x num_items)
        Returns:
            a_j_j: Upper-triangular off-diagonal matrix (num_items x num_items)
            a_j: 1D array of node capacities (num_items,)
            positive: Boolean mask for node direction (num_items,)
        """
        a_j_j = np.triu(- P_j_j, k=1)
        assert np.all(a_j_j >= 0)  # This may not hold for modular-only cases
        b_j = np.diag(P_j_j)
        signed_a_j = b_j - a_j_j.sum(axis=1)
        positive = signed_a_j >= 0
        a_j = np.abs(signed_a_j)
        return a_j_j, a_j, positive

    @staticmethod
    def build_graph(a_j_j, a_j, positive, nodes):
        """
        Build the directed graph for the min-cut problem.
        Args:
            a_j_j: Upper-triangular off-diagonal matrix
            a_j: Node capacities
            positive: Boolean mask for node direction
            nodes: List of feasible node indices
        Returns:
            G: networkx.DiGraph for min-cut
        """
        G = nx.DiGraph()
        for i in nodes:
            for j in nodes:
                if j > i: 
                    G.add_edge(i, j, capacity=a_j_j[i, j])
        G.add_node('s')
        G.add_node('t')
        for i in nodes:
            if positive[i]:
                G.add_edge(i, 't', capacity=a_j[i])
            else:
                G.add_edge('s', i, capacity=a_j[i])
        return G

    @staticmethod
    def solve_mincut(G):
        """
        Solve the min-cut problem on the given graph.
        Args:
            G: networkx.DiGraph with source 's' and sink 't'
        Returns:
            S: List of node indices on the source side of the min-cut (excluding 's')
        """
        cut_value, partition = nx.minimum_cut(G, 's', 't', flow_func=nx.algorithms.flow.preflow_push)
        S, T = partition
        S = list(S - {'s'})
        return S

    def solve_QSM(self):
        """
        Solve the quadratic supermodular minimization problem for the given matrix and constraints.
        Returns:
            optimal_bundle: Boolean array (num_items,) indicating selected items
        """
        a_j_j, a_j, positive = self.build_posiform(self.P_j_j)
        G = self.build_graph(a_j_j, a_j, positive, self.choice_set )
        S = self.solve_mincut(G)
        optimal_bundle = np.zeros(self.num_items, dtype=bool)
        optimal_bundle[S] = True
        return optimal_bundle

    