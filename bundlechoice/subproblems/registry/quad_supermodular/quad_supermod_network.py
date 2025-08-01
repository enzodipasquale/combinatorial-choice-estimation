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
        P_i_j_j = self.build_quadratic_matrix(theta)
        optimal_bundles =  np.zeros((self.num_local_agents, self.num_items), dtype=bool)
        for i in range(self.num_local_agents):
            solver = MinCutSubmodularSolver(-P_i_j_j[i], self.constraint_mask[i])
            optimal_bundle = solver.solve_QSM()
            optimal_bundles[i] = optimal_bundle
        return optimal_bundles

class MinCutSubmodularSolver:
    """
    Encapsulates the quadratic supermodular minimization via min-cut reduction.
    Given a quadratic matrix and a constraint mask, builds the posiform, constructs the graph,
    and solves for the optimal bundle using a min-cut algorithm.
    """
    def __init__(self, b_j_j: np.ndarray, constraint_mask: Optional[np.ndarray] = None):
        """
        Initialize the minimization problem.
        Args:
            b_j_j: Upper Triangular matrix (num_items x num_items)
            constraint_mask: Boolean mask, True for feasible items, False for infeasible
        """

        self.b_j_j = b_j_j
        self.num_items = b_j_j.shape[0]
        assert np.all(b_j_j * np.tril(np.ones((self.num_items, self.num_items)), k=-1) == 0), "b_j_j must be upper triangular"
        assert np.all(b_j_j * (1 - np.eye(self.num_items)) <= 0), "b_j_j must have non-sign off-diagonal elements"
   
        self.constraint_mask = constraint_mask
        if constraint_mask is not None:
            self.choice_set = np.where(constraint_mask)[0].tolist()
        else:
            self.choice_set = list(range(self.num_items))

    def build_posiform(self):
        """
        Convert the quadratic matrix to posiform representation for min-cut.
        Args:
            b_j_j: Quadratic matrix (num_items x num_items)
        Returns:
            a_j_j: Upper-triangular off-diagonal matrix (num_items x num_items)
            a_j: 1D array of node capacities (num_items,)
            sign: Boolean mask for node direction (num_items,)
        """
        a_j_j = - self.b_j_j * np.triu(np.ones((self.num_items, self.num_items)), k=1)
        a_j = np.diag(self.b_j_j) - a_j_j.sum(axis=1)

        return a_j_j, a_j

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
        a_j *= 1e14
        a_j_j *= 1e14
        for i in nodes:
            if a_j[i] >= 0:
                G.add_edge(i, 't', capacity= int(a_j[i]))
            else:
                G.add_edge('s', i, capacity= -int(a_j[i]))
            for j in nodes:
                if j > i: 
                    G.add_edge(i, j, capacity=int(a_j_j[i, j]))
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
        cut_value, partition = nx.minimum_cut(G, 's', 't', flow_func=nx.algorithms.flow.edmonds_karp)
        S,T = partition
        S = list(S - {'s'})
        
        return S, cut_value

    def solve_QSM(self):
        """
        Solve the quadratic supermodular minimization problem for the given matrix and constraints.
        Returns:
            optimal_bundle: Boolean array (num_items,) indicating selected items
        """
        a_j_j, a_j = self.build_posiform()
        G = self.build_graph(a_j_j, a_j, self.choice_set )
        S, cut_value = self.solve_mincut(G)
        optimal_bundle = np.zeros(self.num_items, dtype=bool)
        optimal_bundle[S] = True
    
        return optimal_bundle

    
    def plot_graph_(self, G):
        pos = nx.spring_layout(G, seed=77)

        # Draw the graph
        nx.draw(G, pos, with_labels=True, node_color='lightblue', font_weight='bold', 
                arrows=True, edge_color='gray', connectionstyle='arc3,rad=0.1')

        # Round and draw edge labels
        edge_labels = {
            (u, v): f"{d['capacity']:.2f}"
            for u, v, d in G.edges(data=True)
        }
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')

        plt.title("Directed Graph with Edge Capacities")
        plt.axis('off')
        plt.show()