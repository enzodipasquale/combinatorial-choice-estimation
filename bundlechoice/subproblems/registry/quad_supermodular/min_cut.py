import numpy as np
import networkx as nx
from typing import Any, Optional
from .quadratic_supermodular_base import QuadraticSupermodular

class QuadraticSOptNetwork(QuadraticSupermodular):

    def solve(self, theta, pb=None):
        agent_data = self.data_manager.local_data.get('agent_data') or {}
        constraint_mask = agent_data.get('constraint_mask') if self.has_constraint_mask else None
        linear, quadratic = self.build_quadratic_matrix(theta)
        optimal_bundles = np.zeros((self.data_manager.num_local_agents, self.dimensions_cfg.num_items), dtype=bool)
        for i in range(self.data_manager.num_local_agents):
            agent_mask = constraint_mask[i] if constraint_mask is not None else None
            solver = MinCutSubmodularSolver(-linear[i], -quadratic[i], agent_mask)
            optimal_bundle = solver.solve_QSM()
            optimal_bundles[i] = optimal_bundle
        return optimal_bundles

def get_scale_factor(arr, digits=12):
    max_val = np.max(np.abs(arr))
    if max_val == 0:
        return 1
    return 10 ** (digits - 1 - int(np.floor(np.log10(max_val))))

class MinCutSubmodularSolver:

    def __init__(self, b_j, b_j_j, constraint_mask=None):
        self.b_j = b_j
        self.b_j_j = b_j_j
        self.num_items = b_j_j.shape[0]
        self.constraint_mask = constraint_mask
        if constraint_mask is not None:
            if isinstance(constraint_mask, np.ndarray) and constraint_mask.dtype == bool:
                self.choice_set = np.where(constraint_mask)[0].tolist()
            else:
                self.choice_set = constraint_mask.tolist() if isinstance(constraint_mask, np.ndarray) else constraint_mask
        else:
            self.choice_set = list(range(self.num_items))

    @staticmethod
    def build_graph(a_j_j, a_j, nodes):
        G = nx.DiGraph()
        G.add_node('s')
        G.add_node('t')
        G.add_nodes_from(nodes)
        combined_array = np.concatenate([a_j.flatten(), a_j_j.flatten()])
        scale = get_scale_factor(combined_array)
        a_j = np.round(a_j * scale).astype(np.int64)
        a_j_j = np.round(a_j_j * scale).astype(np.int64)
        for i in nodes:
            if a_j[i] >= 0:
                G.add_edge(i, 't', capacity=a_j[i])
            else:
                G.add_edge('s', i, capacity=-a_j[i])
            for j in nodes:
                if j > i:
                    G.add_edge(i, j, capacity=a_j_j[i, j])
        G.add_edge('s', 't', capacity=0)
        return G

    @staticmethod
    def solve_mincut(G):
        cut_value, partition = nx.minimum_cut(G, 's', 't', flow_func=nx.algorithms.flow.preflow_push)
        S, T = partition
        S = list(S - {'s'})
        return (S, cut_value)

    def solve_QSM(self):
        a_j_j = -self.b_j_j
        a_j = self.b_j - a_j_j.sum(axis=1)
        G = self.build_graph(a_j_j, a_j, self.choice_set)
        S, _ = self.solve_mincut(G)
        optimal_bundle = np.zeros(self.num_items, dtype=bool)
        optimal_bundle[S] = True
        return optimal_bundle