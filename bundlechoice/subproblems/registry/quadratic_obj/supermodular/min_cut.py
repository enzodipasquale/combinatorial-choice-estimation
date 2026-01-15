import numpy as np
import networkx as nx
from .quadratic_supermodular_base import QuadraticSupermodular

class QuadraticSOptNetwork(QuadraticSupermodular):

    def solve(self, theta):
        linear, quadratic = self.build_quadratic_matrix(theta)
        ad = self.data_manager.local_data.get('agent_data') or {}
        constraint_mask = ad.get('constraint_mask') if self.has_constraint_mask else None
        n_agents = self.data_manager.num_local_agent
        n_items = self.dimensions_cfg.num_items
        bundles = np.zeros((n_agents, n_items), dtype=bool)
        for i in range(n_agents):
            mask = constraint_mask[i] if constraint_mask is not None else None
            solver = MinCutSolver(-linear[i], -quadratic[i], mask)
            bundles[i] = solver.solve()
        return bundles

class MinCutSolver:

    def __init__(self, b_j, b_jj, constraint_mask=None):
        self.b_j = b_j
        self.b_jj = b_jj
        self.n = b_jj.shape[0]
        if constraint_mask is not None:
            self.nodes = np.where(constraint_mask)[0].tolist() if constraint_mask.dtype == bool else list(constraint_mask)
        else:
            self.nodes = list(range(self.n))

    def solve(self):
        a_jj = -self.b_jj
        a_j = self.b_j - a_jj.sum(axis=1)
        G = self._build_graph(a_jj, a_j)
        _, partition = nx.minimum_cut(G, 's', 't', flow_func=nx.algorithms.flow.preflow_push)
        S = list(partition[0] - {'s'})
        bundle = np.zeros(self.n, dtype=bool)
        bundle[S] = True
        return bundle

    def _build_graph(self, a_jj, a_j):
        scale = self._get_scale(np.concatenate([a_j.flatten(), a_jj.flatten()]))
        a_j = np.round(a_j * scale).astype(np.int64)
        a_jj = np.round(a_jj * scale).astype(np.int64)
        G = nx.DiGraph()
        G.add_nodes_from(['s', 't'] + self.nodes)
        for i in self.nodes:
            if a_j[i] >= 0:
                G.add_edge(i, 't', capacity=a_j[i])
            else:
                G.add_edge('s', i, capacity=-a_j[i])
            for j in self.nodes:
                if j > i:
                    G.add_edge(i, j, capacity=a_jj[i, j])
        G.add_edge('s', 't', capacity=0)
        return G

    def _get_scale(self, arr, digits=12):
        max_val = np.max(np.abs(arr))
        return 1 if max_val == 0 else 10 ** (digits - 1 - int(np.floor(np.log10(max_val))))
