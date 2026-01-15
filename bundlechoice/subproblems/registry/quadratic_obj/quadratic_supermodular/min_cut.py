import numpy as np
import networkx as nx
from ....subproblem_base import SerialSubproblemBase
from .quadratic_supermodular_base import SupermodularQuadraticObjectiveMixin

class QuadraticSOptNetwork(SupermodularQuadraticObjectiveMixin, SerialSubproblemBase):

    def initialize_single_pb(self, local_id):
        if local_id == 0:
            self._init_quadratic_info()
        ad = self.data_manager.local_data['agent_data']
        mask = ad.get('constraint_mask', None)
        return mask[local_id] if mask is not None else None

    def solve_single_pb(self, local_id, theta, constraint_mask):
        linear, quadratic = self._build_linear_coeff_single(local_id, theta), self._build_quadratic_coeff_single(local_id, theta)
        return MinCutSolver(-linear, -quadratic, constraint_mask).solve()

class MinCutSolver:

    def __init__(self, b_j, b_jj, constraint_mask=None):
        self.b_j, self.b_jj = b_j, b_jj
        self.n = b_jj.shape[0]
        self.nodes = np.where(constraint_mask)[0].tolist() if constraint_mask is not None and constraint_mask.dtype == bool else list(range(self.n)) if constraint_mask is None else list(constraint_mask)

    def solve(self):
        a_jj = -self.b_jj
        a_j = self.b_j - a_jj.sum(axis=1)
        G = self._build_graph(a_jj, a_j)
        _, partition = nx.minimum_cut(G, 's', 't', flow_func=nx.algorithms.flow.preflow_push)
        bundle = np.zeros(self.n, dtype=bool)
        bundle[list(partition[0] - {'s'})] = True
        return bundle

    def _build_graph(self, a_jj, a_j):
        scale = self._get_scale(np.concatenate([a_j.flatten(), a_jj.flatten()]))
        a_j, a_jj = np.round(a_j * scale).astype(np.int64), np.round(a_jj * scale).astype(np.int64)
        G = nx.DiGraph()
        G.add_nodes_from(['s', 't'] + self.nodes)
        for i in self.nodes:
            G.add_edge(i, 't', capacity=a_j[i]) if a_j[i] >= 0 else G.add_edge('s', i, capacity=-a_j[i])
            for j in self.nodes:
                if j > i:
                    G.add_edge(i, j, capacity=a_jj[i, j])
        G.add_edge('s', 't', capacity=0)
        return G

    def _get_scale(self, arr, digits=12):
        max_val = np.max(np.abs(arr))
        return 1 if max_val == 0 else 10 ** (digits - 1 - int(np.floor(np.log10(max_val))))
