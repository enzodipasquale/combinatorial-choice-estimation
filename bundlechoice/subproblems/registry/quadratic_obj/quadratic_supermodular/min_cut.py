import numpy as np
import networkx as nx
from ....subproblem_base import SerialSubproblemBase
from .supermodular_quadratic_obj_base import SupermodularQuadraticObjectiveMixin

class QuadraticSupermodularMinCut(SupermodularQuadraticObjectiveMixin, SerialSubproblemBase):

    def initialize_single_pb(self, local_id):
        return self._qinfo.constraint_mask[local_id] if self._qinfo.constraint_mask is not None else None

    def solve_single_pb(self, local_id, theta, constraint_mask):
        linear, quadratic = self._build_linear_coeff_single(local_id, theta), self._build_quadratic_coeff_single(local_id, theta)
        return MinCutSolver(-linear, -quadratic, constraint_mask).solve()

class MinCutSolver:

    def __init__(self, linear_coeff, quadratic_coeff, constraint_mask=None):
        self.linear_coeff, self.quadratic_coeff = linear_coeff, quadratic_coeff
        self.n = quadratic_coeff.shape[0]
        self.nodes = np.where(constraint_mask)[0].tolist() if constraint_mask is not None and constraint_mask.dtype == bool else list(range(self.n)) if constraint_mask is None else list(constraint_mask)

    def solve(self):
        posiform_quadratic_coeff = -self.quadratic_coeff
        posiform_linear_coeff = self.linear_coeff - posiform_quadratic_coeff.sum(axis=1)
        G = self._build_graph(posiform_quadratic_coeff, posiform_linear_coeff)
        _, partition = nx.minimum_cut(G, 's', 't', flow_func=nx.algorithms.flow.preflow_push)
        bundle = np.zeros(self.n, dtype=bool)
        bundle[list(partition[0] - {'s'})] = True
        return bundle

    def _build_graph(self, posiform_quadratic_coeff, posiform_linear_coeff):
        scale = self._get_scale(np.concatenate([posiform_linear_coeff.flatten(), posiform_quadratic_coeff.flatten()]))
        posiform_linear_coeff, posiform_quadratic_coeff = np.round(posiform_linear_coeff * scale).astype(np.int64), np.round(posiform_quadratic_coeff * scale).astype(np.int64)
        G = nx.DiGraph()
        G.add_nodes_from(['s', 't'] + self.nodes)
        for i in self.nodes:
            G.add_edge(i, 't', capacity=posiform_linear_coeff[i]) if posiform_linear_coeff[i] >= 0 else G.add_edge('s', i, capacity=-posiform_linear_coeff[i])
            for j in self.nodes:
                if j > i:
                    G.add_edge(i, j, capacity=posiform_quadratic_coeff[i, j])
        G.add_edge('s', 't', capacity=0)
        return G

    def _get_scale(self, arr, digits=12):
        max_val = np.max(np.abs(arr))
        return 1 if max_val == 0 else 10 ** (digits - 1 - int(np.floor(np.log10(max_val))))
