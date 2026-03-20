import numpy as np
import networkx as nx
from ....solver_base import SubproblemSolver
from .supermodular_quadratic_obj_base import SupermodularQuadraticObjectiveMixin

class QuadraticSupermodularMinCutSolver(SupermodularQuadraticObjectiveMixin, SubproblemSolver):

    def initialize(self):
        mask = self.data_manager.local_data.id_data["constraint_mask"]
        self._solvers = [MinCutSolver(mask[i] if mask is not None else None,
                                      self.dimensions_cfg.n_items)
                         for i in range(self.comm_manager.num_local_agent)]

    def solve(self, theta):
        L_all = self._build_linear_coeff_batch(theta)
        Q_all = self._build_quadratic_coeff_batch(theta)
        n_agents = len(self._solvers)
        results = np.zeros((n_agents, self.dimensions_cfg.n_items), dtype=bool)
        for i, solver in enumerate(self._solvers):
            results[i] = solver.solve(-L_all[i], -Q_all[i])
        return results

class MinCutSolver:

    def __init__(self, constraint_mask, n_items):
        self.n = n_items
        if constraint_mask is None:
            self.nodes = list(range(n_items))
        elif constraint_mask.dtype == bool:
            self.nodes = np.where(constraint_mask)[0].tolist()
        else:
            self.nodes = list(constraint_mask)

    def solve(self, linear_coeff, quadratic_coeff):
        posiform_quadratic_coeff = -quadratic_coeff
        posiform_linear_coeff = linear_coeff - posiform_quadratic_coeff.sum(axis=1)
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
