"""Custom quadratic knapsack solver for zero-noise estimation.

Supports fixed (coefficient=1) terms in both the linear and quadratic
parts of the objective, separate from the estimated parameters.

The subproblem for agent i is:
    max_b  L_i'b + b'Q_i b
    s.t.   w'b <= cap_i,  b binary

where:
    L_i = fixed_linear_i + modular_agent_i @ theta_mod_agent
                         + modular_item @ theta_mod_item
    Q_i = fixed_quad_i   + quad_agent_i @ theta_quad_agent
                         + quad_item @ theta_quad_item
"""
import numpy as np
import gurobipy as gp


def _create_model(gurobi_params=None):
    from combest.utils import suppress_output
    with suppress_output():
        model = gp.Model()
        model.setParam('OutputFlag', 0)
        model.setParam('Threads', 1)
        if gurobi_params:
            for k, v in gurobi_params.items():
                if v is not None:
                    model.setParam(k, v)
        model.setAttr('ModelSense', gp.GRB.MAXIMIZE)
    return model


class ZeroNoiseKnapsack:
    """Knapsack solver with fixed linear and/or quadratic terms."""

    def __init__(self, comm_manager, data_manager, features_manager,
                 dimensions_cfg, gurobi_params=None,
                 fixed_linear=None, fixed_quadratic=None):
        """
        fixed_linear:    (n_local, n_items) array added to L with coeff 1, or None
        fixed_quadratic: (n_local, n_items, n_items) or (n_items, n_items) array
                         added to Q with coeff 1, or None
        """
        self.comm_manager = comm_manager
        self.data_manager = data_manager
        self.features_manager = features_manager
        self.dimensions_cfg = dimensions_cfg
        self.gurobi_params = gurobi_params or {}
        self.fixed_linear = fixed_linear
        self.fixed_quadratic = fixed_quadratic

        # covariate structure
        self._qinfo = data_manager.get_quadratic_data_info()
        self._slices = self._qinfo.slices

    def initialize(self):
        weights = self.data_manager.local_data.item_data['weight']
        capacities = self.data_manager.local_data.id_data['capacity']
        masks = self.data_manager.local_data.id_data.get("item_mask")
        n_local = self.comm_manager.num_local_agent
        n_items = self.dimensions_cfg.n_items

        self._zero_capacity = []
        self.local_problems = []
        for i in range(n_local):
            cap = capacities[i]
            self._zero_capacity.append(cap <= 0)
            model = _create_model(self.gurobi_params)
            ub = masks[i].astype(float) if masks is not None else 1.0
            B = model.addMVar(n_items, vtype=gp.GRB.BINARY, ub=ub, name='bundle')
            model.addConstr(weights @ B <= max(cap, 0))
            model.update()
            self.local_problems.append(model)

    def _build_linear(self, theta):
        n_local = self.comm_manager.num_local_agent
        n_items = self.dimensions_cfg.n_items
        L = np.zeros((n_local, n_items))

        if self.fixed_linear is not None:
            L += self.fixed_linear

        if 'modular_agent' in self._slices:
            L += (self.data_manager.local_data.id_data['modular']
                  @ theta[self._slices['modular_agent']])
        if 'modular_item' in self._slices:
            L += (self.data_manager.local_data.item_data['modular']
                  @ theta[self._slices['modular_item']])
        return L

    def _build_quadratic(self, theta):
        n_local = self.comm_manager.num_local_agent
        n_items = self.dimensions_cfg.n_items
        Q = np.zeros((n_local, n_items, n_items))

        if self.fixed_quadratic is not None:
            Q += self.fixed_quadratic

        if 'quadratic_agent' in self._slices:
            Q += (self.data_manager.local_data.id_data['quadratic']
                  @ theta[self._slices['quadratic_agent']])
        if 'quadratic_item' in self._slices:
            Q += (self.data_manager.local_data.item_data['quadratic']
                  @ theta[self._slices['quadratic_item']])
        return Q

    def solve(self, theta):
        L_all = self._build_linear(theta)
        Q_all = self._build_quadratic(theta)
        n_items = self.dimensions_cfg.n_items
        results = np.zeros((len(self.local_problems), n_items), dtype=bool)
        for i, model in enumerate(self.local_problems):
            if self._zero_capacity[i]:
                continue
            model.setMObjective(Q_all[i], L_all[i], 0.0, sense=gp.GRB.MAXIMIZE)
            model.optimize()
            results[i] = np.array(model.x, dtype=bool)
        return results

    def build_linear_and_quadratic_coef(self, theta):
        return self._build_linear(theta), self._build_quadratic(theta)

    def update_solver_settings(self, settings_dict):
        for model in self.local_problems:
            for param, value in settings_dict.items():
                model.setParam(param, value)
