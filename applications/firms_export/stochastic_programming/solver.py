import numpy as np
import gurobipy as gp
from combest.subproblems.solver_base import SubproblemSolver, GurobiMixin


class TwoStageSolver(GurobiMixin, SubproblemSolver):

    def initialize(self):
        ld = self.data_manager.local_data
        M = self.dimensions_cfg.n_items
        n = self.comm_manager.num_local_agent
        self.M, self.R = M, ld["item_data"]["R"]
        self.beta = ld["item_data"]["beta"]
        self.rev = ld["item_data"]["revenue"]
        self.cap = ld["id_data"]["capacity"]
        self.state = ld["id_data"]["state"].astype(float)
        self.obs_b = ld["id_data"]["obs_bundles"].astype(float)
        self.eps1 = ld["errors"]["eps1"]
        self.eps2 = ld["errors"]["eps2"]

        self.local_problems = []
        self.b_vars, self.d_vars = [], []
        for i in range(n):
            m = self._create_gurobi_model()
            b = m.addMVar(M, vtype=gp.GRB.BINARY, name='b')
            d = m.addMVar((self.R, M), vtype=gp.GRB.BINARY, name='d')
            m.addConstr(b.sum() <= self.cap[i])
            for r in range(self.R):
                m.addConstr(d[r, :].sum() <= self.cap[i])
            m.update()
            self.local_problems.append(m)
            self.b_vars.append(b)
            self.d_vars.append(d)

        self.q_models, self.q_vars = [], []
        for i in range(n):
            m = self._create_gurobi_model()
            d = m.addMVar(M, vtype=gp.GRB.BINARY, name='d')
            m.addConstr(d.sum() <= self.cap[i])
            m.update()
            self.q_models.append(m)
            self.q_vars.append(d)

        ld["policies"] = {
            "b_star": np.zeros((n, M), dtype=bool),
            "d_V": np.zeros((n, self.R, M), dtype=bool),
            "d_Q": np.zeros((n, self.R, M), dtype=bool),
        }

    def solve(self, theta):
        bR = self.beta / self.R
        pol = self.data_manager.local_data["policies"]
        for i, model in enumerate(self.local_problems):
            b, d = self.b_vars[i], self.d_vars[i]

            # stage 1: rev·θ₀ + (1-state)·θ₁ + ε₁
            c_b = self.rev * theta[0] + (1 - self.state[i]) * theta[1] + self.eps1[i]
            # stage 2: rev·θ₀ + θ₁ + ε₂, with bilinear -θ₁·b·d for entry cost (1-b)·θ₁
            c_d = (self.rev * theta[0] + theta[1] + self.eps2[i])  # (R, M)
            obj = c_b @ b + bR * ((c_d * d).sum() - theta[1] * b @ d.sum(0))

            model.setObjective(obj, gp.GRB.MAXIMIZE)
            model.optimize()

            pol["b_star"][i] = np.array(b.X) > 0.5
            pol["d_V"][i] = np.array(d.X) > 0.5

            # Q second stage: obs_b fixed, no bilinear — R independent knapsacks
            dq = self.q_vars[i]
            for r in range(self.R):
                c_q = self.rev * theta[0] + (1 - self.obs_b[i]) * theta[1] + self.eps2[i, r]
                self.q_models[i].setObjective(c_q @ dq, gp.GRB.MAXIMIZE)
                self.q_models[i].optimize()
                pol["d_Q"][i, r] = np.array(dq.X) > 0.5

        return pol["b_star"]
