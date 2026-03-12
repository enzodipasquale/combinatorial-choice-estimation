import numpy as np
import gurobipy as gp
from combest.subproblems.solver_base import SubproblemSolver, GurobiMixin


class TwoStageSolver(GurobiMixin, SubproblemSolver):

    def initialize(self):
        id_data = self.data_manager.local_data.id_data
        item_data = self.data_manager.local_data.item_data
        M = self.dimensions_cfg.n_items
        n = self.comm_manager.num_local_agent
        self.M, self.R = M, item_data["R"]
        self.beta = item_data["beta"]
        self.rev_chars = item_data["rev_chars"]        # (n_rev, M)
        self.state_chars = id_data["state_chars"]      # (n, M)
        self.syn_chars = item_data["syn_chars"]        # (M, M) pairwise c_jj'
        self.n_rev = self.rev_chars.shape[0]
        self.cap = id_data["capacity"]
        self.obs_b = id_data["obs_bundles"].astype(float)
        self.eps1 = self.data_manager.local_data.errors["eps1"]
        self.eps2 = self.data_manager.local_data.errors["eps2"]

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

        id_data["policies"] = {
            "b_star": np.zeros((n, M), dtype=bool),
            "d_V": np.zeros((n, self.R, M), dtype=bool),
            "d_Q": np.zeros((n, self.R, M), dtype=bool),
        }

    def solve(self, theta):
        bR = self.beta / self.R
        theta_rev = theta[:self.n_rev]          # revenue coefficients
        theta_s = theta[self.n_rev]             # entry cost
        theta_c = theta[self.n_rev + 1]         # synergy coefficient
        C = self.syn_chars                      # (M, M) pairwise c_jj'

        pol = self.data_manager.local_data.id_data["policies"]
        for i, model in enumerate(self.local_problems):
            b, d = self.b_vars[i], self.d_vars[i]
            s_i = self.state_chars[i]           # (M,) inherited state

            # --- Stage 1: state s_i, choose b ---
            # Σ_j b_j [x_j'θ_r + (1-s_ij)θ_s + ε1_ij]
            # + Σ_{j<j'} b_j b_j' θ_c c_jj'
            # (exit is free)
            rev = self.rev_chars.T @ theta_rev          # (M,)
            c_b = rev + (1 - s_i) * theta_s + self.eps1[i]

            obj = c_b @ b
            obj += theta_c * (b @ C @ b)

            # --- Stage 2: state b, choose d_r ---
            # Σ_j d_rj [x_j'θ_r + (1-b_j)θ_s + ε2_irj]
            # + Σ_{j<j'} d_rj d_rj' θ_c c_jj'
            for r in range(self.R):
                c_d = rev + (1 - b) * theta_s + self.eps2[i, r]
                obj += bR * (c_d @ d[r, :]
                             + theta_c * (d[r, :] @ C @ d[r, :]))

            model.setObjective(obj, gp.GRB.MAXIMIZE)
            model.optimize()

            pol["b_star"][i] = np.array(b.X) > 0.5
            b_star_f = pol["b_star"][i].astype(float)

            # Re-optimize d per scenario given b_star (independent knapsacks)
            dq = self.q_vars[i]
            for r in range(self.R):
                c_v = rev + (1 - b_star_f) * theta_s + self.eps2[i, r]
                self.q_models[i].setObjective(
                    c_v @ dq + theta_c * (dq @ C @ dq),
                    gp.GRB.MAXIMIZE)
                self.q_models[i].optimize()
                pol["d_V"][i, r] = np.array(dq.X) > 0.5

            # Q second stage: obs_b fixed
            obs_b_i = self.obs_b[i]
            for r in range(self.R):
                c_q = rev + (1 - obs_b_i) * theta_s + self.eps2[i, r]
                self.q_models[i].setObjective(
                    c_q @ dq + theta_c * (dq @ C @ dq),
                    gp.GRB.MAXIMIZE)
                self.q_models[i].optimize()
                pol["d_Q"][i, r] = np.array(dq.X) > 0.5

        return pol["b_star"]