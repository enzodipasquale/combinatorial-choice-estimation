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
        self.rev_chars_1 = item_data["rev_chars_1"]    # (n_rev, M) period 1
        self.rev_chars_2 = item_data["rev_chars_2"]    # (n_rev, M) period 2
        self.state_chars = id_data["state_chars"]      # (n, M) inherited state b_0
        self.syn_chars = item_data["syn_chars"]        # (M, M) pairwise c_jj'
        self.n_rev = self.rev_chars_1.shape[0]
        self.cap = id_data["capacity"]
        obs_raw = id_data.get("obs_bundles", None)
        self.obs_b = obs_raw.astype(float) if obs_raw is not None else np.zeros((n, M))
        self.eps1 = self.data_manager.local_data.errors["eps1"]
        self.eps2 = self.data_manager.local_data.errors["eps2"]

        self.local_problems = []
        self.b_1_vars, self.b_2_r_vars = [], []
        for i in range(n):
            m = self._create_gurobi_model()
            b_1 = m.addMVar(M, vtype=gp.GRB.BINARY, name='b_1')
            b_2_r = m.addMVar((self.R, M), vtype=gp.GRB.BINARY, name='b_2_r')
            m.addConstr(b_1.sum() <= self.cap[i])
            for r in range(self.R):
                m.addConstr(b_2_r[r, :].sum() <= self.cap[i])
            m.update()
            self.local_problems.append(m)
            self.b_1_vars.append(b_1)
            self.b_2_r_vars.append(b_2_r)

        self.q_models, self.q_vars = [], []
        for i in range(n):
            m = self._create_gurobi_model()
            b_2 = m.addMVar(M, vtype=gp.GRB.BINARY, name='b_2')
            m.addConstr(b_2.sum() <= self.cap[i])
            m.update()
            self.q_models.append(m)
            self.q_vars.append(b_2)

        id_data["policies"] = {
            "b_1_star": np.zeros((n, M), dtype=bool),
            "b_2_r_V": np.zeros((n, self.R, M), dtype=bool),
            "b_2_r_Q": np.zeros((n, self.R, M), dtype=bool),
        }

    def solve(self, theta):
        bR = self.beta / self.R
        theta_rev = theta[:self.n_rev]          # revenue coefficients
        theta_s = theta[self.n_rev]             # entry cost
        theta_c = theta[self.n_rev + 1]         # synergy coefficient
        C = self.syn_chars                      # (M, M) pairwise c_jj'

        pol = self.data_manager.local_data.id_data["policies"]
        for i, model in enumerate(self.local_problems):
            b_1, b_2_r = self.b_1_vars[i], self.b_2_r_vars[i]
            b_0_i = self.state_chars[i]         # (M,) inherited state

            # --- Period 1: state b_0, choose b_1 ---
            # Σ_j b_1_j [x_j'θ_r + (1-b_0_j)θ_s + ε1_ij]
            # + Σ_{j<j'} b_1_j b_1_j' θ_c c_jj'
            rev1 = self.rev_chars_1.T @ theta_rev       # (M,)
            rev2 = self.rev_chars_2.T @ theta_rev       # (M,)
            mod_1 = rev1 + (1 - b_0_i) * theta_s + self.eps1[i]

            obj = mod_1 @ b_1
            obj += theta_c * (b_1 @ C @ b_1)

            # --- Period 2: state b_1, choose b_2_r ---
            # Σ_j b_2_rj [x_j'θ_r + (1-b_1_j)θ_s + ε2_irj]
            # + Σ_{j<j'} b_2_rj b_2_rj' θ_c c_jj'
            for r in range(self.R):
                mod_2 = rev2 + (1 - b_1) * theta_s + self.eps2[i, r]
                obj += bR * (mod_2 @ b_2_r[r, :]
                             + theta_c * (b_2_r[r, :] @ C @ b_2_r[r, :]))

            model.setObjective(obj, gp.GRB.MAXIMIZE)
            model.optimize()

            pol["b_1_star"][i] = np.array(b_1.X) > 0.5
            b_1_star_f = pol["b_1_star"][i]
            # pol["b_2_r_V"][i] = np.array(b_2_r.X) > 0.5

            # Re-optimize b_2 per scenario given b_1_star (independent knapsacks)
            b_2 = self.q_vars[i]
            for r in range(self.R):
                c_v = rev2 + (1 - b_1_star_f) * theta_s + self.eps2[i, r]
                self.q_models[i].setObjective(
                    c_v @ b_2 + theta_c * (b_2 @ C @ b_2),
                    gp.GRB.MAXIMIZE)
                self.q_models[i].optimize()
                pol["b_2_r_V"][i, r] = np.array(b_2.X) > 0.5

            # Q second stage: obs_b fixed
            for r in range(self.R):
                c_q = rev2 + (1 - self.obs_b[i]) * theta_s + self.eps2[i, r]
                self.q_models[i].setObjective(
                    c_q @ b_2 + theta_c * (b_2 @ C @ b_2),
                    gp.GRB.MAXIMIZE)
                self.q_models[i].optimize()
                pol["b_2_r_Q"][i, r] = np.array(b_2.X) > 0.5

        return pol["b_1_star"]
