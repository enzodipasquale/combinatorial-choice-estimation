import numpy as np
import gurobipy as gp
from combest.subproblems.solver_base import SubproblemSolver, GurobiMixin


class TwoStageNoEntrySolver(GurobiMixin, SubproblemSolver):

    def initialize(self):
        id_data = self.data_manager.local_data.id_data
        item_data = self.data_manager.local_data.item_data
        M = self.dimensions_cfg.n_items
        n = self.comm_manager.num_local_agent
        self.M, self.R = M, item_data["R"]
        self.beta = item_data["beta"]
        self.rev_chars_1 = item_data["rev_chars_1"]    # (n_rev, M) period 1
        self.rev_chars_2 = item_data["rev_chars_2"]    # (n_rev, M) period 2
        self.syn_chars = item_data["syn_chars"]         # (n_syn, M, M)
        self.n_rev = self.rev_chars_1.shape[0]
        self.n_syn = self.syn_chars.shape[0]
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

    def _syn_obj(self, b, theta_syn):
        """Synergy contribution: Σ_l θ_l b'C_l b"""
        obj = 0
        for l in range(self.n_syn):
            obj += theta_syn[l] * (b @ self.syn_chars[l] @ b)
        return obj

    def solve(self, theta):
        bR = self.beta / self.R
        theta_rev = theta[:self.n_rev]
        theta_syn = theta[self.n_rev:]

        pol = self.data_manager.local_data.id_data["policies"]
        for i, model in enumerate(self.local_problems):
            b_1, b_2_r = self.b_1_vars[i], self.b_2_r_vars[i]

            rev1 = self.rev_chars_1.T @ theta_rev
            rev2 = self.rev_chars_2.T @ theta_rev
            mod_1 = rev1 + self.eps1[i]

            obj = mod_1 @ b_1 + self._syn_obj(b_1, theta_syn)

            for r in range(self.R):
                mod_2 = rev2 + self.eps2[i, r]
                obj += bR * (mod_2 @ b_2_r[r, :]
                             + self._syn_obj(b_2_r[r, :], theta_syn))

            model.setObjective(obj, gp.GRB.MAXIMIZE)
            model.optimize()

            pol["b_1_star"][i] = np.array(b_1.X) > 0.5

            # Re-optimize b_2 per scenario (independent of b_1 without entry cost)
            b_2 = self.q_vars[i]
            for r in range(self.R):
                c_v = rev2 + self.eps2[i, r]
                self.q_models[i].setObjective(
                    c_v @ b_2 + self._syn_obj(b_2, theta_syn),
                    gp.GRB.MAXIMIZE)
                self.q_models[i].optimize()
                pol["b_2_r_V"][i, r] = np.array(b_2.X) > 0.5

            # Q second stage: same problem (no coupling to obs_b)
            for r in range(self.R):
                c_q = rev2 + self.eps2[i, r]
                self.q_models[i].setObjective(
                    c_q @ b_2 + self._syn_obj(b_2, theta_syn),
                    gp.GRB.MAXIMIZE)
                self.q_models[i].optimize()
                pol["b_2_r_Q"][i, r] = np.array(b_2.X) > 0.5

        return pol["b_1_star"]
