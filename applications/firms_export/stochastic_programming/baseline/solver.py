import numpy as np
import gurobipy as gp
from combest.subproblems.solver_base import SubproblemSolver, create_gurobi_model


class EntryProblem:

    def __init__(self, M, R, solver_cfg):
        self.M, self.R = M, R

        self.model = create_gurobi_model(solver_cfg)
        self.b_1 = self.model.addMVar(M, vtype=gp.GRB.BINARY, name='b_1')
        self.b_2_r = self.model.addMVar((R, M), vtype=gp.GRB.BINARY, name='b_2_r')
        self.e_2_r = self.model.addMVar((R, M), vtype=gp.GRB.BINARY, name='e_2_r')
        for r in range(R):
            self.model.addConstr(self.e_2_r[r, :] <= self.b_2_r[r, :])
            self.model.addConstr(self.e_2_r[r, :] <= 1 - self.b_1)
            self.model.addConstr(self.e_2_r[r, :] >= self.b_2_r[r, :] - self.b_1)
        self.model.update()

    def solve_joint(self, mod_1, mod_2, entry_2, syn_1, syn_coeff_2, C_d):
        b_1, b_2_r, e_2_r = self.b_1, self.b_2_r, self.e_2_r

        obj = (mod_1 + syn_1) @ b_1
        obj += (1 / self.R) * (mod_2 * b_2_r).sum()
        obj += (1 / self.R) * (entry_2[None, :] * e_2_r).sum()
        obj += syn_coeff_2 * (1 / self.R) * (e_2_r @ C_d @ b_1).sum()

        self.model.setObjective(obj, gp.GRB.MAXIMIZE)
        self.model.optimize()
        return np.array(self.b_1.X) > 0.5, np.array(self.b_2_r.X) > 0.5

    def solve_second_stage(self, b_1_fixed, mod_2, entry_2, syn_coeff_2, C_d):
        b_2_r_out = np.zeros((self.R, self.M), dtype=bool)
        switch = 1 - b_1_fixed
        syn_2 = syn_coeff_2 * switch * (C_d @ b_1_fixed)
        entry_mod = switch * entry_2
        for r in range(self.R):
            b_2_r_out[r] = mod_2[r] + entry_mod + syn_2 > 0
        return b_2_r_out


class TwoStageSolver(SubproblemSolver):

    def initialize(self):
        id_data = self.data_manager.local_data.id_data
        item_data = self.data_manager.local_data.item_data
        M = self.dimensions_cfg.n_items
        n = self.comm_manager.num_local_agent
        self.M = M
        self.R = item_data["R"]
        self.rev_chars_1 = id_data["rev_chars_1"]
        self.rev_chars_2_d = id_data["rev_chars_2_d"]
        self.state_chars = id_data["state_chars"]
        self.entry_chars = item_data["entry_chars"]
        self.entry_chars_2 = item_data["entry_chars_2"]
        self.syn_chars = item_data["syn_chars"]
        self.discount_2 = item_data["discount_2"]
        self.n_rev = self.rev_chars_1.shape[1]
        obs_raw = id_data.get("obs_bundles", None)
        self.obs_b = obs_raw.astype(float) if obs_raw is not None else np.zeros((n, M))
        self.eps_1 = self.data_manager.local_data.errors["eps_1"]
        self.eps_2 = self.data_manager.local_data.errors["eps_2"]
        self.C_d = self.syn_chars * self.entry_chars[:, None]
        self.C_d_2 = item_data["C_d_2"]

        self.problems = []
        self.local_problems = []
        for i in range(n):
            ep = EntryProblem(M, self.R, self.subproblem_cfg)
            self.problems.append(ep)
            self.local_problems.append(ep.model)

        id_data["policies"] = {
            "b_1_star": np.zeros((n, M), dtype=bool),
            "b_2_r_V": np.zeros((n, self.R, M), dtype=bool),
            "b_2_r_Q": np.zeros((n, self.R, M), dtype=bool),
        }

    def _unpack_theta(self, theta):
        theta_rev = theta[:self.n_rev]
        theta_s = theta[self.n_rev]
        theta_sd = theta[self.n_rev + 1]
        theta_syn = theta[self.n_rev + 2]

        rev1 = np.einsum('inm,n->im', self.rev_chars_1, theta_rev)
        rev2_d = np.einsum('inm,n->im', self.rev_chars_2_d, theta_rev)
        entry = theta_s + theta_sd * self.entry_chars
        entry_2 = self.discount_2 * theta_s + theta_sd * self.entry_chars_2

        return rev1, rev2_d, entry, entry_2, theta_syn

    def solve(self, theta):
        rev1, rev2_d, entry, entry_2, theta_syn = self._unpack_theta(theta)

        pol = self.data_manager.local_data.id_data["policies"]
        for i, ep in enumerate(self.problems):
            b_0 = self.state_chars[i]
            switch_1 = 1 - b_0
            syn_1 = theta_syn * switch_1 * (self.C_d @ b_0)
            mod_1 = rev1[i] + switch_1 * entry + self.eps_1[i]
            mod_2 = rev2_d[i] + self.eps_2[i]

            b_1_star, b_2_r_V = ep.solve_joint(
                mod_1, mod_2, entry_2, syn_1, theta_syn, self.C_d_2)
            pol["b_1_star"][i] = b_1_star
            pol["b_2_r_V"][i] = b_2_r_V

        return pol["b_1_star"]

    def solve_Q(self, theta):
        _, rev2_d, _, entry_2, theta_syn = self._unpack_theta(theta)

        pol = self.data_manager.local_data.id_data["policies"]
        for i, ep in enumerate(self.problems):
            mod_2 = rev2_d[i] + self.eps_2[i]
            pol["b_2_r_Q"][i] = ep.solve_second_stage(
                self.obs_b[i], mod_2, entry_2, theta_syn, self.C_d_2)
