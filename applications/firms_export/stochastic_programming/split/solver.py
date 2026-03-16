import sys
from pathlib import Path
import numpy as np
from combest.subproblems.solver_base import SubproblemSolver

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "baseline"))
from solver import EntryProblem


class TwoStageSolverSplit(SubproblemSolver):

    def initialize(self):
        id_data = self.data_manager.local_data.id_data
        item_data = self.data_manager.local_data.item_data
        M = self.dimensions_cfg.n_items
        n = self.comm_manager.num_local_agent
        self.M = M
        self.R = item_data["R"]
        self.rev_chars_1 = id_data["rev_chars_1"]
        self.rev_chars_2 = id_data["rev_chars_2"]
        self.state_chars = id_data["state_chars"]
        self.entry_chars = item_data["entry_chars"]
        self.syn_chars = item_data["syn_chars"]
        self.n_rev = self.rev_chars_1.shape[1]
        self.n_per_period = self.n_rev + 3
        obs_raw = id_data.get("obs_bundles", None)
        self.obs_b = obs_raw.astype(float) if obs_raw is not None else np.zeros((n, M))
        self.eps_1 = self.data_manager.local_data.errors["eps_1"]
        self.eps_2 = self.data_manager.local_data.errors["eps_2"]
        self.C_d = self.syn_chars * self.entry_chars[:, None]

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
        k = self.n_per_period
        theta_rev_1 = theta[:self.n_rev]
        theta_s_1 = theta[self.n_rev]
        theta_sd_1 = theta[self.n_rev + 1]
        theta_syn_1 = theta[self.n_rev + 2]
        theta_rev_2 = theta[k:k + self.n_rev]
        theta_s_2 = theta[k + self.n_rev]
        theta_sd_2 = theta[k + self.n_rev + 1]
        theta_syn_2 = theta[k + self.n_rev + 2]

        rev1 = np.einsum('inm,n->im', self.rev_chars_1, theta_rev_1)
        rev2 = np.einsum('inm,n->im', self.rev_chars_2, theta_rev_2)
        entry = theta_s_1 + theta_sd_1 * self.entry_chars
        entry_2 = theta_s_2 + theta_sd_2 * self.entry_chars

        return rev1, rev2, entry, entry_2, theta_syn_1, theta_syn_2

    def solve(self, theta):
        rev1, rev2, entry, entry_2, theta_syn_1, theta_syn_2 = \
            self._unpack_theta(theta)

        pol = self.data_manager.local_data.id_data["policies"]
        for i, ep in enumerate(self.problems):
            b_0 = self.state_chars[i]
            switch_1 = 1 - b_0
            syn_1 = theta_syn_1 * switch_1 * (self.C_d @ b_0)
            mod_1 = rev1[i] + switch_1 * entry + self.eps_1[i]
            mod_2 = rev2[i] + self.eps_2[i]

            b_1_star, b_2_r_V = ep.solve_joint(
                mod_1, mod_2, entry_2, syn_1, theta_syn_2, self.C_d)
            pol["b_1_star"][i] = b_1_star
            pol["b_2_r_V"][i] = b_2_r_V

        return pol["b_1_star"]

    def solve_Q(self, theta):
        _, rev2, _, entry_2, _, theta_syn_2 = self._unpack_theta(theta)

        pol = self.data_manager.local_data.id_data["policies"]
        for i, ep in enumerate(self.problems):
            mod_2 = rev2[i] + self.eps_2[i]
            pol["b_2_r_Q"][i] = ep.solve_second_stage(
                self.obs_b[i], mod_2, entry_2, theta_syn_2, self.C_d)
