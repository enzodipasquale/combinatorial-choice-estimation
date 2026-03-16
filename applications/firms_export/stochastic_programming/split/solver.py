import sys
from pathlib import Path
import numpy as np
import gurobipy as gp
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
        self.rev_chars_2_d = id_data["rev_chars_2_d"]
        self.state_chars = id_data["state_chars"]
        self.entry_chars = item_data["entry_chars"]
        self.syn_chars = item_data["syn_chars"]
        self.syn_chars_2 = item_data["syn_chars_2"]
        self.beta_s = item_data["beta_s"]
        self.n_rev = self.rev_chars_1.shape[1]
        self.n_per_period = self.n_rev + 3
        obs_raw = id_data.get("obs_bundles", None)
        self.obs_b = obs_raw.astype(float) if obs_raw is not None else np.zeros((n, M))
        self.eps_1 = self.data_manager.local_data.errors["eps_1"]
        self.eps_2 = self.data_manager.local_data.errors["eps_2"]

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

    def solve(self, theta):
        k = self.n_per_period
        theta_rev1 = theta[:self.n_rev]
        theta_s1 = theta[self.n_rev]
        theta_sc1 = theta[self.n_rev + 1]
        theta_c1 = theta[self.n_rev + 2]
        theta_rev2 = theta[k:k + self.n_rev]
        theta_s2 = theta[k + self.n_rev]
        theta_sc2 = theta[k + self.n_rev + 1]
        theta_c2 = theta[k + self.n_rev + 2]

        rev1 = np.einsum('inm,n->im', self.rev_chars_1, theta_rev1)
        rev2_d = np.einsum('inm,n->im', self.rev_chars_2_d, theta_rev2)
        entry_1 = theta_s1 + theta_sc1 * self.entry_chars
        entry_2 = self.beta_s * (theta_s2 + theta_sc2 * self.entry_chars)
        syn_1 = theta_c1 * self.syn_chars
        syn_2 = theta_c2 * self.syn_chars_2

        pol = self.data_manager.local_data.id_data["policies"]
        for i, ep in enumerate(self.problems):
            mod_1 = rev1[i] + (1 - self.state_chars[i]) * entry_1 + self.eps_1[i]
            mod_2 = rev2_d[i] + self.eps_2[i]

            pol["b_1_star"][i] = ep.solve_joint(mod_1, mod_2, entry_2, syn_1, syn_2)
            pol["b_2_r_V"][i] = ep.solve_second_stage(
                pol["b_1_star"][i].astype(float), mod_2, entry_2, syn_2)
            pol["b_2_r_Q"][i] = ep.solve_second_stage(
                self.obs_b[i], mod_2, entry_2, syn_2)

        return pol["b_1_star"]
