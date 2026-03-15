import numpy as np
import gurobipy as gp
from combest.subproblems.solver_base import SubproblemSolver, create_gurobi_model


class EntryProblem:

    def __init__(self, M, R, solver_cfg):
        self.M, self.R = M, R

        # Joint model: b_1 (M,) + b_2_r (R, M)
        self.model = create_gurobi_model(solver_cfg)
        self.b_1 = self.model.addMVar(M, vtype=gp.GRB.BINARY, name='b_1')
        self.b_2_r = self.model.addMVar((R, M), vtype=gp.GRB.BINARY, name='b_2_r')
        self.model.update()

        # Q model: single b_2 (M,) for per-scenario re-optimization
        self.q_model = create_gurobi_model(solver_cfg)
        self.b_2 = self.q_model.addMVar(M, vtype=gp.GRB.BINARY, name='b_2')
        self.q_model.update()

    def solve_joint(self, mod_1, mod_2, entry_2, syn_1, syn_coeff_2, C_d):
        """
        mod_1: (M,)  period-1 modular utility per item (rev + entry + errors)
        mod_2: (R, M) period-2 modular utility (rev + errors)
        entry_2: (M,) discounted entry cost vector (theta_s + theta_sd * d_j)
        syn_1: (M,) period-1 synergy for each item: theta_syn * d_j * Σ_k b0_k * C_jk
        syn_coeff_2: scalar  beta * theta_syn
        C_d: (M, M)  C_jk * d_j matrix for period-2 synergy computation
        """
        b_1, b_2_r = self.b_1, self.b_2_r

        # Period 1: modular + synergy (syn_1 is data, linear in b_1)
        obj = (mod_1 + syn_1) @ b_1

        # Period 2 (vectorized over R scenarios)
        # Entry cost: (1 - b_1_j) * b_2rj * entry_2_j
        #           = entry_2 @ b_2_r - entry_2 * b_1 @ b_2_r (bilinear)
        mod_2_full = mod_2 + entry_2[None, :]                  # (R, M)
        obj += (1 / self.R) * (
            (mod_2_full * b_2_r).sum()
            - (b_2_r @ (entry_2 * b_1)).sum())

        # Period-2 synergy: (1 - b_1_j) * b_2rj * d_j * Σ_k b_1_k * C_jk
        # = b_2rj * d_j * Σ_k b_1_k * C_jk  -  b_1_j * b_2rj * d_j * Σ_k b_1_k * C_jk
        # First term: b_2_r @ C_d @ b_1 (bilinear)
        # Second term: cubic → approximate by dropping it (small: only nonzero
        #   when b_1_j=1 AND b_2rj=1, but b_2rj is new entry so b_1_j=0 typically)
        # So: syn_2 ≈ syn_coeff_2 * (1/R) * Σ_r b_2_r @ C_d @ b_1
        obj += syn_coeff_2 * (1 / self.R) * (b_2_r @ C_d @ b_1).sum()

        self.model.setObjective(obj, gp.GRB.MAXIMIZE)
        self.model.optimize()
        return np.array(self.b_1.X) > 0.5

    def solve_second_stage(self, b_1_fixed, mod_2, entry_2, syn_coeff_2, C_d):
        """Solve period-2 given fixed b_1."""
        b_2_r_out = np.zeros((self.R, self.M), dtype=bool)
        # Synergy in period 2: d_j * Σ_k b_1_k * C_jk
        # With b_1 fixed, this is a known vector
        syn_2 = syn_coeff_2 * (1 - b_1_fixed) * (C_d @ b_1_fixed)  # (M,)
        for r in range(self.R):
            c = mod_2[r] + (1 - b_1_fixed) * entry_2 + syn_2
            self.q_model.setObjective(c @ self.b_2, gp.GRB.MAXIMIZE)
            self.q_model.optimize()
            b_2_r_out[r] = np.array(self.b_2.X) > 0.5
        return b_2_r_out


class TwoStageSolver(SubproblemSolver):

    def initialize(self):
        id_data = self.data_manager.local_data.id_data
        item_data = self.data_manager.local_data.item_data
        M = self.dimensions_cfg.n_items
        n = self.comm_manager.num_local_agent
        self.M = M
        self.R = item_data["R"]
        self.rev_chars_1 = id_data["rev_chars_1"]         # (n, n_rev, M)
        self.rev_chars_2_d = id_data["rev_chars_2_d"]     # (n, n_rev, M) pre-discounted
        self.state_chars = id_data["state_chars"]          # (n, M)
        self.entry_chars = item_data["entry_chars"]        # (M,) d_j thousands km
        self.syn_chars = item_data["syn_chars"]            # (M, M) exp(-dist_jk), zero diag
        self.beta_s = item_data["beta_s"]                  # scalar
        self.n_rev = self.rev_chars_1.shape[1]
        obs_raw = id_data.get("obs_bundles", None)
        self.obs_b = obs_raw.astype(float) if obs_raw is not None else np.zeros((n, M))
        self.eps_1 = self.data_manager.local_data.errors["eps_1"]   # (n, M)
        self.eps_2 = self.data_manager.local_data.errors["eps_2"]   # (n, R, M)

        # Pre-compute C_d: C_jk * d_j  (M, M)
        self.C_d = self.syn_chars * self.entry_chars[:, None]  # (M, M)

        # One EntryProblem per agent
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
        theta_rev = theta[:self.n_rev]
        theta_s = theta[self.n_rev]
        theta_sd = theta[self.n_rev + 1]
        theta_syn = theta[self.n_rev + 2]

        rev1 = np.einsum('inm,n->im', self.rev_chars_1, theta_rev)     # (n, M)
        rev2_d = np.einsum('inm,n->im', self.rev_chars_2_d, theta_rev) # (n, M)
        entry = theta_s + theta_sd * self.entry_chars     # (M,) entry cost per item
        entry_2 = self.beta_s * entry                     # (M,) discounted
        syn_coeff_2 = self.beta_s * theta_syn             # scalar

        pol = self.data_manager.local_data.id_data["policies"]
        for i, ep in enumerate(self.problems):
            b_0 = self.state_chars[i]
            switch_1 = 1 - b_0                             # (M,)

            # Period-1 synergy: theta_syn * switch_1_j * d_j * Σ_k b0_k * C_jk
            syn_1 = theta_syn * switch_1 * (self.C_d @ b_0)  # (M,)

            mod_1 = rev1[i] + switch_1 * entry + self.eps_1[i]
            mod_2 = rev2_d[i] + self.eps_2[i]             # (R, M)

            pol["b_1_star"][i] = ep.solve_joint(
                mod_1, mod_2, entry_2, syn_1, syn_coeff_2, self.C_d)
            pol["b_2_r_V"][i] = ep.solve_second_stage(
                pol["b_1_star"][i].astype(float), mod_2, entry_2,
                syn_coeff_2, self.C_d)
            pol["b_2_r_Q"][i] = ep.solve_second_stage(
                self.obs_b[i], mod_2, entry_2,
                syn_coeff_2, self.C_d)

        return pol["b_1_star"]
