import numpy as np
import gurobipy as gp
from combest.subproblems.solver_base import SubproblemSolver, create_gurobi_model


class EntryProblem:

    def __init__(self, M, R, capacity, solver_cfg):
        self.M, self.R = M, R

        # Joint model: b_1 (M,) + b_2_r (R, M)
        self.model = create_gurobi_model(solver_cfg)
        self.b_1 = self.model.addMVar(M, vtype=gp.GRB.BINARY, name='b_1')
        self.b_2_r = self.model.addMVar((R, M), vtype=gp.GRB.BINARY, name='b_2_r')
        self.model.addConstr(self.b_1.sum() <= capacity)
        for r in range(R):
            self.model.addConstr(self.b_2_r[r, :].sum() <= capacity)
        self.model.update()

        # Q model: single b_2 (M,) for per-scenario re-optimization
        self.q_model = create_gurobi_model(solver_cfg)
        self.b_2 = self.q_model.addMVar(M, vtype=gp.GRB.BINARY, name='b_2')
        self.q_model.addConstr(self.b_2.sum() <= capacity)
        self.q_model.update()

    def solve_joint(self, mod_1, mod_2, entry_2, syn_1, syn_2):
        b_1, b_2_r = self.b_1, self.b_2_r

        # Period 1
        obj = mod_1 @ b_1 + b_1 @ syn_1 @ b_1

        # Period 2 (vectorized over R scenarios)
        mod_2_full = mod_2 + entry_2[None, :]                  # (R, M)
        obj += (1 / self.R) * (
            (mod_2_full * b_2_r).sum()                          # linear in b_2
            - (b_2_r @ (entry_2 * b_1)).sum()
            + (b_2_r @ syn_2 * b_2_r).sum())                   # quadratic synergy

        self.model.setObjective(obj, gp.GRB.MAXIMIZE)
        self.model.optimize()
        return np.array(self.b_1.X) > 0.5

    def solve_second_stage(self, b_1_fixed, mod_2, entry_2, syn_2):
        b_2_r_out = np.zeros((self.R, self.M), dtype=bool)
        for r in range(self.R):
            c = mod_2[r] + (1 - b_1_fixed) * entry_2
            self.q_model.setObjective(
                c @ self.b_2 + self.b_2 @ syn_2 @ self.b_2,
                gp.GRB.MAXIMIZE)
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
        self.entry_chars = item_data["entry_chars"]        # (M,)
        self.syn_chars = item_data["syn_chars"]            # (M, M)
        self.syn_chars_2 = item_data["syn_chars_2"]       # (M, M) pre-discounted
        self.beta_s = item_data["beta_s"]                  # scalar
        self.n_rev = self.rev_chars_1.shape[1]
        self.cap = id_data["capacity"]
        obs_raw = id_data.get("obs_bundles", None)
        self.obs_b = obs_raw.astype(float) if obs_raw is not None else np.zeros((n, M))
        self.eps_1 = self.data_manager.local_data.errors["eps_1"]   # (n, M)
        self.eps_2 = self.data_manager.local_data.errors["eps_2"]   # (n, R, M)

        # One EntryProblem per agent
        self.problems = []
        self.local_problems = []
        for i in range(n):
            ep = EntryProblem(M, self.R, self.cap[i], self.subproblem_cfg)
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
        theta_sc = theta[self.n_rev + 1]
        theta_c = theta[self.n_rev + 2]

        rev1 = np.einsum('inm,n->im', self.rev_chars_1, theta_rev)     # (n, M)
        rev2_d = np.einsum('inm,n->im', self.rev_chars_2_d, theta_rev) # (n, M)
        entry = theta_s + theta_sc * self.entry_chars     # (M,) per-item entry cost
        entry_2 = self.beta_s * entry                     # (M,) discounted
        syn_1 = theta_c * self.syn_chars                  # (M, M)
        syn_2 = theta_c * self.syn_chars_2                # (M, M) already discounted

        pol = self.data_manager.local_data.id_data["policies"]
        for i, ep in enumerate(self.problems):
            mod_1 = rev1[i] + (1 - self.state_chars[i]) * entry + self.eps_1[i]
            mod_2 = rev2_d[i] + self.eps_2[i]             # (R, M) via broadcasting

            pol["b_1_star"][i] = ep.solve_joint(mod_1, mod_2, entry_2, syn_1, syn_2)
            pol["b_2_r_V"][i] = ep.solve_second_stage(
                pol["b_1_star"][i].astype(float), mod_2, entry_2, syn_2)
            pol["b_2_r_Q"][i] = ep.solve_second_stage(
                self.obs_b[i], mod_2, entry_2, syn_2)

        return pol["b_1_star"]
