import numpy as np
from scipy.special import ndtr
import gurobipy as gp
from combest.subproblems.solver_base import SubproblemSolver, create_gurobi_model

_SQRT_2PI_INV = 1.0 / np.sqrt(2 * np.pi)


def ev2_closed(mu, sigma):
    t = mu / sigma
    phi_t = _SQRT_2PI_INV * np.exp(-0.5 * t * t)
    return mu * ndtr(t) + sigma * phi_t


class TwoStageSolverCF(SubproblemSolver):

    def initialize(self):
        id_data = self.data_manager.local_data.id_data
        item_data = self.data_manager.local_data.item_data
        n = self.comm_manager.num_local_agent
        self.M = self.dimensions_cfg.n_items
        self.sigma_2 = item_data["sigma_2"]
        self.rev_chars_1 = id_data["rev_chars_1"]
        self.rev_chars_2_d = id_data["rev_chars_2_d"]
        self.state_chars = id_data["state_chars"]
        self.entry_chars = item_data["entry_chars"]
        self.entry_chars_2 = item_data["entry_chars_2"]
        self.syn_chars = item_data["syn_chars"]
        self.discount_2 = item_data["discount_2"]
        self.n_rev = self.rev_chars_1.shape[1]
        obs_raw = id_data.get("obs_bundles", None)
        self.obs_b = obs_raw.astype(float) if obs_raw is not None else np.zeros((n, self.M))
        self.eps_1 = self.data_manager.local_data.errors["eps_1"]
        self.C = self.syn_chars
        self.C_d = self.syn_chars * self.entry_chars[:, None]
        self.C_2 = item_data["C_2"]
        self.C_d_2 = item_data["C_d_2"]
        self.local_problems = []

        # precompute enumeration table (M <= 20)
        if self.M <= 20:
            self._all_b = ((np.arange(2**self.M)[:, None]
                            >> np.arange(self.M)[None, :]) & 1).astype(float)
            self._switch_all = 1.0 - self._all_b

        id_data["policies"] = {
            "b_1_star": np.zeros((n, self.M), dtype=bool),
            "mu_V": np.zeros((n, self.M)),
            "mu_Q": np.zeros((n, self.M)),
        }

    def _unpack_theta(self, theta):
        theta_rev = theta[:self.n_rev]
        theta_s = theta[self.n_rev]
        theta_sd = theta[self.n_rev + 1]
        theta_syn = theta[self.n_rev + 2]
        theta_syn_d = theta[self.n_rev + 3]

        rev1 = np.einsum('inm,n->im', self.rev_chars_1, theta_rev)
        rev2_d = np.einsum('inm,n->im', self.rev_chars_2_d, theta_rev)
        entry = theta_s + theta_sd * self.entry_chars
        entry_2 = self.discount_2 * theta_s + theta_sd * self.entry_chars_2

        C_syn = theta_syn * self.C + theta_syn_d * self.C_d
        C_syn_2 = theta_syn * self.C_2 + theta_syn_d * self.C_d_2

        return rev1, rev2_d, entry, entry_2, C_syn, C_syn_2

    def _compute_mu(self, b_1, rev2_d_i, entry_2, C_syn_2):
        b = b_1.astype(float)
        switch = 1.0 - b
        return rev2_d_i + switch * entry_2 + switch * (C_syn_2 @ b)

    def _objective(self, b_1, c_1, rev2_d_i, entry_2, C_syn_2):
        mu = self._compute_mu(b_1, rev2_d_i, entry_2, C_syn_2)
        return c_1 @ b_1.astype(float) + ev2_closed(mu, self.sigma_2).sum()

    def _solve_b1_enum(self, c_1, rev2_d_i, base_all):
        obj_1 = self._all_b @ c_1
        mu = rev2_d_i + base_all
        obj_2 = ev2_closed(mu, self.sigma_2).sum(axis=1)
        best = np.argmax(obj_1 + obj_2)
        return self._all_b[best] > 0.5

    def _solve_b1_mip(self, c_1, rev2_d_i, entry_2, C_syn_2):
        M = self.M
        sigma = self.sigma_2
        m = create_gurobi_model(self.subproblem_cfg)
        m.Params.Threads = 1

        b_1 = m.addMVar(M, vtype=gp.GRB.BINARY, name='b_1')
        base = m.addMVar(M, lb=-gp.GRB.INFINITY, name='base')
        w = m.addMVar(M, lb=-gp.GRB.INFINITY, name='w')
        mu = m.addMVar(M, lb=-gp.GRB.INFINITY, name='mu')
        z = m.addMVar(M, lb=-gp.GRB.INFINITY, name='z')

        # base = entry_2 + C_syn_2 @ b_1
        m.addConstr(base == entry_2 + C_syn_2 @ b_1)

        # McCormick: w = (1 - b_1) * base
        L = entry_2 + np.minimum(C_syn_2, 0).sum(1)
        U = entry_2 + np.maximum(C_syn_2, 0).sum(1)
        for j in range(M):
            m.addConstr(w[j] <= U[j] * (1 - b_1[j]))
            m.addConstr(w[j] >= L[j] * (1 - b_1[j]))
            m.addConstr(w[j] <= base[j] - L[j] * b_1[j])
            m.addConstr(w[j] >= base[j] - U[j] * b_1[j])

        m.addConstr(mu == rev2_d_i + w)

        # PWL breakpoints (adaptive range)
        lo = float(rev2_d_i.min() + min(L.min(), 0)) - 3 * sigma
        hi = float(rev2_d_i.max() + max(U.max(), 0)) + 3 * sigma
        bp_x = np.linspace(lo, hi, 300)
        bp_y = ev2_closed(bp_x, sigma)
        for j in range(M):
            m.addGenConstrPWL(
                mu[j].item(), z[j].item(),
                bp_x.tolist(), bp_y.tolist(), f'pwl_{j}')

        m.setObjective(c_1 @ b_1 + z.sum(), gp.GRB.MAXIMIZE)
        m.optimize()

        if m.Status != gp.GRB.OPTIMAL:
            return np.zeros(M, dtype=bool)

        return np.array(b_1.X) > 0.5

    def solve(self, theta):
        rev1, rev2_d, entry, entry_2, C_syn, C_syn_2 = self._unpack_theta(theta)
        pol = self.data_manager.local_data.id_data["policies"]

        # precompute shared part for enumeration: (2^M, M)
        if self.M <= 20:
            syn_all = self._all_b @ C_syn_2.T
            base_all = self._switch_all * (entry_2 + syn_all)

        for i in range(len(pol["b_1_star"])):
            b_0 = self.state_chars[i]
            switch_1 = 1 - b_0
            syn_1 = switch_1 * (C_syn @ b_0)
            mod_1 = rev1[i] + switch_1 * entry + self.eps_1[i]
            c_1 = mod_1 + syn_1

            if self.M <= 20:
                b_1_star = self._solve_b1_enum(c_1, rev2_d[i], base_all)
            else:
                b_1_star = self._solve_b1_mip(c_1, rev2_d[i], entry_2, C_syn_2)
            pol["b_1_star"][i] = b_1_star
            pol["mu_V"][i] = self._compute_mu(b_1_star, rev2_d[i], entry_2, C_syn_2)

        return pol["b_1_star"]

    def solve_Q(self, theta):
        _, rev2_d, _, entry_2, _, C_syn_2 = self._unpack_theta(theta)
        pol = self.data_manager.local_data.id_data["policies"]

        for i in range(len(pol["mu_Q"])):
            pol["mu_Q"][i] = self._compute_mu(
                self.obs_b[i], rev2_d[i], entry_2, C_syn_2)
