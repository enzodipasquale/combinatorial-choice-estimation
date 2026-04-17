"""MIQP subproblem solver for the portfolio-choice scenario."""

import numpy as np
import gurobipy as gp
from combest.subproblems.solver_base import SubproblemSolver, create_gurobi_model


class PortfolioSolver(SubproblemSolver):
    """Solves the portfolio MIQP for each agent.

    Agent i's problem:
        max_{s, w}  sum_j w_j (X_ij @ beta) - (gamma/2) w^T Sigma w - kappa sum_j s_j
        s.t.  sum_j w_j = 1,  w_j <= s_j,  s_j in {0,1},  w_j >= 0

    theta = [beta_0, beta_1, beta_2, gamma, kappa]  (length 5)

    Returns (n_local, 2*M) float array: first M columns are s, last M are w.
    """

    def initialize(self):
        M = self.dimensions_cfg.n_items // 2  # n_items = 2*M in the config
        self._M = M
        X_agents = self.data_manager.local_data.id_data['X_agents']  # (n_local, M, K)
        Sigma = self.data_manager.local_data.item_data['Sigma']       # (M, M)

        self.local_problems = []
        self._s_vars = []
        self._w_vars = []

        for local_id in range(self.comm_manager.num_local_agent):
            model = create_gurobi_model(self.subproblem_cfg)

            s = model.addMVar(M, vtype=gp.GRB.BINARY, name='s')
            w = model.addMVar(M, lb=0.0, ub=1.0, name='w')

            # Budget constraint: sum(w) = 1
            model.addConstr(w.sum() == 1, name='budget')

            # Linking: w_j <= s_j
            model.addConstr(w <= s, name='link')

            model.update()
            self.local_problems.append(model)
            self._s_vars.append(s)
            self._w_vars.append(w)

        self._X_agents = X_agents
        self._Sigma = Sigma

    def solve(self, theta):
        M = self._M
        K = self._X_agents.shape[2]

        beta = theta[:K]       # (K,)
        gamma = theta[K]       # scalar
        kappa = theta[K + 1]   # scalar

        n_local = len(self.local_problems)
        results = np.zeros((n_local, 2 * M), dtype=float)

        for i, model in enumerate(self.local_problems):
            X_i = self._X_agents[i]  # (M, K)
            mu_i = X_i @ beta        # (M,) expected returns

            s = self._s_vars[i]
            w = self._w_vars[i]

            # Objective: sum_j w_j mu_ij - (gamma/2) w^T Sigma w - kappa sum_j s_j
            # Linear part: mu_i^T w - kappa 1^T s
            # Quadratic part: -(gamma/2) w^T Sigma w
            lin_w = mu_i
            lin_s = -kappa * np.ones(M)

            Q = -0.5 * gamma * self._Sigma  # (M, M) quadratic coefficient for w

            # Build full objective: maximize lin_s^T s + lin_w^T w + w^T Q w
            # Gurobi MVar API: setMObjective(Q_mat, c_vec, constant, ...)
            # We need to work with the full variable vector [s, w]
            n_vars = 2 * M
            Q_full = np.zeros((n_vars, n_vars))
            Q_full[M:, M:] = Q  # quadratic term only involves w

            c_full = np.concatenate([lin_s, lin_w])

            all_vars = gp.MVar.fromlist(list(s.tolist()) + list(w.tolist()))
            model.setMObjective(
                Q=Q_full, c=c_full, constant=0.0,
                xQ_L=all_vars, xQ_R=all_vars, xc=all_vars,
                sense=gp.GRB.MAXIMIZE
            )
            model.optimize()

            if model.status == gp.GRB.OPTIMAL:
                s_val = np.array([v.X for v in s.tolist()])
                w_val = np.array([v.X for v in w.tolist()])
                results[i, :M] = np.round(s_val).astype(float)  # clean binary
                results[i, M:] = w_val
            else:
                raise ValueError(
                    f"Portfolio subproblem infeasible/unbounded at agent {i}, "
                    f"status={model.status}"
                )

        return results
