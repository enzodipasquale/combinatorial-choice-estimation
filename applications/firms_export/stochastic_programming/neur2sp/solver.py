"""Surrogate two-stage solver using an embedded ReLU neural network.

Drop-in replacement for ``solver.TwoStageSolver``.  The period-2 expected
value is approximated by a trained ReLU network whose forward pass is
encoded as MIP constraints (big-M formulation).  The surrogate MIP
optimises b_1 jointly with the NN auxiliary variables, then recovers
b_2_r via independent knapsacks (same as the original solver).

The solver reads the NN weights from a file whose path is stored in
``item_data["nn_model_path"]``.
"""
import numpy as np
import torch
import gurobipy as gp
from combest.subproblems.solver_base import SubproblemSolver, GurobiMixin
from neur2sp.net2mip import compute_layer_bounds, embed_relu


class TwoStageSolverNN(GurobiMixin, SubproblemSolver):

    # ── one-time setup ──────────────────────────────────────────────
    def initialize(self):
        id_data = self.data_manager.local_data.id_data
        item_data = self.data_manager.local_data.item_data
        M = self.dimensions_cfg.n_items
        n = self.comm_manager.num_local_agent
        self.M = M
        self.R = item_data["R"]
        self.beta = item_data["beta"]
        self.rev_chars_1 = item_data["rev_chars_1"]
        self.rev_chars_2 = item_data["rev_chars_2"]
        self.state_chars = id_data["state_chars"]
        self.syn_chars = item_data["syn_chars"]
        self.n_rev = self.rev_chars_1.shape[0]
        self.cap = id_data["capacity"]
        obs_raw = id_data.get("obs_bundles", None)
        self.obs_b = (obs_raw.astype(float) if obs_raw is not None
                      else np.zeros((n, M)))
        self.eps1 = self.data_manager.local_data.errors["eps1"]
        self.eps2 = self.data_manager.local_data.errors["eps2"]

        # ── load trained NN ─────────────────────────────────────────
        nn_path = item_data["nn_model_path"]
        ckpt = torch.load(nn_path, map_location="cpu", weights_only=False)
        self.nn_weights = ckpt["weights"]      # list of ndarray
        self.nn_biases = ckpt["biases"]        # list of ndarray
        theta_lb = ckpt["theta_lb"]
        theta_ub = ckpt["theta_ub"]

        # input bounds for big-M: [b_1 in {0,1}^M, theta in bounds]
        input_lb = np.concatenate([np.zeros(M), theta_lb])
        input_ub = np.concatenate([np.ones(M), theta_ub])
        self.pre_bounds = compute_layer_bounds(
            self.nn_weights, self.nn_biases, input_lb, input_ub)

        # ── build surrogate Gurobi models (one per agent) ───────────
        n_cov = self.n_rev + 2
        self.local_problems = []
        self.b_1_vars = []
        self.theta_vars = []
        self.nn_out_vars = []

        for i in range(n):
            m = self._create_gurobi_model()
            b_1 = m.addMVar(M, vtype=gp.GRB.BINARY, name="b_1")
            m.addConstr(b_1.sum() <= self.cap[i], name="cap")

            # theta as fixed variables (bounds updated each solve())
            tvars = [m.addVar(lb=-100, ub=100, name=f"th_{t}")
                     for t in range(n_cov)]
            m.update()

            # embed NN: input = [b_1, theta]
            input_vars = list(b_1.tolist()) + tvars
            nn_out = embed_relu(m, self.nn_weights, self.nn_biases,
                                input_vars, self.pre_bounds)

            self.local_problems.append(m)
            self.b_1_vars.append(b_1)
            self.theta_vars.append(tvars)
            self.nn_out_vars.append(nn_out)

        # ── q-models for recovering b_2_r (same as original solver) ─
        self.q_models, self.q_vars = [], []
        for i in range(n):
            m = self._create_gurobi_model()
            b_2 = m.addMVar(M, vtype=gp.GRB.BINARY, name="b_2")
            m.addConstr(b_2.sum() <= self.cap[i])
            m.update()
            self.q_models.append(m)
            self.q_vars.append(b_2)

        id_data["policies"] = {
            "b_1_star": np.zeros((n, M), dtype=bool),
            "b_2_r_V": np.zeros((n, self.R, M), dtype=bool),
            "b_2_r_Q": np.zeros((n, self.R, M), dtype=bool),
        }

    # ── called every outer iteration with a new theta ───────────────
    def solve(self, theta):
        theta_rev = theta[:self.n_rev]
        theta_s = theta[self.n_rev]
        theta_c = theta[self.n_rev + 1]
        C = self.syn_chars
        rev2 = self.rev_chars_2.T @ theta_rev

        pol = self.data_manager.local_data.id_data["policies"]

        for i, model in enumerate(self.local_problems):
            b_1 = self.b_1_vars[i]
            b_0_i = self.state_chars[i]

            # ── fix theta variables to current values ───────────────
            for t in range(len(theta)):
                self.theta_vars[i][t].lb = float(theta[t])
                self.theta_vars[i][t].ub = float(theta[t])

            # ── first-stage objective (explicit) + NN output ────────
            rev1 = self.rev_chars_1.T @ theta_rev
            mod_1 = rev1 + (1 - b_0_i) * theta_s + self.eps1[i]
            obj = mod_1 @ b_1 + theta_c * (b_1 @ C @ b_1)
            obj += self.nn_out_vars[i]

            model.setObjective(obj, gp.GRB.MAXIMIZE)
            model.optimize()

            pol["b_1_star"][i] = np.array(b_1.X) > 0.5
            b_1_star_f = pol["b_1_star"][i].astype(float)

            # ── recover b_2_r given b_1_star (V-model) ─────────────
            b_2 = self.q_vars[i]
            for r in range(self.R):
                c_v = rev2 + (1 - b_1_star_f) * theta_s + self.eps2[i, r]
                self.q_models[i].setObjective(
                    c_v @ b_2 + theta_c * (b_2 @ C @ b_2),
                    gp.GRB.MAXIMIZE)
                self.q_models[i].optimize()
                pol["b_2_r_V"][i, r] = np.array(b_2.X) > 0.5

            # ── recover b_2_r given obs_b (Q-model) ────────────────
            for r in range(self.R):
                c_q = rev2 + (1 - self.obs_b[i]) * theta_s + self.eps2[i, r]
                self.q_models[i].setObjective(
                    c_q @ b_2 + theta_c * (b_2 @ C @ b_2),
                    gp.GRB.MAXIMIZE)
                self.q_models[i].optimize()
                pol["b_2_r_Q"][i, r] = np.array(b_2.X) > 0.5

        return pol["b_1_star"]
