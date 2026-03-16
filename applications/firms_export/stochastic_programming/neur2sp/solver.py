import sys
from pathlib import Path
import numpy as np
import torch
import gurobipy as gp
from combest.subproblems.solver_base import SubproblemSolver, create_gurobi_model

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "baseline"))
from solver import EntryProblem
from neur2sp.net2mip import compute_layer_bounds, embed_relu


class TwoStageSolverNN(SubproblemSolver):

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
        obs_raw = id_data.get("obs_bundles", None)
        self.obs_b = obs_raw.astype(float) if obs_raw is not None else np.zeros((n, M))
        self.eps_1 = self.data_manager.local_data.errors["eps_1"]
        self.eps_2 = self.data_manager.local_data.errors["eps_2"]
        self.eps_2_perm = self.data_manager.local_data.errors["eps_2_perm"]

        ckpt = torch.load(item_data["nn_model_path"],
                          map_location="cpu", weights_only=False)
        self.nn_weights = ckpt["weights"]
        self.nn_biases = ckpt["biases"]
        input_lb = ckpt["input_lb"]
        input_ub = ckpt["input_ub"]
        self.pre_bounds = compute_layer_bounds(
            self.nn_weights, self.nn_biases, input_lb, input_ub)

        n_ctx = M + 3
        self.surrogate_models, self.b_1_vars = [], []
        self.eff_rev_vars, self.theta_vars, self.nn_out_vars = [], [], []
        for i in range(n):
            m = create_gurobi_model(self.subproblem_cfg)
            b_1 = m.addMVar(M, vtype=gp.GRB.BINARY, name="b_1")
            eff_vars = [m.addVar(lb=-1e6, ub=1e6, name=f"eff_{j}")
                        for j in range(M)]
            tvars = [m.addVar(lb=-100, ub=100, name=f"th_{t}")
                     for t in range(3)]
            m.update()
            nn_in = list(b_1.tolist()) + eff_vars + tvars
            nn_out = embed_relu(m, self.nn_weights, self.nn_biases,
                                nn_in, self.pre_bounds)
            self.surrogate_models.append(m)
            self.b_1_vars.append(b_1)
            self.eff_rev_vars.append(eff_vars)
            self.theta_vars.append(tvars)
            self.nn_out_vars.append(nn_out)

        self.problems = [EntryProblem(M, self.R, self.subproblem_cfg)
                         for _ in range(n)]

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

        rev1 = np.einsum('inm,n->im', self.rev_chars_1, theta_rev)
        rev2_d = np.einsum('inm,n->im', self.rev_chars_2_d, theta_rev)
        entry = theta_s + theta_sc * self.entry_chars
        entry_2 = self.beta_s * entry
        syn_1 = theta_c * self.syn_chars
        syn_2 = theta_c * self.syn_chars_2

        pol = self.data_manager.local_data.id_data["policies"]
        for i, ep in enumerate(self.problems):
            eff_rev = rev2_d[i] + self.eps_2_perm[i]

            for j in range(self.M):
                self.eff_rev_vars[i][j].lb = float(eff_rev[j])
                self.eff_rev_vars[i][j].ub = float(eff_rev[j])
            self.theta_vars[i][0].lb = self.theta_vars[i][0].ub = float(theta_s)
            self.theta_vars[i][1].lb = self.theta_vars[i][1].ub = float(theta_sc)
            self.theta_vars[i][2].lb = self.theta_vars[i][2].ub = float(theta_c)

            mod_1 = rev1[i] + (1 - self.state_chars[i]) * entry + self.eps_1[i]
            b_1 = self.b_1_vars[i]
            obj = mod_1 @ b_1 + b_1 @ syn_1 @ b_1 + self.nn_out_vars[i]
            self.surrogate_models[i].setObjective(obj, gp.GRB.MAXIMIZE)
            self.surrogate_models[i].optimize()
            pol["b_1_star"][i] = np.array(b_1.X) > 0.5

            mod_2 = rev2_d[i] + self.eps_2[i]
            pol["b_2_r_V"][i] = ep.solve_second_stage(
                pol["b_1_star"][i].astype(float), mod_2, entry_2, syn_2)
            pol["b_2_r_Q"][i] = ep.solve_second_stage(
                self.obs_b[i], mod_2, entry_2, syn_2)

        return pol["b_1_star"]

    def update_solver_settings(self, settings_dict):
        for m in self.surrogate_models:
            for param, value in settings_dict.items():
                m.setParam(param, value)
        for ep in self.problems:
            for param, value in settings_dict.items():
                ep.q_model.setParam(param, value)
