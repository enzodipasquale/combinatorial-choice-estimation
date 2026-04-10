import numpy as np
import gurobipy as gp
from combest.subproblems.solver_base import SubproblemSolver, create_gurobi_model
from costs import compute_rev_factor, compute_facility_costs
from milp import build_milp_shell


THETA_NAMES = [
    'FE_1_As', 'FE_1_Eu', 'FE_2_As', 'FE_2_Eu',
    'delta_1_Am', 'delta_1_As', 'delta_1_Eu',
    'delta_2_Am', 'delta_2_As', 'delta_2_Eu',
    'rho_HQ_1', 'rho_HQ_2', 'rho_xi_1', 'rho_xi_2',
]


def unpack_theta(theta):
    return dict(zip(THETA_NAMES, theta))


def pack_theta(theta_dict):
    return np.array([theta_dict[k] for k in THETA_NAMES])


class MultiStageSolver(SubproblemSolver):

    def initialize(self):
        id_data = self.data_manager.local_data.id_data
        item_data = self.data_manager.local_data.item_data
        n = self.comm_manager.num_local_agent

        self.geo = item_data['geo']
        self.coefs = item_data['sourcing_coefs']
        all_firms = id_data['firms']
        self.obs_ids = self.comm_manager.obs_ids
        self.firms = [all_firms[oid] for oid in self.obs_ids]

        ng_max = item_data['ng_max']
        P_max = item_data['P_max']
        nm_max = item_data['nm_max']
        L1, L2, N = self.geo['L1'], self.geo['L2'], self.geo['n_markets']
        self.ng_max, self.P_max, self.nm_max = ng_max, P_max, nm_max
        self.L1, self.L2, self.N = L1, L2, N
        self.y1_size = ng_max * L1
        self.y2_size = P_max * L2
        self.z_size = nm_max * N
        self._eta_m1_kappa = (self.coefs['eta'] - 1) / abs(self.coefs['beta_2_T'])
        self._beta_2_phi = self.coefs['beta_2_phi']

        self.phi1 = self.data_manager.local_data.errors['phi1']
        self.phi2 = self.data_manager.local_data.errors['phi2']
        self.nu = self.data_manager.local_data.errors['nu']

        self._cont1_masks = [self.geo['cont1'] == c for c in (1, 2)]
        self._cont2_masks = [self.geo['cont2'] == c for c in (1, 2)]

        # Padded shares per firm (indexed by obs_id, not agent)
        n_obs = len(all_firms)
        self._shares_pad = np.zeros((n_obs, nm_max, N))
        for f, firm in enumerate(all_firms):
            nm = firm['n_models']
            self._shares_pad[f, :nm] = firm['shares']

        id_data['policies'] = {
            'x_V': np.zeros((n, nm_max, N, L1, L2)),
            'x_Q': id_data.get('x_Q', np.zeros((n, nm_max, N, L1, L2))),
            'lin_V': {},
            'lin_Q': {},
        }

        # Build one MILP per UNIQUE firm, reuse across simulations
        unique_obs = np.unique(self.obs_ids)
        self._firm_models = {}
        for oid in unique_obs:
            firm = all_firms[oid]
            ng = len(firm['ln_xi_1'])
            P, nm = firm['n_platforms'], firm['n_models']

            cg = np.zeros(nm_max, dtype=int)
            cg[:nm] = firm['cell_groups']
            pl = np.zeros(nm_max, dtype=int)
            pl[:nm] = firm['platforms']
            feas = np.zeros((nm_max, N), dtype=bool)
            feas[:nm] = firm['feasible']

            mdl = create_gurobi_model(self.subproblem_cfg)
            y1, y2, z, x = build_milp_shell(
                mdl, ng_max, P_max, nm_max, N, L1, L2, feas, cg, pl)

            for g in range(ng, ng_max):
                y1[g, :].ub = 0
            for p in range(P, P_max):
                y2[p, :].ub = 0
            mdl.update()

            self._firm_models[int(oid)] = (mdl, y1, y2, z, x)

        # local_problems for update_solver_settings compatibility
        self.local_problems = [self._firm_models[int(oid)][0]
                               for oid in unique_obs]

    def _compute_pi_and_grad(self, theta_d):
        """Compute pi and grad_pi w.r.t. 4 FE params for all obs.

        Returns pi (n_obs, nm_max, N, L1, L2), grad (n_obs, 4, ...), theta_fe (4,).
        """
        rf = compute_rev_factor(self.geo, theta_d, self.coefs)
        rf_t = rf.transpose(2, 0, 1)
        n_obs = self._shares_pad.shape[0]

        pi = (self._shares_pad[:, :, :, None, None]
              * self.geo['R_n'][None, None, :, None, None]
              * rf_t[None, None, :, :, :])

        c = self._eta_m1_kappa
        bp = self._beta_2_phi
        grad = np.zeros((n_obs, 4, self.nm_max, self.N, self.L1, self.L2))
        for k, mask in enumerate(self._cont1_masks):
            g_k = grad[:, k]
            g_k[:, :, :, mask, :] = c * bp * pi[:, :, :, mask, :]
        for k, mask in enumerate(self._cont2_masks):
            g_k = grad[:, 2 + k]
            g_k[:, :, :, :, mask] = c * pi[:, :, :, :, mask]

        theta_fe = np.array([theta_d['FE_1_As'], theta_d['FE_1_Eu'],
                             theta_d['FE_2_As'], theta_d['FE_2_Eu']])
        return pi, grad, theta_fe

    def _store_linearization(self, key, pi, grad_pi, theta_fe):
        pol = self.data_manager.local_data.id_data['policies']
        # Store at agent level by indexing global arrays by obs_ids
        pol[key] = {'pi': pi[self.obs_ids], 'grad_pi': grad_pi[self.obs_ids],
                    'theta_fe': theta_fe}

    def set_q_linearization(self, theta):
        theta_d = unpack_theta(theta)
        pi, grad, theta_fe = self._compute_pi_and_grad(theta_d)
        self._store_linearization('lin_Q', pi, grad, theta_fe)

    def solve(self, theta):
        theta_d = unpack_theta(theta)
        pi_global, grad_global, theta_fe = self._compute_pi_and_grad(theta_d)
        self._store_linearization('lin_V', pi_global, grad_global, theta_fe)

        n_items = self.dimensions_cfg.n_items
        n = self.comm_manager.num_local_agent
        results = np.zeros((n, n_items), dtype=bool)
        pol = self.data_manager.local_data.id_data['policies']

        for i in range(n):
            oid = int(self.obs_ids[i])
            firm = self.firms[i]
            ng = len(firm['ln_xi_1'])
            P, nm = firm['n_platforms'], firm['n_models']
            mdl, y1, y2, z, x = self._firm_models[oid]

            fc1_r, fc2_r = compute_facility_costs(firm, self.geo, theta_d)
            fc1 = np.zeros((self.ng_max, self.L1))
            fc1[:ng] = fc1_r
            fc2 = np.zeros((self.P_max, self.L2))
            fc2[:P] = fc2_r

            pi_i = pi_global[oid]

            mdl.setObjective(
                (pi_i + self.nu[i]).ravel() @ x.reshape(-1)
                - (fc1 + self.phi1[i]).ravel() @ y1.reshape(-1)
                - (fc2 + self.phi2[i]).ravel() @ y2.reshape(-1),
                gp.GRB.MAXIMIZE)
            mdl.optimize()

            pol['x_V'][i] = np.asarray(x.X)

            off = 0
            results[i, off:off + self.y1_size] = (
                np.asarray(y1.X) > 0.5).ravel()
            off += self.y1_size
            results[i, off:off + self.y2_size] = (
                np.asarray(y2.X) > 0.5).ravel()
            off += self.y2_size
            results[i, off:off + self.z_size] = (
                np.asarray(z.X) > 0.5).ravel()

        return results


def flatten_bundle(bun, ng_max, P_max, nm_max, L1, L2, N):
    n_items = ng_max * L1 + P_max * L2 + nm_max * N
    b = np.zeros(n_items, dtype=bool)
    ng, P, nm = bun['y1'].shape[0], bun['y2'].shape[0], bun['z'].shape[0]
    off = 0
    y1 = np.zeros((ng_max, L1), dtype=bool)
    y1[:ng] = bun['y1']
    b[off:off + ng_max * L1] = y1.ravel()
    off += ng_max * L1
    y2 = np.zeros((P_max, L2), dtype=bool)
    y2[:P] = bun['y2']
    b[off:off + P_max * L2] = y2.ravel()
    off += P_max * L2
    z = np.zeros((nm_max, N), dtype=bool)
    z[:nm] = bun['z']
    b[off:off + nm_max * N] = z.ravel()
    return b
