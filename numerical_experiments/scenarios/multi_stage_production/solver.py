import numpy as np
import gurobipy as gp
from combest.subproblems.solver_base import SubproblemSolver, create_gurobi_model
from combest.utils import get_logger
from costs import compute_facility_costs
from milp import build_milp_shell

logger = get_logger(__name__)


THETA_NAMES = [
    'delta_1', 'delta_2', 'rho_xi_1', 'rho_xi_2',
    'rho_HQ_1', 'rho_HQ_2', 'FE_1_r1', 'FE_1_r2',
    'FE_2_r1', 'FE_2_r2', 'rho_d_1', 'rho_d_2',
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

        self.phi1 = self.data_manager.local_data.errors['phi1']
        self.phi2 = self.data_manager.local_data.errors['phi2']
        self.nu = self.data_manager.local_data.errors['nu']

        id_data['policies'] = {
            'x_V': np.zeros((n, nm_max, N, L1, L2)),
            'x_Q': id_data.get('x_Q', np.zeros((n, nm_max, N, L1, L2))),
        }

        # Build one MILP per local agent (one per simulation × observation)
        self.local_problems = []
        self._agent_vars = []                                  # (y1, y2, z, x) per agent
        for i in range(n):
            oid = int(self.obs_ids[i])
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

            mdl.ModelSense = gp.GRB.MINIMIZE
            self.local_problems.append(mdl)
            self._agent_vars.append((y1, y2, z, x))

    def solve(self, theta):
        theta_d = unpack_theta(theta)

        n_items = self.dimensions_cfg.n_items
        n = self.comm_manager.num_local_agent
        results = np.zeros((n, n_items), dtype=bool)
        pol = self.data_manager.local_data.id_data['policies']

        # Distance matrices for pi computation (FE moved to cost side)
        d12 = self.geo['d_12']                                  # (L1, L2)
        d2m = self.geo['d_2m']                                  # (L2, N)
        R_n = self.geo['R_n']                                   # (N,)

        # rev_factor (N, L1, L2) — shared across firms
        rev_factor = (1.0
                      - theta_d['rho_d_1'] * d12[None, :, :]
                      - theta_d['rho_d_2'] * d2m.T[:, None, :])

        for i in range(n):
            firm = self.firms[i]
            mdl = self.local_problems[i]
            y1, y2, z, x = self._agent_vars[i]

            # Per-firm pi: s_{m,n} * R_n * rev_factor → (nm_max, N, L1, L2)
            nm = firm['n_models']
            P = firm['n_platforms']
            sR = firm['shares'] * R_n[None, :]                  # (nm, N)
            pi_i = np.zeros((self.nm_max, self.N, self.L1, self.L2))
            pi_i[:nm] = sR[:, :, None, None] * rev_factor[None, :, :, :]

            ng = len(firm['ln_xi_1'])
            fc1_r, fc2_r = compute_facility_costs(firm, self.geo, theta_d)  # (ng, L1), (P, L2)
            fc1 = np.zeros((self.ng_max, self.L1))
            fc1[:ng] = fc1_r
            fc2 = np.zeros((self.P_max, self.L2))
            fc2[:P] = fc2_r

            # Update obj coefficients in-place (preserves warm start)
            x.Obj = -(pi_i + self.nu[i])                       # negate for minimization
            y1.Obj = (fc1 + self.phi1[i])
            y2.Obj = (fc2 + self.phi2[i])
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
