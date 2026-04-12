"""Facility costs for multi-stage production — 12-param model with HQ distance and 3-region FE."""

import numpy as np


def compute_facility_costs(firm, geo, theta):
    """Structural fixed costs with HQ distance and 3-region FE on cost side.

    fc1[g,l1] = delta_1 + rho_xi_1*ln_xi_1[g] + rho_HQ_1*d_hq1[l1] - fe1[region[l1]]
    fc2[p,l2] = delta_2 + rho_xi_2*ln_xi_2[p] + rho_HQ_2*d_hq2[l2] - fe2[region[l2]]

    Returns (ng, L1), (P, L2).
    """
    fe1 = np.array([0.0, theta.get('FE_1_r1', 0.0), theta.get('FE_1_r2', 0.0)])
    fe2 = np.array([0.0, theta.get('FE_2_r1', 0.0), theta.get('FE_2_r2', 0.0)])

    fc1 = (theta['delta_1']
           + theta['rho_xi_1'] * firm['ln_xi_1'][:, None]     # (ng, 1)
           + theta['rho_HQ_1'] * firm['d_hq1'][None, :]       # (1, L1)
           - fe1[geo['cell_region']][None, :])                 # (1, L1) → (ng, L1)
    fc2 = (theta['delta_2']
           + theta['rho_xi_2'] * firm['ln_xi_2'][:, None]     # (P, 1)
           + theta['rho_HQ_2'] * firm['d_hq2'][None, :]       # (1, L2)
           - fe2[geo['asm_region']][None, :])                  # (1, L2) → (P, L2)
    return fc1, fc2                                            # (ng, L1), (P, L2)
