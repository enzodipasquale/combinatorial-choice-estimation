"""Facility costs for multi-stage production — 10-param model with HQ distance."""

import numpy as np


def compute_facility_costs(firm, theta):
    """Structural fixed costs with HQ distance.

    fc1 = delta_1 + rho_xi_1 * ln_xi_1[0] + rho_HQ_1 * d_hq1   → (1, L1)
    fc2_p = delta_2 + rho_xi_2 * ln_xi_2[p] + rho_HQ_2 * d_hq2  → (P, L2)
    """
    fc1 = (theta['delta_1']
           + theta['rho_xi_1'] * firm['ln_xi_1'][0]
           + theta['rho_HQ_1'] * firm['d_hq1'])               # (L1,)
    fc2 = (theta['delta_2']
           + theta['rho_xi_2'] * firm['ln_xi_2'][:, None]     # (P, 1)
           + theta['rho_HQ_2'] * firm['d_hq2'][None, :])      # (1, L2)  → (P, L2)
    return fc1[None, :], fc2                                   # (1, L1), (P, L2)
