"""Revenue factors and facility costs for multi-stage production."""

import numpy as np


def compute_rev_factor(geo, theta, coefs):
    """Base revenue factor (L1, L2, N). Excludes firm shares and R_n."""
    eta = coefs['eta']
    kappa = 1.0 / abs(coefs['beta_2_T'])
    b = coefs

    fe1 = np.array([0.0, theta['FE_1_As'], theta['FE_1_Eu']])
    fe2 = np.array([0.0, theta['FE_2_As'], theta['FE_2_Eu']])

    src_1 = (b['beta_1_D'] * geo['ln_d_12']
             + b['beta_1_T'] * geo['tau_12']
             + fe1[geo['cont1']][:, None])                          # (L1, L2)
    src_2 = (b['beta_2_D'] * geo['ln_d_2m']
             + b['beta_2_T'] * geo['tau_2m']
             + fe2[geo['cont2']][:, None])                          # (L2, N)

    src = b['beta_2_phi'] * src_1[:, :, None] + src_2[None, :, :]    # (L1, L2, N)
    return (1.0 / eta) * np.exp((eta - 1) * kappa * src)


def compute_facility_costs(firm, geo, theta):
    """Structural fixed costs: cell (ng, L1), assembly (P, L2)."""
    hq = firm['hq_cont']
    d1 = np.array([theta['delta_1_Am'], theta['delta_1_As'], theta['delta_1_Eu']])
    d2 = np.array([theta['delta_2_Am'], theta['delta_2_As'], theta['delta_2_Eu']])

    fc1 = (d1[geo['cont1']][None, :]
           + theta['rho_HQ_1'] * geo['ln_d_hq1'][hq][None, :]
           + theta['rho_xi_1'] * firm['ln_xi_1'][:, None])

    fc2 = (d2[geo['cont2']][None, :]
           + theta['rho_HQ_2'] * geo['ln_d_hq2'][hq][None, :]
           + theta['rho_xi_2'] * firm['ln_xi_2'][:, None])

    return fc1, fc2
