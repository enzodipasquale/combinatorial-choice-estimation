"""Covariates and error oracles for multi-stage facility location (DC version)."""

import numpy as np

N_PARAMS = 14


def _build_linear_features(firms, geo, ng_max, P_max, nm_max):
    """(n_firms, n_items, 10) feature matrix for the 10 linear params (delta, rho)."""
    L1, L2, N = geo['L1'], geo['L2'], geo['n_markets']
    n_items = ng_max * L1 + P_max * L2 + nm_max * N
    nf = len(firms)
    X = np.zeros((nf, n_items, 10))

    for f, firm in enumerate(firms):
        hq = firm['hq_cont']
        ng = len(firm['ln_xi_1'])
        P = firm['n_platforms']

        for g in range(ng):
            for l in range(L1):
                idx = g * L1 + l
                c = geo['cont1'][l]
                X[f, idx, c] = -1.0                              # delta_1_{cont}
                X[f, idx, 6] = -geo['ln_d_hq1'][hq, l]           # rho_HQ_1
                X[f, idx, 8] = -firm['ln_xi_1'][g]                # rho_xi_1

        off = ng_max * L1
        for p in range(P):
            for l in range(L2):
                idx = off + p * L2 + l
                c = geo['cont2'][l]
                X[f, idx, 3 + c] = -1.0                           # delta_2_{cont}
                X[f, idx, 7] = -geo['ln_d_hq2'][hq, l]            # rho_HQ_2
                X[f, idx, 9] = -firm['ln_xi_2'][p]                # rho_xi_2

    return X


def build_oracles(model, geo, firms, ng_max, P_max, nm_max):
    obs_ids = model.comm_manager.obs_ids
    L1, L2, N = geo['L1'], geo['L2'], geo['n_markets']

    X_linear_global = _build_linear_features(firms, geo, ng_max, P_max, nm_max)
    X_linear = X_linear_global[obs_ids]                              # (n_local, n_items, 10)

    y1_size = ng_max * L1
    y2_size = P_max * L2

    def _get_lin(bundles, ids, data):
        pol = data.id_data['policies']
        is_obs = bundles is data.id_data['obs_bundles']
        lin = pol['lin_Q'] if is_obs else pol['lin_V']
        x = pol['x_Q'][ids] if is_obs else pol['x_V'][ids]
        return lin['pi'][ids], lin['grad_pi'][ids], lin['theta_fe'], x

    def covariates_oracle(bundles, ids, data):
        pi, grad_pi, theta_fe, x_vals = _get_lin(bundles, ids, data)
        n = len(ids)
        cov = np.zeros((n, N_PARAMS))

        # Columns 0-3: FE gradient = sum over paths of grad_pi * x
        # grad_pi shape: (n, 4, nm_max, N, L1, L2), x shape: (n, nm_max, N, L1, L2)
        for k in range(4):
            cov[:, k] = (grad_pi[:, k] * x_vals).reshape(n, -1).sum(-1)

        # Columns 4-13: linear features from y1, y2
        b = bundles.astype(float)
        cov[:, 4:] = np.einsum('fj,fjk->fk', b, X_linear[ids])

        return cov

    def error_oracle(bundles, ids, data):
        pi, grad_pi, theta_fe, x_vals = _get_lin(bundles, ids, data)
        n = len(ids)

        # Path revenue at linearization point
        pi_x = (pi * x_vals).reshape(n, -1).sum(-1)

        # FE gradient dot FE_at_linearization (to subtract from intercept)
        grad_dot_fe = 0.0
        for k in range(4):
            grad_dot_fe = grad_dot_fe + theta_fe[k] * (
                grad_pi[:, k] * x_vals).reshape(n, -1).sum(-1)

        # Modular errors: nu * x - phi1 * y1 - phi2 * y2
        phi1 = data.errors['phi1'][ids]
        phi2 = data.errors['phi2'][ids]
        nu = data.errors['nu'][ids]

        e_nu = (nu * x_vals).reshape(n, -1).sum(-1)

        b = bundles.astype(float)
        e_y1 = -(phi1.reshape(n, -1) * b[:, :y1_size]).sum(-1)
        e_y2 = -(phi2.reshape(n, -1) * b[:, y1_size:y1_size + y2_size]).sum(-1)

        return pi_x - grad_dot_fe + e_nu + e_y1 + e_y2

    return covariates_oracle, error_oracle
