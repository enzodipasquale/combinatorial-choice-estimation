"""Covariates and error oracles for multi-stage facility location (10-param linear model)."""

import numpy as np

N_PARAMS = 10


def _build_linear_features(firms, geo, ng_max, P_max, nm_max):
    """(n_firms, n_items, 6) feature matrix for the 6 facility-cost params.

    Columns: delta_1, delta_2, rho_xi_1, rho_xi_2, rho_HQ_1, rho_HQ_2.
    These multiply y-variables (bundle bits), not x.
    """
    L1, L2, N = geo['L1'], geo['L2'], geo['n_markets']
    n_items = ng_max * L1 + P_max * L2 + nm_max * N
    nf = len(firms)
    X = np.zeros((nf, n_items, 6))

    for f, firm in enumerate(firms):
        # Cell items: y1[g=0, l1] — single cell group
        for l in range(L1):
            idx = l
            X[f, idx, 0] = -1.0                             # delta_1
            X[f, idx, 2] = -firm['ln_xi_1'][0]              # rho_xi_1
            X[f, idx, 4] = -firm['d_hq1'][l]                # rho_HQ_1

        # Assembly items: y2[p, l2] for p in range(P)
        P = firm['n_platforms']
        off = ng_max * L1
        for p in range(P):
            for l in range(L2):
                idx = off + p * L2 + l
                X[f, idx, 1] = -1.0                         # delta_2
                X[f, idx, 3] = -firm['ln_xi_2'][p]          # rho_xi_2 (platform-specific)
                X[f, idx, 5] = -firm['d_hq2'][l]            # rho_HQ_2

    return X


def _build_distance_weights(firms, geo, nm_max):
    """Precompute per-firm weight tensors for rho_d, FE, and constant.

    All shape (n_firms, nm_max, N, L1, L2).
    """
    nf = len(firms)
    N, L1, L2 = geo['n_markets'], geo['L1'], geo['L2']
    d12 = geo['d_12']
    d2m = geo['d_2m']
    cell_r1 = (geo['cell_region'] == 1).astype(float)
    asm_r1 = (geo['asm_region'] == 1).astype(float)

    w_d1 = np.zeros((nf, nm_max, N, L1, L2))
    w_d2 = np.zeros((nf, nm_max, N, L1, L2))
    w_const = np.zeros((nf, nm_max, N, L1, L2))
    w_fe1 = np.zeros((nf, nm_max, N, L1, L2))
    w_fe2 = np.zeros((nf, nm_max, N, L1, L2))

    for f, firm in enumerate(firms):
        nm = firm['n_models']
        for m in range(nm):
            sR = firm['shares'][m, :] * geo['R_n']           # (N,)
            sR_3d = sR[:, None, None]
            w_const[f, m] = sR_3d
            w_d1[f, m] = sR_3d * d12[None, :, :]
            w_d2[f, m] = sR_3d * d2m.T[:, None, :]
            w_fe1[f, m] = sR_3d * cell_r1[None, :, None]
            w_fe2[f, m] = sR_3d * asm_r1[None, None, :]

    return w_d1, w_d2, w_const, w_fe1, w_fe2


def build_oracles(model, geo, firms, ng_max, P_max, nm_max):
    obs_ids = model.comm_manager.obs_ids
    L1, L2, N = geo['L1'], geo['L2'], geo['n_markets']

    X_linear_global = _build_linear_features(firms, geo, ng_max, P_max, nm_max)
    X_linear = X_linear_global[obs_ids]

    w_d1_global, w_d2_global, w_const_global, w_fe1_global, w_fe2_global = \
        _build_distance_weights(firms, geo, nm_max)

    y1_size = ng_max * L1
    y2_size = P_max * L2

    def _get_x(bundles, ids, data):
        pol = data.id_data['policies']
        is_obs = bundles is data.id_data['obs_bundles']
        return pol['x_Q'][ids] if is_obs else pol['x_V'][ids]

    def covariates_oracle(bundles, ids, data):
        x_vals = _get_x(bundles, ids, data)
        n = len(ids)
        cov = np.zeros((n, N_PARAMS))

        b = bundles.astype(float)
        cov[:, :6] = np.einsum('fj,fjk->fk', b, X_linear[ids])

        oids = obs_ids[ids]
        cov[:, 6] = -(w_d1_global[oids] * x_vals).reshape(n, -1).sum(-1)
        cov[:, 7] = -(w_d2_global[oids] * x_vals).reshape(n, -1).sum(-1)
        cov[:, 8] = (w_fe1_global[oids] * x_vals).reshape(n, -1).sum(-1)
        cov[:, 9] = (w_fe2_global[oids] * x_vals).reshape(n, -1).sum(-1)

        return cov

    def error_oracle(bundles, ids, data):
        x_vals = _get_x(bundles, ids, data)
        n = len(ids)

        phi1 = data.errors['phi1'][ids]
        phi2 = data.errors['phi2'][ids]
        nu = data.errors['nu'][ids]

        e_const = (w_const_global[obs_ids[ids]] * x_vals).reshape(n, -1).sum(-1)
        e_nu = (nu * x_vals).reshape(n, -1).sum(-1)

        b = bundles.astype(float)
        e_y1 = -(phi1.reshape(n, -1) * b[:, :y1_size]).sum(-1)
        e_y2 = -(phi2.reshape(n, -1) * b[:, y1_size:y1_size + y2_size]).sum(-1)

        return e_const + e_nu + e_y1 + e_y2

    return covariates_oracle, error_oracle
