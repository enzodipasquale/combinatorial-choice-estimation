"""Covariates and error oracles for multi-stage facility location (12-param linear model)."""

import numpy as np

N_PARAMS = 12


def _build_linear_features(firms, geo, ng_max, P_max, nm_max):
    """(n_firms, n_items, 10) feature matrix for the 10 facility-cost params.

    Columns 0-5: delta_1, delta_2, rho_xi_1, rho_xi_2, rho_HQ_1, rho_HQ_2.
    Columns 6-9: FE_1_r1, FE_1_r2, FE_2_r1, FE_2_r2.
    All multiply y-variables (bundle bits), not x.

    Sign: cost = ... - FE*indicator, so dv/dFE = +indicator on y items.
    """
    L1, L2, N = geo['L1'], geo['L2'], geo['n_markets']
    n_items = ng_max * L1 + P_max * L2 + nm_max * N
    nf = len(firms)
    X = np.zeros((nf, n_items, 10))

    cell_r1 = (geo['cell_region'] == 1)
    cell_r2 = (geo['cell_region'] == 2)
    asm_r1 = (geo['asm_region'] == 1)
    asm_r2 = (geo['asm_region'] == 2)

    for f, firm in enumerate(firms):
        # Cell items: y1[g=0, l1]
        for l in range(L1):
            idx = l
            X[f, idx, 0] = -1.0                             # delta_1
            X[f, idx, 2] = -firm['ln_xi_1'][0]              # rho_xi_1
            X[f, idx, 4] = -firm['d_hq1'][l]                # rho_HQ_1
            if cell_r1[l]:
                X[f, idx, 6] = +1.0                         # FE_1_r1
            if cell_r2[l]:
                X[f, idx, 7] = +1.0                         # FE_1_r2

        # Assembly items: y2[p, l2]
        P = firm['n_platforms']
        off = ng_max * L1
        for p in range(P):
            for l in range(L2):
                idx = off + p * L2 + l
                X[f, idx, 1] = -1.0                         # delta_2
                X[f, idx, 3] = -firm['ln_xi_2'][p]          # rho_xi_2
                X[f, idx, 5] = -firm['d_hq2'][l]            # rho_HQ_2
                if asm_r1[l]:
                    X[f, idx, 8] = +1.0                     # FE_2_r1
                if asm_r2[l]:
                    X[f, idx, 9] = +1.0                     # FE_2_r2

    return X


def _build_distance_weights(firms, geo, nm_max):
    """Precompute per-firm weight tensors for rho_d and constant.

    All shape (n_firms, nm_max, N, L1, L2).
    """
    nf = len(firms)
    N, L1, L2 = geo['n_markets'], geo['L1'], geo['L2']
    d12 = geo['d_12']
    d2m = geo['d_2m']

    w_d1 = np.zeros((nf, nm_max, N, L1, L2))
    w_d2 = np.zeros((nf, nm_max, N, L1, L2))
    w_const = np.zeros((nf, nm_max, N, L1, L2))

    for f, firm in enumerate(firms):
        nm = firm['n_models']
        for m in range(nm):
            sR = firm['shares'][m, :] * geo['R_n']
            sR_3d = sR[:, None, None]
            w_const[f, m] = sR_3d
            w_d1[f, m] = sR_3d * d12[None, :, :]
            w_d2[f, m] = sR_3d * d2m.T[:, None, :]

    return w_d1, w_d2, w_const


def build_oracles(model, geo, firms, ng_max, P_max, nm_max):
    obs_ids = model.comm_manager.obs_ids
    L1, L2, N = geo['L1'], geo['L2'], geo['n_markets']

    X_linear_global = _build_linear_features(firms, geo, ng_max, P_max, nm_max)
    X_linear = X_linear_global[obs_ids]

    w_d1_global, w_d2_global, w_const_global = \
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

        # Columns 0-9: facility cost features from bundle bits (y1, y2)
        b = bundles.astype(float)
        cov[:, :10] = np.einsum('fj,fjk->fk', b, X_linear[ids])

        # Columns 10-11: rho_d_1 and rho_d_2 from x side-channel
        oids = obs_ids[ids]
        cov[:, 10] = -(w_d1_global[oids] * x_vals).reshape(n, -1).sum(-1)
        cov[:, 11] = -(w_d2_global[oids] * x_vals).reshape(n, -1).sum(-1)

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
