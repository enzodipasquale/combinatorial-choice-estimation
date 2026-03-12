import numpy as np


def build_oracles(model, seed=42):
    ld = model.data.local_data
    n = model.comm_manager.num_local_agent
    M = model.config.dimensions.n_items
    R = ld.item_data["R"]
    beta = ld.item_data["beta"]
    rev_chars = ld.item_data["rev_chars"]       # (n_rev, M)
    C = ld.item_data["syn_chars"]               # (M, M) pairwise complementarity
    n_rev = rev_chars.shape[0]

    eps1 = np.zeros((n, M))
    eps2 = np.zeros((n, R, M))
    for i, gid in enumerate(model.comm_manager.agent_ids):
        eps1[i] = np.random.default_rng((seed, gid, 0)).normal(0, 1, M)
        eps2[i] = np.random.default_rng((seed, gid, 1)).normal(0, 1, (R, M))

    ld.errors["eps1"] = eps1
    ld.errors["eps2"] = eps2

    def _get_d(bundles, ids, data):
        pol = data.id_data["policies"]
        is_obs = bundles is data.id_data["obs_bundles"]
        return pol["d_Q"][ids] if is_obs else pol["d_V"][ids]

    def covariates_oracle(bundles, ids, data):
        state = data.id_data["state_chars"][ids]  # (batch, M)
        bf = bundles.astype(float)
        d = _get_d(bundles, ids, data).astype(float)  # (batch, R, M)

        # Revenue features (θ_rev): b·rev + β·d·rev/R
        x_rev = bf @ rev_chars.T + beta * np.einsum('nrm,km->nk', d, rev_chars) / R

        # Entry cost feature (θ_s): (1-state)·b + β·(1-b)·d
        x_s = ((1 - state) * bf).sum(-1) \
            + beta * (d * (1 - bf[:, None, :])).sum(-1).mean(-1)

        # Synergy feature (θ_c): b'Cb + β·d'Cd/R
        x_c = np.einsum('nj,jk,nk->n', bf, C, bf) \
            + beta * np.einsum('nrj,jk,nrk->n', d, C, d) / R

        return np.column_stack([x_rev, x_s, x_c])

    def error_oracle(bundles, ids, data):
        bf = bundles.astype(float)
        e1 = (eps1[ids] * bf).sum(-1)
        d = _get_d(bundles, ids, data).astype(float)
        e2 = (eps2[ids] * d).sum(-1).mean(-1)
        return e1 + beta * e2

    return covariates_oracle, error_oracle
