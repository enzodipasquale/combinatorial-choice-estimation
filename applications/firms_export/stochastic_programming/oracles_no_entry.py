import numpy as np


def build_oracles(model, seed=42):
    ld = model.data.local_data
    n = model.comm_manager.num_local_agent
    M = model.config.dimensions.n_items
    R = ld.item_data["R"]
    beta = ld.item_data["beta"]
    rev_chars_1 = ld.item_data["rev_chars_1"]   # (n_rev, M) period 1
    rev_chars_2 = ld.item_data["rev_chars_2"]   # (n_rev, M) period 2
    syn_chars = ld.item_data["syn_chars"]         # (n_syn, M, M)
    n_syn = syn_chars.shape[0]

    eps1 = np.zeros((n, M))
    eps2 = np.zeros((n, R, M))
    for i, gid in enumerate(model.comm_manager.agent_ids):
        eps1[i] = np.random.default_rng((seed, gid, 0)).normal(0, 1, M)
        eps2[i] = np.random.default_rng((seed, gid, 1)).normal(0, 1, (R, M))

    ld.errors["eps1"] = eps1
    ld.errors["eps2"] = eps2

    def _get_b_2_r(bundles, ids, data):
        pol = data.id_data["policies"]
        is_obs = bundles is data.id_data["obs_bundles"]
        return pol["b_2_r_Q"][ids] if is_obs else pol["b_2_r_V"][ids]

    def covariates_oracle(bundles, ids, data):
        b_1 = bundles.astype(float)
        b_2_r = _get_b_2_r(bundles, ids, data).astype(float)

        # Revenue features (θ_rev): b_1·rev1 + β·b_2_r·rev2/R
        x_rev = (b_1 @ rev_chars_1.T
                 + beta * np.einsum('nrm,km->nk', b_2_r, rev_chars_2) / R)

        # Synergy features (θ_syn): b_1'C_l b_1 + β·b_2_r'C_l b_2_r/R
        x_syn = np.zeros((b_1.shape[0], n_syn))
        for l in range(n_syn):
            x_syn[:, l] = (np.einsum('nj,jk,nk->n', b_1, syn_chars[l], b_1)
                           + beta * np.einsum('nrj,jk,nrk->n', b_2_r,
                                              syn_chars[l], b_2_r) / R)

        return np.column_stack([x_rev, x_syn])

    def error_oracle(bundles, ids, data):
        b_1 = bundles.astype(float)
        e1 = (eps1[ids] * b_1).sum(-1)
        b_2_r = _get_b_2_r(bundles, ids, data).astype(float)
        e2 = (eps2[ids] * b_2_r).sum(-1).mean(-1)
        return e1 + beta * e2

    return covariates_oracle, error_oracle
