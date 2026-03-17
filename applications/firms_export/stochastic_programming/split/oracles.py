import numpy as np


def build_oracles(model, seed=42, sigma_1=1.0, sigma_2=1.0):
    ld = model.data.local_data
    n = model.comm_manager.num_local_agent
    M = model.config.dimensions.n_items
    R = ld.item_data["R"]
    rev_chars_1 = ld.id_data["rev_chars_1"]
    rev_chars_2 = ld.id_data["rev_chars_2"]
    C = ld.item_data["syn_chars"]
    entry_chars = ld.item_data["entry_chars"]

    eps_1 = np.zeros((n, M))
    eps_2 = np.zeros((n, R, M))
    for i, gid in enumerate(model.comm_manager.agent_ids):
        eps_1[i] = np.random.default_rng((seed, gid, 0)).normal(0, sigma_1, M)
        eps_2[i] = np.random.default_rng((seed, gid, 1)).normal(0, sigma_2, (R, M))

    ld.errors["eps_1"] = eps_1
    ld.errors["eps_2"] = eps_2

    def _get_b_2_r(bundles, ids, data):
        pol = data.id_data["policies"]
        is_obs = bundles is data.id_data["obs_bundles"]
        return pol["b_2_r_Q"][ids] if is_obs else pol["b_2_r_V"][ids]

    def covariates_oracle(bundles, ids, data):
        b_0 = data.id_data["state_chars"][ids]
        b_1 = bundles.astype(float)
        b_2_r = _get_b_2_r(bundles, ids, data).astype(float)

        switch_1 = (1 - b_0) * b_1
        switch_2 = (1 - b_1)[:, None, :] * b_2_r

        x_rev_1 = np.einsum('nm,nkm->nk', b_1, rev_chars_1[ids])
        x_s_1 = switch_1.sum(-1)
        x_sd_1 = (switch_1 * entry_chars).sum(-1)
        syn_base_1 = b_0 @ C
        x_syn_1 = (switch_1 * syn_base_1).sum(-1)
        x_syn_d_1 = (switch_1 * syn_base_1 * entry_chars).sum(-1)

        x_rev_2 = np.einsum('nrm,nkm->nk', b_2_r, rev_chars_2[ids]) / R
        x_s_2 = switch_2.sum(-1).mean(-1)
        x_sd_2 = (switch_2 * entry_chars).sum(-1).mean(-1)
        syn_base_2 = b_1 @ C
        x_syn_2 = (switch_2 * syn_base_2[:, None, :]).sum(-1).mean(-1)
        x_syn_d_2 = (switch_2 * syn_base_2[:, None, :] * entry_chars).sum(-1).mean(-1)

        return np.column_stack([x_rev_1, x_s_1, x_sd_1, x_syn_1, x_syn_d_1,
                                x_rev_2, x_s_2, x_sd_2, x_syn_2, x_syn_d_2])

    def error_oracle(bundles, ids, data):
        b_1 = bundles.astype(float)
        b_2_r = _get_b_2_r(bundles, ids, data).astype(float)

        e_1 = (eps_1[ids] * b_1).sum(-1)
        e_2 = (eps_2[ids] * b_2_r).sum(-1).mean(-1)

        return e_1 + e_2

    return covariates_oracle, error_oracle
