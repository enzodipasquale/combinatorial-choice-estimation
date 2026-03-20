import numpy as np
from scipy.stats import norm


def build_oracles(model, beta=0.0, seed=42, sigma_1=1.0, sigma_2=1.0):
    ld = model.data.local_data
    n = model.comm_manager.num_local_agent
    M = model.config.dimensions.n_items
    perpetual = 1 / (1 - beta) if beta < 1 else 1.0
    rev_chars_1 = ld.id_data["rev_chars_1"]
    rev_chars_2 = ld.id_data["rev_chars_2"]
    C = ld.item_data["syn_chars"]
    entry_chars = ld.item_data["entry_chars"]

    rev_chars_2_d = beta * perpetual * rev_chars_2
    entry_chars_2 = beta * entry_chars
    C_d_2 = beta * C * entry_chars[:, None]
    C_2 = beta * C
    discount_2 = beta
    beta_s = beta

    eps_1 = np.zeros((n, M))
    for i, gid in enumerate(model.comm_manager.agent_ids):
        eps_1[i] = np.random.default_rng((seed, gid, 0)).normal(0, sigma_1, M)

    ld.errors["eps_1"] = eps_1
    ld.id_data["rev_chars_2_d"] = rev_chars_2_d
    ld.item_data["entry_chars_2"] = entry_chars_2
    ld.item_data["C_d_2"] = C_d_2
    ld.item_data["C_2"] = C_2
    ld.item_data["discount_2"] = discount_2
    ld.item_data["sigma_2"] = sigma_2

    def _get_mu(bundles, ids, data):
        pol = data.id_data["policies"]
        is_obs = bundles is data.id_data["obs_bundles"]
        return pol["mu_Q"][ids] if is_obs else pol["mu_V"][ids]

    def covariates_oracle(bundles, ids, data):
        b_0 = data.id_data["state_chars"][ids]
        b_1 = bundles.astype(float)
        mu = _get_mu(bundles, ids, data)
        phi = norm.cdf(mu / sigma_2)

        x_rev = (np.einsum('nm,nkm->nk', b_1, rev_chars_1[ids])
                 + np.einsum('nm,nkm->nk', phi, rev_chars_2_d[ids]))

        switch_1 = (1 - b_0) * b_1
        switch_2 = (1 - b_1) * phi
        x_s = switch_1.sum(-1) + beta_s * switch_2.sum(-1)

        x_sd = ((switch_1 * entry_chars).sum(-1)
                + beta_s * (switch_2 * entry_chars).sum(-1))

        syn_base_1 = b_0 @ C
        x_syn = (switch_1 * syn_base_1).sum(-1)
        syn_base_2 = b_1 @ C
        x_syn += beta_s * (switch_2 * syn_base_2).sum(-1)

        x_syn_d = (switch_1 * syn_base_1 * entry_chars).sum(-1)
        x_syn_d += beta_s * (switch_2 * syn_base_2 * entry_chars).sum(-1)

        return np.column_stack([x_rev, x_s, x_sd, x_syn, x_syn_d])

    def error_oracle(bundles, ids, data):
        b_1 = bundles.astype(float)
        mu = _get_mu(bundles, ids, data)

        e_1 = (eps_1[ids] * b_1).sum(-1)
        e_2 = (sigma_2 * norm.pdf(mu / sigma_2)).sum(-1)

        return e_1 + e_2

    return covariates_oracle, error_oracle
