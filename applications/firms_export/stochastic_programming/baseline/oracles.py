import numpy as np


def build_oracles(model, seed=42, sigma_eps=1.0, sigma_nu_1=1.0, sigma_nu_2=1.0):
    ld = model.data.local_data
    n = model.comm_manager.num_local_agent
    M = model.config.dimensions.n_items
    R = ld.item_data["R"]
    beta = ld.item_data["beta"]
    perpetual = 1 / (1 - beta)
    rev_chars_1 = ld.id_data["rev_chars_1"]     # (n, n_rev, M)
    rev_chars_2 = ld.id_data["rev_chars_2"]     # (n, n_rev, M)
    C = ld.item_data["syn_chars"]               # (M, M)
    entry_chars = ld.item_data["entry_chars"]    # (M,)
    n_rev = rev_chars_1.shape[1]

    # ── Draw raw shocks ─────────────────────────────────────────────
    eps = np.zeros((n, M))
    nu1 = np.zeros((n, M))
    nu2 = np.zeros((n, R, M))
    for i, gid in enumerate(model.comm_manager.agent_ids):
        eps[i] = np.random.default_rng((seed, gid, 0)).normal(0, sigma_eps, M)
        nu1[i] = np.random.default_rng((seed, gid, 1)).normal(0, sigma_nu_1, M)
        nu2[i] = np.random.default_rng((seed, gid, 2)).normal(0, sigma_nu_2, (R, M))

    # ── Pre-compute composite errors (period 2 already discounted) ──
    eps_1 = eps + nu1                                        # (n, M)
    eps_2 = beta * (perpetual * eps[:, None, :] + nu2)       # (n, R, M)

    ld.errors["eps_1"] = eps_1
    ld.errors["eps_2"] = eps_2

    # ── Pre-discounted period-2 characteristics ─────────────────────
    rev_chars_2_d = beta * perpetual * rev_chars_2           # (n, n_rev, M)
    syn_chars_2 = beta * perpetual * C                       # (M, M)
    beta_s = beta                                            # scalar (entry cost only)

    ld.id_data["rev_chars_2_d"] = rev_chars_2_d
    ld.item_data["syn_chars_2"] = syn_chars_2
    ld.item_data["beta_s"] = beta_s

    # ── Helper ──────────────────────────────────────────────────────
    def _get_b_2_r(bundles, ids, data):
        pol = data.id_data["policies"]
        is_obs = bundles is data.id_data["obs_bundles"]
        return pol["b_2_r_Q"][ids] if is_obs else pol["b_2_r_V"][ids]

    # ── Covariates oracle ───────────────────────────────────────────
    def covariates_oracle(bundles, ids, data):
        b_0 = data.id_data["state_chars"][ids]       # (batch, M)
        b_1 = bundles.astype(float)                   # (batch, M)
        b_2_r = _get_b_2_r(bundles, ids, data).astype(float)  # (batch, R, M)

        # Revenue: period 1 + discounted period 2
        x_rev = (np.einsum('nm,nkm->nk', b_1, rev_chars_1[ids])
                 + np.einsum('nrm,nkm->nk', b_2_r, rev_chars_2_d[ids]) / R)

        # Constant entry cost: period 1 + beta * period 2
        switch_1 = (1 - b_0) * b_1                                      # (batch, M)
        switch_2 = (1 - b_1)[:, None, :] * b_2_r                        # (batch, R, M)
        x_s = switch_1.sum(-1) + beta_s * switch_2.sum(-1).mean(-1)

        # Cost-proportional entry cost
        x_sc = ((switch_1 * entry_chars).sum(-1)
                + beta_s * (switch_2 * entry_chars).sum(-1).mean(-1))

        # Synergy: period 1 (C) + discounted period 2 (syn_chars_2)
        x_c = (np.einsum('nj,jk,nk->n', b_1, C, b_1)
               + np.einsum('nrj,jk,nrk->n', b_2_r, syn_chars_2, b_2_r) / R)

        return np.column_stack([x_rev, x_s, x_sc, x_c])

    # ── Error oracle ────────────────────────────────────────────────
    def error_oracle(bundles, ids, data):
        b_1 = bundles.astype(float)
        b_2_r = _get_b_2_r(bundles, ids, data).astype(float)

        e_1 = (eps_1[ids] * b_1).sum(-1)
        e_2 = (eps_2[ids] * b_2_r).sum(-1).mean(-1)

        return e_1 + e_2

    return covariates_oracle, error_oracle
