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
    C = ld.item_data["syn_chars"]               # (M, M) exp(-dist_jk), zero diag
    entry_chars = ld.item_data["entry_chars"]    # (M,) d_j in thousands km
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
    eps_2_perm = beta * perpetual * eps                      # (n, M)
    eps_2 = eps_2_perm[:, None, :] + beta * nu2              # (n, R, M)

    ld.errors["eps_1"] = eps_1
    ld.errors["eps_2"] = eps_2
    ld.errors["eps_2_perm"] = eps_2_perm

    # ── Pre-discounted period-2 characteristics ─────────────────────
    rev_chars_2_d = beta * perpetual * rev_chars_2           # (n, n_rev, M)
    beta_s = beta                                            # scalar (entry cost only)

    ld.id_data["rev_chars_2_d"] = rev_chars_2_d
    ld.item_data["beta_s"] = beta_s

    # ── Helper ──────────────────────────────────────────────────────
    def _get_b_2_r(bundles, ids, data):
        pol = data.id_data["policies"]
        is_obs = bundles is data.id_data["obs_bundles"]
        return pol["b_2_r_Q"][ids] if is_obs else pol["b_2_r_V"][ids]

    # ── Covariates oracle ───────────────────────────────────────────
    # 4 covariates: rev, entry_c, entry_dist, entry_syn_d
    def covariates_oracle(bundles, ids, data):
        b_0 = data.id_data["state_chars"][ids]       # (batch, M)
        b_1 = bundles.astype(float)                   # (batch, M)
        b_2_r = _get_b_2_r(bundles, ids, data).astype(float)  # (batch, R, M)

        # Revenue: period 1 + discounted period 2
        x_rev = (np.einsum('nm,nkm->nk', b_1, rev_chars_1[ids])
                 + np.einsum('nrm,nkm->nk', b_2_r, rev_chars_2_d[ids]) / R)

        # Entry cost constant: period 1 + beta * period 2
        switch_1 = (1 - b_0) * b_1                                      # (batch, M)
        switch_2 = (1 - b_1)[:, None, :] * b_2_r                        # (batch, R, M)
        x_s = switch_1.sum(-1) + beta_s * switch_2.sum(-1).mean(-1)

        # Entry cost * distance
        x_sd = ((switch_1 * entry_chars).sum(-1)
                + beta_s * (switch_2 * entry_chars).sum(-1).mean(-1))

        # Entry synergy: (1-s_j) * b_j * d_j * Σ_k s_k * C_jk
        # Period 1: s = b_0, gate = (1-b_0)*b_1 = switch_1
        syn_1 = b_0 @ C * entry_chars                                   # (batch, M)
        x_syn = (switch_1 * syn_1).sum(-1)
        # Period 2: s = b_1, gate = (1-b_1)*b_2r = switch_2
        syn_2 = b_1 @ C * entry_chars                                   # (batch, M)
        x_syn += beta_s * (switch_2 * syn_2[:, None, :]).sum(-1).mean(-1)

        return np.column_stack([x_rev, x_s, x_sd, x_syn])

    # ── Error oracle ────────────────────────────────────────────────
    def error_oracle(bundles, ids, data):
        b_1 = bundles.astype(float)
        b_2_r = _get_b_2_r(bundles, ids, data).astype(float)

        e_1 = (eps_1[ids] * b_1).sum(-1)
        e_2 = (eps_2[ids] * b_2_r).sum(-1).mean(-1)

        return e_1 + e_2

    return covariates_oracle, error_oracle
