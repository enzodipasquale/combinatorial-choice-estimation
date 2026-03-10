import numpy as np


def static_covariates(state, action, revenue):
    return np.column_stack([action @ revenue, (action * (1 - state)).sum(-1)])


def build_oracles(model, seed=42):
    ld = model.data.local_data
    n = model.comm_manager.num_local_agent
    M = model.config.dimensions.n_items
    R = ld["item_data"]["R"]
    beta = ld["item_data"]["beta"]
    rev = ld["item_data"]["revenue"]

    eps1 = np.zeros((n, M))
    eps2 = np.zeros((n, R, M))
    for i, gid in enumerate(model.comm_manager.agent_ids):
        eps1[i] = np.random.default_rng((seed, gid, 0)).normal(0, 1, M)
        eps2[i] = np.random.default_rng((seed, gid, 1)).normal(0, 1, (R, M))

    ld["errors"] = {"eps1": eps1, "eps2": eps2}

    def _get_d(bundles, ids, data):
        pol = data["policies"]
        is_V = np.array_equal(bundles, pol["b_star"][ids])
        return pol["d_V"][ids] if is_V else pol["d_Q"][ids]

    def covariates_oracle(bundles, ids, data):
        state = data["id_data"]["state"][ids].astype(float)
        bf = bundles.astype(float)
        x1 = static_covariates(state, bf, rev)
        d = _get_d(bundles, ids, data).astype(float)
        c0 = np.einsum('nrm,m->n', d, rev) / R
        c1 = (d * (1 - bf[:, None, :])).sum(-1).mean(-1)
        return x1 + beta * np.column_stack([c0, c1])

    def error_oracle(bundles, ids, data):
        bf = bundles.astype(float)
        e1 = (eps1[ids] * bf).sum(-1)
        d = _get_d(bundles, ids, data).astype(float)
        e2 = (eps2[ids] * d).sum(-1).mean(-1)
        return e1 + beta * e2

    return covariates_oracle, error_oracle
