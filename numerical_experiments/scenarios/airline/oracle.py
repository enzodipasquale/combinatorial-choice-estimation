"""Oracle functions for the airline / gross-substitutes scenario.

- compute_utility:       V_i(b) for a single airline
- brute_force_demand:    exact optimizer (enumerate 2^M bundles)
- greedy_demand:         GS greedy optimizer (standalone)
- make_find_best_item:   vectorized O(M) oracle for combest GreedySolver
- build_covariates_oracle: covariates with firm-specific fixed costs
"""

import numpy as np
from itertools import product as cartesian_product


def compute_utility(bundle, phi, theta_rev, theta_fc_i, theta_gs, hubs_i,
                    origin_of, errors_i):
    """Full utility V_i(b) for one airline."""
    n_active = bundle.sum()
    modular = (phi[bundle] @ theta_rev).sum() - theta_fc_i * n_active + errors_i[bundle].sum()
    congestion = 0.0
    if theta_gs > 0:
        for h in hubs_i:
            n_h = int((bundle & (origin_of == h)).sum())
            congestion += n_h ** 2
    return modular - theta_gs * congestion


def brute_force_demand(phi, theta_rev, theta_fc_i, theta_gs, hubs_i, origin_of,
                       errors_i, M):
    """Enumerate all 2^M bundles, return maximizer. For verification only."""
    best_val = float('-inf')
    best_bundle = None
    for bits in cartesian_product([False, True], repeat=M):
        b = np.array(bits, dtype=bool)
        val = compute_utility(b, phi, theta_rev, theta_fc_i, theta_gs, hubs_i, origin_of, errors_i)
        if val > best_val:
            best_val = val
            best_bundle = b.copy()
    return best_bundle, best_val


def greedy_demand(phi, theta_rev, theta_fc_i, theta_gs, hubs_i, origin_of,
                  errors_i, M):
    """Greedy optimizer for GS valuations. Returns (bundle, utility)."""
    bundle = np.zeros(M, dtype=bool)
    base_marginals = (phi @ theta_rev).ravel() - theta_fc_i + errors_i
    hub_counts = {h: 0 for h in hubs_i}
    base_val = 0.0

    while True:
        marginals = base_marginals.copy()
        if theta_gs > 0:
            for h in hubs_i:
                mask = (~bundle) & (origin_of == h)
                if mask.any():
                    marginals[mask] -= theta_gs * (2 * hub_counts[h] + 1)
        marginals[bundle] = -np.inf

        best_item = np.argmax(marginals)
        best_marginal = marginals[best_item]
        if best_marginal <= 0:
            break
        bundle[best_item] = True
        o = origin_of[best_item]
        if o in hub_counts:
            hub_counts[o] += 1
        base_val += best_marginal

    return bundle, base_val


def make_find_best_item():
    """Vectorized find_best_item for combest GreedySolver.

    theta layout: [theta_rev..., theta_fc_0, ..., theta_fc_{N-1}, theta_gs]

    Caches base marginals and hub masks per agent (via the `cache` dict
    managed by the greedy solver). Each call is O(M) numpy ops.
    """
    def find_best_item(local_id, bundle, items_left, theta, best_val,
                       data, modular_error, cache=None):
        if cache is None:
            cache = {}

        if 'base' not in cache:
            phi = data.item_data['phi']
            origin_of = data.item_data['origin_of']
            obs_ids = data.item_data.get('obs_ids', None)
            obs_idx = int(obs_ids[local_id]) if obs_ids is not None else local_id
            hubs_i = data.id_data['hubs'][obs_idx]
            n_shared = phi.shape[1]

            theta_rev = theta[:n_shared]
            theta_fc_i = theta[n_shared + obs_idx]

            cache['base'] = (phi @ theta_rev).ravel() - theta_fc_i + modular_error
            cache['hub_masks'] = {h: (origin_of == h) for h in hubs_i}
            cache['hub_counts'] = {h: 0 for h in hubs_i}

        base = cache['base']
        hub_masks = cache['hub_masks']
        hub_counts = cache['hub_counts']
        theta_gs = theta[-1]

        marginals = base.copy()
        if theta_gs > 0:
            for h, mask in hub_masks.items():
                marginals[mask] -= theta_gs * (2 * hub_counts[h] + 1)
        marginals[~items_left] = -np.inf

        best_item = int(np.argmax(marginals))
        best_marginal = marginals[best_item]

        if best_marginal <= 0:
            return -1, best_val

        for h, mask in hub_masks.items():
            if mask[best_item]:
                hub_counts[h] += 1
                break

        return best_item, best_val + best_marginal

    return find_best_item


def build_covariates_oracle(N_firms):
    """Covariates oracle with firm-specific fixed costs.

    theta layout: [theta_rev..., theta_fc_0, ..., theta_fc_{N-1}, theta_gs]

    Returns (n_agents, n_shared + N_firms + 1) features per bundle.
    """
    def covariates_oracle(bundles, ids, data):
        phi = data.item_data['phi']
        origin_of = data.item_data['origin_of']
        hubs_list = data.id_data['hubs']
        obs_ids = data.item_data.get('obs_ids', None)
        n_shared = phi.shape[1]

        n_agents = bundles.shape[0]
        n_total = n_shared + N_firms + 1
        features = np.zeros((n_agents, n_total))

        for i_local, idx in enumerate(ids):
            obs_idx = int(obs_ids[idx]) if obs_ids is not None else idx
            b = bundles[i_local]
            features[i_local, :n_shared] = phi[b].sum(axis=0) if b.any() else 0.0
            features[i_local, n_shared + obs_idx] = -float(b.sum())
            hubs_i = hubs_list[obs_idx]
            cong = 0.0
            for h in hubs_i:
                n_h = int((b & (origin_of == h)).sum())
                cong += n_h ** 2
            features[i_local, -1] = -cong

        return features

    return covariates_oracle
