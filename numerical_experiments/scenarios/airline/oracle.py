"""Oracle functions for the airline / gross-substitutes scenario.

Undirected-edge version. For a firm with hubs H_i, edges are partitioned
into three categories:
  - hub-hub edges: both endpoints are hubs, always open (forced to 1)
  - hub-nonhub "spokes": exactly one endpoint is a hub, optimization set
  - nonhub-nonhub: forbidden (always 0)

Congestion counts only spokes, per hub, squared, summed across hubs.

- compute_utility:       V_i(b) for a single airline
- greedy_demand:         GS greedy optimizer (standalone; consistent with
                         the DGP)
- make_find_best_item:   O(M) oracle for combest GreedySolver
- build_covariates_oracle: covariates with firm-specific fixed costs
"""

import numpy as np

from generate_data import edge_categories


def compute_utility(bundle, phi, theta_rev, theta_fc_i, theta_gs, hubs_i,
                    endpoints_a, endpoints_b, errors_i):
    """Full utility V_i(b) for one airline (undirected)."""
    n_active = int(bundle.sum())
    modular = ((phi[bundle] @ theta_rev).sum()
               - theta_fc_i * n_active
               + errors_i[bundle].sum())
    congestion = 0.0
    if theta_gs > 0 and len(hubs_i) > 0:
        _, spoke_mask, _ = edge_categories(hubs_i, endpoints_a, endpoints_b)
        hubs_set = set(hubs_i)
        n_by_hub = {h: 0 for h in hubs_i}
        for j in np.where(bundle & spoke_mask)[0]:
            a, b = int(endpoints_a[j]), int(endpoints_b[j])
            h = a if a in hubs_set else b
            n_by_hub[h] += 1
        congestion = sum(n ** 2 for n in n_by_hub.values())
    return modular - theta_gs * congestion


def greedy_demand(phi, theta_rev, theta_fc_i, theta_gs, hubs_i,
                  endpoints_a, endpoints_b, errors_i, M):
    """Greedy with hub-clique seeded open and non-hub spokes forbidden.

    Returns (bundle, utility).
    """
    hh_mask, spoke_mask, _ = edge_categories(hubs_i, endpoints_a, endpoints_b)
    bundle = hh_mask.copy()

    base_marginals = (phi @ theta_rev).ravel() - theta_fc_i + errors_i

    hubs_set = set(hubs_i)
    hub_counts = {h: 0 for h in hubs_i}
    # Which hub each spoke touches
    spoke_hub = -np.ones(M, dtype=int)
    for j in np.where(spoke_mask)[0]:
        a, b = int(endpoints_a[j]), int(endpoints_b[j])
        spoke_hub[j] = a if a in hubs_set else b

    # Initial utility from the forced-open clique
    base_val = float(base_marginals[hh_mask].sum())

    while True:
        marginals = base_marginals.copy()
        marginals[~spoke_mask] = -np.inf
        marginals[bundle] = -np.inf

        if theta_gs > 0 and len(hubs_i) > 0:
            for h in hubs_i:
                mask = (spoke_hub == h) & ~bundle
                if mask.any():
                    marginals[mask] -= theta_gs * (2 * hub_counts[h] + 1)

        best_item = int(np.argmax(marginals))
        if marginals[best_item] <= 0:
            break
        bundle[best_item] = True
        hub_counts[spoke_hub[best_item]] += 1
        base_val += float(marginals[best_item])

    return bundle, base_val


def make_find_best_item():
    """Vectorized find_best_item for combest GreedySolver.

    Note: the greedy solver starts from an empty bundle, but the DGP's
    hub-clique edges are pre-set. To match, we override the initial
    bundle inside the cache on first call — after which the solver's
    `bundle` argument will reflect the progression including the clique.
    """
    def find_best_item(local_id, bundle, items_left, theta, best_val,
                       data, modular_error, cache=None):
        if cache is None:
            cache = {}

        if 'base' not in cache:
            phi = data.item_data['phi']
            endpoints_a = data.item_data['endpoints_a']
            endpoints_b = data.item_data['endpoints_b']
            obs_ids = data.item_data.get('obs_ids', None)
            obs_idx = int(obs_ids[local_id]) if obs_ids is not None else local_id
            hubs_i = data.id_data['hubs'][obs_idx]
            n_shared = phi.shape[1]

            theta_rev = theta[:n_shared]
            theta_fc_i = theta[n_shared + obs_idx]

            hh_mask, spoke_mask, _ = edge_categories(
                hubs_i, endpoints_a, endpoints_b)

            # Force hub-hub edges open in the caller's bundle and remove
            # them from items_left. (This mutates the arrays the greedy
            # solver reuses across iterations.)
            bundle |= hh_mask
            items_left &= ~hh_mask
            # Also remove forbidden nonhub-nonhub edges
            items_left &= spoke_mask

            # Cache
            cache['base'] = ((phi @ theta_rev).ravel()
                              - theta_fc_i + modular_error)
            cache['spoke_mask'] = spoke_mask
            cache['hh_mask'] = hh_mask
            # Spoke -> hub index
            hubs_set = set(hubs_i)
            M = phi.shape[0]
            spoke_hub = -np.ones(M, dtype=int)
            for j in np.where(spoke_mask)[0]:
                a, b = int(endpoints_a[j]), int(endpoints_b[j])
                spoke_hub[j] = a if a in hubs_set else b
            cache['spoke_hub'] = spoke_hub
            cache['hubs'] = list(hubs_i)
            cache['hub_counts'] = {h: 0 for h in hubs_i}

            # Include clique utility in best_val (the combest greedy
            # starts tracking from 0; we need to credit the clique so
            # future marginals are comparable).
            clique_util = float(cache['base'][hh_mask].sum())
            best_val = best_val + clique_util
            cache['_clique_credit'] = clique_util

        base = cache['base']
        spoke_mask = cache['spoke_mask']
        spoke_hub = cache['spoke_hub']
        hub_counts = cache['hub_counts']
        theta_gs = theta[-1]

        marginals = base.copy()
        marginals[~spoke_mask] = -np.inf
        marginals[~items_left] = -np.inf
        marginals[bundle] = -np.inf

        if theta_gs > 0 and len(cache['hubs']) > 0:
            for h in cache['hubs']:
                mask = (spoke_hub == h) & ~bundle
                if mask.any():
                    marginals[mask] -= theta_gs * (2 * hub_counts[h] + 1)

        best_item = int(np.argmax(marginals))
        best_marginal = marginals[best_item]

        if best_marginal <= 0:
            return -1, best_val

        hub_counts[spoke_hub[best_item]] += 1
        return best_item, best_val + best_marginal

    return find_best_item


def build_covariates_oracle(N_firms):
    """Covariates oracle with firm-specific fixed costs (undirected)."""
    def covariates_oracle(bundles, ids, data):
        phi = data.item_data['phi']
        endpoints_a = data.item_data['endpoints_a']
        endpoints_b = data.item_data['endpoints_b']
        hubs_list = data.id_data['hubs']
        obs_ids = data.item_data.get('obs_ids', None)
        n_shared = phi.shape[1]

        n_agents = bundles.shape[0]
        n_total = n_shared + N_firms + 1
        features = np.zeros((n_agents, n_total))

        for i_local, idx in enumerate(ids):
            obs_idx = int(obs_ids[idx]) if obs_ids is not None else idx
            b = bundles[i_local]
            features[i_local, :n_shared] = (phi[b].sum(axis=0)
                                            if b.any() else 0.0)
            features[i_local, n_shared + obs_idx] = -float(b.sum())
            hubs_i = hubs_list[obs_idx]
            # Congestion: only count hub-nonhub spokes, per hub, squared
            if len(hubs_i) > 0:
                _, spoke_mask, _ = edge_categories(
                    hubs_i, endpoints_a, endpoints_b)
                hubs_set = set(hubs_i)
                n_by_hub = {h: 0 for h in hubs_i}
                for j in np.where(b & spoke_mask)[0]:
                    a_, b_ = int(endpoints_a[j]), int(endpoints_b[j])
                    h = a_ if a_ in hubs_set else b_
                    n_by_hub[h] += 1
                cong = sum(n ** 2 for n in n_by_hub.values())
            else:
                cong = 0.0
            features[i_local, -1] = -cong

        return features

    return covariates_oracle
