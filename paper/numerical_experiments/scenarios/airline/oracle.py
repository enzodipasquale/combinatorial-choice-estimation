"""Oracle functions for airline / gross-substitutes scenario.

Provides:
- compute_utility:  full utility V_i(b) for a single airline
- brute_force_demand: enumerate all 2^M bundles, return maximizer
- find_best_item:  O(M) marginal-value oracle for greedy solver
- build_covariates_oracle / build_error_oracle: for combest integration
"""

import numpy as np
from itertools import product as cartesian_product


# ---------------------------------------------------------------------------
# Utility (Step 2)
# ---------------------------------------------------------------------------

def compute_utility(bundle, phi, theta_rev, theta_fc_i, theta_gs, hubs_i,
                    origin_of, errors_i):
    """Compute V_i(b) for one airline.

    Args:
        bundle:     (M,) bool
        phi:        (M, n_shared) shared covariates
        theta_rev:  (n_shared,) shared coefficients
        theta_fc_i: scalar, this firm's per-route fixed cost
        theta_gs:   scalar >= 0
        hubs_i:     set of hub city indices
        origin_of:  (M,) origin city of each edge
        errors_i:   (M,) modular errors

    Returns:
        scalar utility
    """
    n_active = bundle.sum()
    modular = (phi[bundle] @ theta_rev).sum() - theta_fc_i * n_active + errors_i[bundle].sum()
    congestion = 0.0
    if theta_gs > 0:
        for h in hubs_i:
            n_h = int((bundle & (origin_of == h)).sum())
            congestion += n_h ** 2
    return modular - theta_gs * congestion


# ---------------------------------------------------------------------------
# Brute-force demand (Step 3)
# ---------------------------------------------------------------------------

def brute_force_demand(phi, theta_rev, theta_fc_i, theta_gs, hubs_i, origin_of,
                       errors_i, M):
    """Enumerate all 2^M bundles, return the one with highest utility."""
    best_val = float('-inf')
    best_bundle = None
    for bits in cartesian_product([False, True], repeat=M):
        b = np.array(bits, dtype=bool)
        val = compute_utility(b, phi, theta_rev, theta_fc_i, theta_gs, hubs_i, origin_of, errors_i)
        if val > best_val:
            best_val = val
            best_bundle = b.copy()
    return best_bundle, best_val


# ---------------------------------------------------------------------------
# Greedy demand (Step 4 — standalone, not wired to combest)
# ---------------------------------------------------------------------------

def greedy_demand(phi, theta_rev, theta_fc_i, theta_gs, hubs_i, origin_of,
                  errors_i, M):
    """Greedy oracle for GS valuations. Returns (bundle, utility)."""
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


# ---------------------------------------------------------------------------
# Custom find_best_item for combest GreedySolver (Step 5)
# ---------------------------------------------------------------------------

def make_find_best_item():
    """Return a vectorized find_best_item closure for GreedySolver.

    theta layout: [theta_rev..., theta_fc_0, ..., theta_fc_{N-1}, theta_gs]

    Uses the `cache` dict (managed by the greedy solver, one per agent)
    to store precomputed base marginals and hub masks across iterations.
    """
    def find_best_item(local_id, bundle, items_left, theta, best_val,
                       data, modular_error, cache=None):
        if cache is None:
            cache = {}

        # First call for this agent: precompute and cache
        if 'base' not in cache:
            phi = data.item_data['phi']
            origin_of = data.item_data['origin_of']
            hubs_i = data.id_data['hubs'][local_id]
            n_shared = phi.shape[1]
            N_firms = data.item_data['N_firms']

            theta_rev = theta[:n_shared]
            theta_fc_i = theta[n_shared + local_id]

            # base marginal = revenue - firm FC + error (per route)
            cache['base'] = (phi @ theta_rev).ravel() - theta_fc_i + modular_error
            cache['hub_masks'] = {h: (origin_of == h) for h in hubs_i}
            cache['hub_counts'] = {h: 0 for h in hubs_i}

        base = cache['base']
        hub_masks = cache['hub_masks']
        hub_counts = cache['hub_counts']
        theta_gs = theta[-1]

        # Vectorized marginals: start from base, apply hub penalties
        marginals = base.copy()
        if theta_gs > 0:
            for h, mask in hub_masks.items():
                marginals[mask] -= theta_gs * (2 * hub_counts[h] + 1)

        # Mask out already-selected items
        marginals[~items_left] = -np.inf

        best_item = int(np.argmax(marginals))
        best_marginal = marginals[best_item]

        if best_marginal <= 0:
            return -1, best_val

        # Update cached hub count for the selected item
        for h, mask in hub_masks.items():
            if mask[best_item]:
                hub_counts[h] += 1
                break

        return best_item, best_val + best_marginal

    return find_best_item


# ---------------------------------------------------------------------------
# Combest oracles (for estimation)
# ---------------------------------------------------------------------------

def build_covariates_oracle(N_firms):
    """Build covariates oracle with firm-specific fixed costs.

    theta layout: [theta_rev..., theta_fc_0, ..., theta_fc_{N-1}, theta_gs]

    For agent idx with bundle b, features are:
      [sum_j b_j phi_j, ..., 0, ..., -sum(b), ..., 0, ..., -congestion]
                              ^-- col n_shared + idx = -bundle_size
    """
    def covariates_oracle(bundles, ids, data):
        phi = data.item_data['phi']
        origin_of = data.item_data['origin_of']
        hubs_list = data.id_data['hubs']
        n_shared = phi.shape[1]

        n_agents = bundles.shape[0]
        # n_shared + N_firms (firm FCs) + 1 (congestion)
        n_total = n_shared + N_firms + 1
        features = np.zeros((n_agents, n_total))

        for i_local, idx in enumerate(ids):
            b = bundles[i_local]
            # Shared modular part: sum_j b_j phi_j
            features[i_local, :n_shared] = phi[b].sum(axis=0) if b.any() else 0.0
            # Firm-specific fixed cost: -bundle_size in this firm's column
            features[i_local, n_shared + idx] = -float(b.sum())
            # Congestion: -sum_h n_h^2
            hubs_i = hubs_list[idx]
            cong = 0.0
            for h in hubs_i:
                n_h = int((b & (origin_of == h)).sum())
                cong += n_h ** 2
            features[i_local, -1] = -cong

        return features

    return covariates_oracle
