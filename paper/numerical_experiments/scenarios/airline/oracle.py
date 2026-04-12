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

def compute_utility(bundle, phi, theta_mod, theta_gs, hubs_i, origin_of, errors_i):
    """Compute V_i(b) for one airline.

    Args:
        bundle:    (M,) bool
        phi:       (M, n_mod)
        theta_mod: (n_mod,)
        theta_gs:  scalar >= 0
        hubs_i:    set of hub city indices for this airline
        origin_of: (M,) origin city of each edge
        errors_i:  (M,) modular errors nu_{ij}

    Returns:
        scalar utility
    """
    modular = (phi[bundle] @ theta_mod).sum() + errors_i[bundle].sum()
    congestion = 0.0
    if theta_gs > 0:
        for h in hubs_i:
            n_h = int((bundle & (origin_of == h)).sum())
            congestion += n_h ** 2
    return modular - theta_gs * congestion


# ---------------------------------------------------------------------------
# Brute-force demand (Step 3)
# ---------------------------------------------------------------------------

def brute_force_demand(phi, theta_mod, theta_gs, hubs_i, origin_of, errors_i, M):
    """Enumerate all 2^M bundles, return the one with highest utility."""
    best_val = float('-inf')
    best_bundle = None
    for bits in cartesian_product([False, True], repeat=M):
        b = np.array(bits, dtype=bool)
        val = compute_utility(b, phi, theta_mod, theta_gs, hubs_i, origin_of, errors_i)
        if val > best_val:
            best_val = val
            best_bundle = b.copy()
    return best_bundle, best_val


# ---------------------------------------------------------------------------
# Greedy demand (Step 4 — standalone, not wired to combest)
# ---------------------------------------------------------------------------

def greedy_demand(phi, theta_mod, theta_gs, hubs_i, origin_of, errors_i, M):
    """Greedy oracle for GS valuations. Returns (bundle, utility)."""
    bundle = np.zeros(M, dtype=bool)
    base_marginals = phi @ theta_mod + errors_i
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

    Uses the `cache` dict (managed by the greedy solver, one per agent)
    to store precomputed base marginals and hub masks across iterations.

    Each iteration is O(M) NumPy ops — no Python loops over items.
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
            theta_mod = theta[:-1]

            cache['base'] = phi @ theta_mod + modular_error   # (M,)
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

def build_covariates_oracle():
    """Build covariates oracle for combest FeaturesManager.

    covariates(b) = [sum_j b_j phi_j,  -sum_h (n_o^h)^2]
    So covariates @ theta = modular_part - theta_gs * congestion.
    """
    def covariates_oracle(bundles, ids, data):
        phi = data.item_data['phi']
        origin_of = data.item_data['origin_of']
        hubs_list = data.id_data['hubs']
        n_mod = phi.shape[1]

        n_agents = bundles.shape[0]
        n_total = n_mod + 1  # modular covariates + congestion
        features = np.zeros((n_agents, n_total))

        for i_local, idx in enumerate(ids):
            b = bundles[i_local]
            # Modular part: sum_j b_j phi_j
            features[i_local, :n_mod] = phi[b].sum(axis=0) if b.any() else 0.0
            # Congestion part: -sum_h n_h^2
            hubs_i = hubs_list[idx]
            cong = 0.0
            for h in hubs_i:
                n_h = int((b & (origin_of == h)).sum())
                cong += n_h ** 2
            features[i_local, n_mod] = -cong

        return features

    return covariates_oracle


# ---------------------------------------------------------------------------
# Step 2 self-check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
    from generate_data import build_geography, build_edges, build_covariates, build_hubs

    C, N = 4, 3
    rng = np.random.default_rng(42)
    locations, dists, populations = build_geography(C, rng)
    edges, origin_of, dest_of, M = build_edges(C)
    phi = build_covariates(C, M, origin_of, dest_of, dists, populations, "none")
    hubs = build_hubs(N, C, populations, rng)
    errors = rng.normal(0, 1.0, (N, M))

    theta_mod = np.array([0.5, 0.3])  # [theta_rev, theta_fc]
    theta_gs = 1.0

    print(f"C={C}, M={M}, N={N}")
    print(f"theta_mod={theta_mod}, theta_gs={theta_gs}")

    # Three bundles: empty, full, arbitrary
    for label, b in [("empty", np.zeros(M, dtype=bool)),
                     ("full", np.ones(M, dtype=bool)),
                     ("arbitrary", np.array([True,False,True,True,False,False,
                                             True,False,False,True,False,True]))]:
        for i in range(N):
            val = compute_utility(b, phi, theta_mod, theta_gs, hubs[i], origin_of, errors[i])
            print(f"  V_{i}({label}) = {val:.4f}")

    # Hand-verify congestion for arbitrary bundle, airline 0
    b = np.array([True,False,True,True,False,False,True,False,False,True,False,True])
    i = 0
    print(f"\n  Hand-verify congestion for airline {i}, hubs={hubs[i]}:")
    for h in hubs[i]:
        n_h = (b & (origin_of == h)).sum()
        print(f"    hub {h}: n_h={n_h}, n_h^2={n_h**2}")
    cong_total = sum((b & (origin_of == h)).sum()**2 for h in hubs[i])
    mod_total = (phi[b] @ theta_mod).sum() + errors[i][b].sum()
    print(f"    modular={mod_total:.4f}, congestion={theta_gs}*{cong_total}={theta_gs*cong_total:.4f}")
    print(f"    V = {mod_total - theta_gs*cong_total:.4f}")
