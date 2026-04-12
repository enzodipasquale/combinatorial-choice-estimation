"""DGP for airline / gross-substitutes scenario.

Generates city geography, populations, airline hubs, route covariates,
and runs healthy-theta search to find an informative true parameter.
"""

import numpy as np
from combest.utils import get_logger

logger = get_logger(__name__)


def build_geography(C, rng):
    """Draw C cities with random locations and log-normal populations."""
    locations = rng.uniform(0, 1, (C, 2))
    dists = np.sqrt(((locations[:, None] - locations[None, :]) ** 2).sum(-1))
    log_pop = rng.normal(0, 0.5, C)
    populations = np.exp(log_pop)
    return locations, dists, populations


def build_edges(C):
    """Build directed edge list for the complete directed graph on C cities.

    Returns:
        edges: list of (origin, dest) tuples, length M = C(C-1)
        origin_of: (M,) array of origin city indices
        dest_of:   (M,) array of dest city indices
    """
    edges = [(o, d) for o in range(C) for d in range(C) if o != d]
    M = len(edges)
    origin_of = np.array([e[0] for e in edges])
    dest_of = np.array([e[1] for e in edges])
    return edges, origin_of, dest_of, M


def build_hubs(N, C, rng):
    """Draw hub sets for N airlines. Each airline gets 1-3 hubs."""
    hubs = []
    for _ in range(N):
        n_hubs = rng.integers(1, min(3, C) + 1)  # 1 to min(3, C)
        hub_set = set(rng.choice(C, size=n_hubs, replace=False).tolist())
        hubs.append(hub_set)
    return hubs


def build_covariates(C, M, origin_of, dest_of, dists, populations, fe_mode):
    """Build covariate matrix phi: (M, C_mod).

    fe_mode="none":   2 covariates: log(pop_o * pop_d), -distance
    fe_mode="origin": 2 + C covariates: above + origin one-hot
    """
    log_pop_product = np.log(populations[origin_of] * populations[dest_of])
    neg_distance = -dists[origin_of, dest_of]
    phi = np.column_stack([log_pop_product, neg_distance])

    if fe_mode == "origin":
        origin_fe = np.zeros((M, C))
        origin_fe[np.arange(M), origin_of] = 1.0
        phi = np.column_stack([phi, origin_fe])

    return phi


def n_covariates(C, fe_mode):
    """Number of modular covariates (excludes theta_gs)."""
    if fe_mode == "none":
        return 2
    elif fe_mode == "origin":
        return 2 + C
    else:
        raise ValueError(f"Unknown fe_mode: {fe_mode}")


def n_params(C, fe_mode):
    """Total number of parameters: C_mod + 1 (for theta_gs)."""
    return n_covariates(C, fe_mode) + 1


def compute_utility(bundle, phi, theta_mod, theta_gs, hubs_i, origin_of, errors_i):
    """Compute utility V_i(b) for a single airline.

    Args:
        bundle:    (M,) bool
        phi:       (M, C_mod)
        theta_mod: (C_mod,)
        theta_gs:  scalar >= 0
        hubs_i:    set of hub city indices
        origin_of: (M,) origin city of each edge
        errors_i:  (M,) modular errors nu_{ij}

    Returns:
        scalar utility
    """
    modular = (phi[bundle] @ theta_mod).sum() + errors_i[bundle].sum()
    congestion = 0.0
    if theta_gs > 0:
        for h in hubs_i:
            n_h = (bundle & (origin_of == h)).sum()
            congestion += n_h ** 2
    return modular - theta_gs * congestion


def greedy_demand(phi, theta_mod, theta_gs, hubs_i, origin_of, errors_i, M):
    """Compute optimal bundle via greedy algorithm (GS property).

    Returns (M,) bool array. Vectorized over items for speed.
    """
    bundle = np.zeros(M, dtype=bool)
    # Base modular values (fixed across iterations)
    base_marginals = phi @ theta_mod + errors_i

    # Hub mask: is_hub[j] = True if origin_of[j] is a hub
    hub_set = set(hubs_i)
    is_hub = np.array([origin_of[j] in hub_set for j in range(M)])
    hub_counts = {h: 0 for h in hubs_i}

    while True:
        # Compute marginal values for all remaining items
        marginals = base_marginals.copy()
        # Apply congestion penalty for hub origins
        if theta_gs > 0:
            for h in hubs_i:
                mask = (~bundle) & (origin_of == h)
                if mask.any():
                    marginals[mask] -= theta_gs * (2 * hub_counts[h] + 1)
        marginals[bundle] = -np.inf  # exclude already-selected items

        best_item = np.argmax(marginals)
        if marginals[best_item] <= 0:
            break
        bundle[best_item] = True
        o = origin_of[best_item]
        if o in hub_counts:
            hub_counts[o] += 1

    return bundle


def check_healthy_dgp(bundles, M, N, C, fe_mode, origin_of, hubs):
    """Check all healthy-DGP criteria. Returns (ok, diagnostics)."""
    sizes = np.array([b.sum() for b in bundles], dtype=float)
    mean_size = sizes.mean()
    std_size = sizes.std()

    diag = {
        "mean_bundle_frac": mean_size / M,
        "std_bundle_frac": std_size / M,
        "min_bundle_size": int(sizes.min()),
        "max_bundle_size": int(sizes.max()),
    }

    ok = True
    # Criterion 1: mean bundle size in (0.1M, 0.55M), no empty or full
    if not (0.1 * M < mean_size < 0.55 * M):
        ok = False
    if sizes.min() == 0 or sizes.max() == M:
        ok = False

    # Criterion 2: cross-airline heterogeneity in bundle size.
    # At small M use 0.05*M; at large M scale as 0.01*M (hub-driven
    # size heterogeneity grows sub-linearly with M).
    std_threshold = min(0.05 * M, max(0.01 * M, 5.0))
    if std_size < std_threshold:
        ok = False

    # Criterion 3: origin FE identification (only under fe_mode="origin")
    if fe_mode == "origin":
        utilization = np.zeros(C)
        for i, b in enumerate(bundles):
            origins_used = set(origin_of[b].tolist())
            for o in origins_used:
                utilization[o] += 1
        utilization /= N
        diag["min_origin_util"] = float(utilization.min())
        diag["max_origin_util"] = float(utilization.max())
        if utilization.min() <= 0 or utilization.max() >= 1:
            ok = False

    # Criterion 4: congestion variation — theta_gs must be identifiable.
    # At least 50% of airlines must have >=1 route from a hub, AND
    # the congestion feature must have std > 0.
    hub_route_counts = []
    for i, b in enumerate(bundles):
        cong = 0
        for h in hubs[i]:
            n_h = int((b & (origin_of == h)).sum())
            cong += n_h ** 2
        hub_route_counts.append(cong)
    hub_route_counts = np.array(hub_route_counts, dtype=float)
    frac_with_hub_routes = (hub_route_counts > 0).mean()
    cong_std = hub_route_counts.std()
    diag["frac_with_hub_routes"] = float(frac_with_hub_routes)
    diag["congestion_std"] = float(cong_std)
    if frac_with_hub_routes < 0.5 or cong_std < 0.1:
        ok = False

    diag["healthy"] = ok
    return ok, diag


def search_healthy_theta(C, N, fe_mode, sigma, dgp_seed, search_seed,
                         max_candidates=200, verbose=True):
    """Search for a theta* that produces healthy DGP bundles.

    Returns (theta_star, bundles, dgp_data, diagnostics) or raises if not found.
    """
    rng_dgp = np.random.default_rng(dgp_seed)
    rng_search = np.random.default_rng(search_seed)

    _, origin_of, dest_of, M = build_edges(C)
    locations, dists, populations = build_geography(C, rng_dgp)
    phi = build_covariates(C, M, origin_of, dest_of, dists, populations, fe_mode)
    hubs = build_hubs(N, C, rng_dgp)

    C_mod = n_covariates(C, fe_mode)

    # Draw modular errors (fixed for the search)
    rng_err = np.random.default_rng(dgp_seed + 999)
    errors = rng_err.normal(0, sigma, (N, M))

    # Calibrate search: compute std of each covariate column so we can
    # scale theta candidates to produce route values of order O(sigma).
    phi_stds = np.maximum(phi.std(axis=0), 1e-8)

    best_theta, best_bundles, best_diag, best_score = None, None, None, -np.inf

    for trial in range(max_candidates):
        theta_gs = rng_search.uniform(0.5, 5.0)
        theta_mod = np.zeros(C_mod)
        # Scale each coefficient so its contribution has std ~ O(sigma).
        # phi[:,1] = -distance (negative), so theta_mod[1] > 0 makes
        # longer routes less valuable (distance acts as cost).
        scale = sigma / phi_stds[:2]
        theta_mod[0] = rng_search.uniform(0.1, 2.0) * scale[0]   # positive (revenue from pop)
        theta_mod[1] = rng_search.uniform(0.1, 3.0) * scale[1]   # positive (cost from distance)
        if fe_mode == "origin":
            theta_mod[2:] = rng_search.uniform(-1.0, 1.0, C)

        bundles = []
        for i in range(N):
            b = greedy_demand(phi, theta_mod, theta_gs, hubs[i], origin_of, errors[i], M)
            bundles.append(b)

        ok, diag = check_healthy_dgp(bundles, M, N, C, fe_mode, origin_of, hubs)

        # Score for picking best even if none is strictly healthy
        score = 0.0
        mf = diag["mean_bundle_frac"]
        sf = diag["std_bundle_frac"]
        if 0.1 < mf < 0.5:
            score += 1.0
        else:
            score -= abs(mf - 0.3)
        if sf >= 0.05:
            score += 1.0
        else:
            score -= (0.05 - sf)
        if diag["min_bundle_size"] > 0:
            score += 0.5
        if diag["max_bundle_size"] < M:
            score += 0.5
        if diag.get("frac_with_hub_routes", 0) >= 0.5:
            score += 1.0
        if diag.get("congestion_std", 0) > 0.1:
            score += 0.5

        if ok:
            if verbose:
                logger.info(f"  Healthy theta found at trial {trial}: "
                            f"theta_gs={theta_gs:.3f}, theta_mod={theta_mod}, diag={diag}")
            theta_star = np.concatenate([theta_mod, [theta_gs]])
            dgp_data = {
                "locations": locations, "dists": dists, "populations": populations,
                "phi": phi, "hubs": hubs, "errors": errors,
                "origin_of": origin_of, "dest_of": dest_of,
                "C": C, "M": M, "N": N, "fe_mode": fe_mode,
            }
            return theta_star, bundles, dgp_data, diag

        if score > best_score:
            best_score = score
            best_theta = np.concatenate([theta_mod, [theta_gs]])
            best_bundles = bundles
            best_diag = diag

    raise RuntimeError(
        f"Healthy-DGP search failed after {max_candidates} candidates. "
        f"Best score={best_score:.3f}, best diag={best_diag}"
    )


def generate_data(C, N, fe_mode, sigma, dgp_seed, search_seed,
                  max_candidates=200, verbose=True):
    """Top-level DGP entry point.

    Returns:
        theta_star: (n_params,) true parameter vector [theta_mod | theta_gs]
        obs_bundles: (N, M) bool observed choices
        dgp_data: dict with phi, hubs, origin_of, dest_of, errors, etc.
        diagnostics: dict with healthy-DGP check results
    """
    theta_star, bundles, dgp_data, diag = search_healthy_theta(
        C, N, fe_mode, sigma, dgp_seed, search_seed,
        max_candidates=max_candidates, verbose=verbose)

    obs_bundles = np.stack(bundles)
    return theta_star, obs_bundles, dgp_data, diag


if __name__ == "__main__":
    # Step 1 check: C=4, N=3, inspect covariates and hubs
    C, N = 4, 3
    for fm in ["none", "origin"]:
        print(f"\n{'='*60}")
        print(f"  fe_mode = {fm}, C={C}, N={N}")
        print(f"{'='*60}")

        rng = np.random.default_rng(42)
        locations, dists, populations = build_geography(C, rng)
        edges, origin_of, dest_of, M = build_edges(C)
        phi = build_covariates(C, M, origin_of, dest_of, dists, populations, fm)
        hubs = build_hubs(N, C, rng)

        print(f"  M = {M} edges")
        print(f"  phi shape = {phi.shape}  (expected ({M}, {n_covariates(C, fm)}))")
        print(f"  Populations: {populations}")
        print(f"  Hubs: {hubs}")
        print(f"  Covariate matrix (first 5 rows):")
        print(phi[:5])
        print(f"  Hub cities all in [0, {C}): ",
              all(h < C for hs in hubs for h in hs))
