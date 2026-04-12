"""DGP for airline / gross-substitutes scenario.

Generates city geography, populations, airline hubs, route covariates,
and runs healthy-theta search to find an informative true parameter.
"""

import numpy as np
from combest.utils import get_logger

logger = get_logger(__name__)


def build_geography(C, pop_log_std, rng):
    """Draw C cities with random locations and log-normal populations."""
    locations = rng.uniform(0, 1, (C, 2))
    dists = np.sqrt(((locations[:, None] - locations[None, :]) ** 2).sum(-1))
    log_pop = rng.normal(0, pop_log_std, C)
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


def build_hubs(N, C, populations, min_hubs, max_hubs, hub_pool_frac, rng):
    """Assign hub sets for N airlines from the largest cities."""
    hub_pool_size = max(max_hubs, int(C * hub_pool_frac))
    hub_pool = np.argsort(populations)[-hub_pool_size:]
    hubs = []
    for _ in range(N):
        n_hubs = rng.integers(min_hubs, min(max_hubs, len(hub_pool)) + 1)
        hub_set = set(rng.choice(hub_pool, size=n_hubs, replace=False).tolist())
        hubs.append(hub_set)
    return hubs


def build_covariates(C, M, origin_of, dest_of, dists, populations, fe_mode):
    """Build modular covariate matrix phi: (M, n_mod).

    Columns (fe_mode="none"):
      0: gravity revenue = pop_o * pop_d / dist
      1: -1  (per-route fixed cost indicator)

    With fe_mode="origin", origin one-hot columns are appended.
    """
    d = dists[origin_of, dest_of]
    d_safe = np.maximum(d, 1e-6)
    gravity = populations[origin_of] * populations[dest_of] / d_safe
    fc_indicator = -np.ones(M)
    phi = np.column_stack([gravity, fc_indicator])

    if fe_mode == "origin":
        origin_fe = np.zeros((M, C))
        origin_fe[np.arange(M), origin_of] = 1.0
        phi = np.column_stack([phi, origin_fe])

    return phi


def n_modular(C, fe_mode):
    """Number of modular covariates (gravity + fixed cost + FEs)."""
    base = 2  # gravity, -1
    if fe_mode == "none":
        return base
    elif fe_mode == "origin":
        return base + C
    else:
        raise ValueError(f"Unknown fe_mode: {fe_mode}")


def n_params(C, fe_mode):
    """Total parameters: n_modular + 1 (theta_gs)."""
    return n_modular(C, fe_mode) + 1


def compute_utility(bundle, phi, theta_mod, theta_gs, hubs_i, origin_of, errors_i):
    """Compute utility V_i(b) for a single airline."""
    modular = (phi[bundle] @ theta_mod).sum() + errors_i[bundle].sum()
    congestion = 0.0
    if theta_gs > 0:
        for h in hubs_i:
            n_h = (bundle & (origin_of == h)).sum()
            congestion += n_h ** 2
    return modular - theta_gs * congestion


def greedy_demand(phi, theta_mod, theta_gs, hubs_i, origin_of, errors_i, M):
    """Compute optimal bundle via greedy algorithm (GS property).

    Vectorized over items for speed. Returns (M,) bool array.
    """
    bundle = np.zeros(M, dtype=bool)
    base_marginals = phi @ theta_mod + errors_i
    hub_counts = {h: 0 for h in hubs_i}

    while True:
        marginals = base_marginals.copy()
        if theta_gs > 0:
            for h in hubs_i:
                mask = (~bundle) & (origin_of == h)
                if mask.any():
                    marginals[mask] -= theta_gs * (2 * hub_counts[h] + 1)
        marginals[bundle] = -np.inf

        best_item = np.argmax(marginals)
        if marginals[best_item] <= 0:
            break
        bundle[best_item] = True
        o = origin_of[best_item]
        if o in hub_counts:
            hub_counts[o] += 1

    return bundle


def check_healthy_dgp(bundles, M, N, C, fe_mode, origin_of, hubs, healthy_cfg):
    """Check all healthy-DGP criteria. Returns (ok, diagnostics)."""
    sizes = np.array([b.sum() for b in bundles], dtype=float)
    mean_size = sizes.mean()
    std_size = sizes.std()

    min_mf = healthy_cfg['min_mean_bundle_frac']
    max_mf = healthy_cfg['max_mean_bundle_frac']
    min_frac_hub = healthy_cfg['min_frac_with_hub_routes']
    min_cong_std = healthy_cfg['min_congestion_std']

    diag = {
        "mean_bundle_frac": mean_size / M,
        "std_bundle_frac": std_size / M,
        "min_bundle_size": int(sizes.min()),
        "max_bundle_size": int(sizes.max()),
    }

    ok = True
    if not (min_mf * M < mean_size < max_mf * M):
        ok = False
    if sizes.min() == 0 or sizes.max() == M:
        ok = False

    # Cross-airline heterogeneity (scales sub-linearly with M)
    std_threshold = min(0.05 * M, max(0.008 * M, 5.0))
    if std_size < std_threshold:
        ok = False

    # Origin FE identification
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

    # Congestion variation
    hub_route_counts = []
    for i, b in enumerate(bundles):
        cong = sum(int((b & (origin_of == h)).sum()) ** 2 for h in hubs[i])
        hub_route_counts.append(cong)
    hub_route_counts = np.array(hub_route_counts, dtype=float)
    frac_with_hub_routes = (hub_route_counts > 0).mean()
    cong_std = hub_route_counts.std()
    diag["frac_with_hub_routes"] = float(frac_with_hub_routes)
    diag["congestion_std"] = float(cong_std)
    if frac_with_hub_routes < min_frac_hub or cong_std < min_cong_std:
        ok = False

    diag["healthy"] = ok
    return ok, diag


def search_healthy_theta(dgp_cfg, healthy_cfg, seeds, verbose=True):
    """Search for a theta* that produces healthy DGP bundles.

    Returns (theta_star, bundles, dgp_data, diagnostics) or raises.
    """
    C = dgp_cfg['C']
    N = dgp_cfg['N']
    fe_mode = dgp_cfg['fe_mode']
    sigma = dgp_cfg['sigma']
    pop_log_std = dgp_cfg['pop_log_std']
    hub_pool_frac = dgp_cfg['hub_pool_frac']
    min_hubs = dgp_cfg['min_hubs']
    max_hubs = dgp_cfg['max_hubs']

    max_candidates = healthy_cfg['max_candidates']
    theta_rev_range = healthy_cfg['theta_rev_range']
    theta_fc_range = healthy_cfg['theta_fc_range']
    theta_gs_range = healthy_cfg['theta_gs_range']

    rng_dgp = np.random.default_rng(seeds['dgp'])
    rng_search = np.random.default_rng(seeds['search'])

    _, origin_of, dest_of, M = build_edges(C)
    locations, dists, populations = build_geography(C, pop_log_std, rng_dgp)
    phi = build_covariates(C, M, origin_of, dest_of, dists, populations, fe_mode)
    hubs = build_hubs(N, C, populations, min_hubs, max_hubs, hub_pool_frac, rng_dgp)

    n_mod = n_modular(C, fe_mode)

    rng_err = np.random.default_rng(seeds['dgp'] + 999)
    errors = rng_err.normal(0, sigma, (N, M))

    gravity_std = max(phi[:, 0].std(), 1e-8)

    best_theta, best_bundles, best_diag, best_score = None, None, None, -np.inf

    for trial in range(max_candidates):
        theta_gs = rng_search.uniform(*theta_gs_range)
        theta_mod = np.zeros(n_mod)
        theta_mod[0] = rng_search.uniform(*theta_rev_range) * sigma / gravity_std
        theta_mod[1] = rng_search.uniform(*theta_fc_range) * sigma
        if fe_mode == "origin":
            theta_mod[2:] = rng_search.uniform(-1.0, 1.0, C)

        bundles = []
        for i in range(N):
            b = greedy_demand(phi, theta_mod, theta_gs, hubs[i], origin_of, errors[i], M)
            bundles.append(b)

        ok, diag = check_healthy_dgp(bundles, M, N, C, fe_mode, origin_of, hubs,
                                     healthy_cfg)

        score = 0.0
        mf = diag["mean_bundle_frac"]
        if 0.1 < mf < 0.5:
            score += 1.0
        else:
            score -= abs(mf - 0.3)
        if diag["std_bundle_frac"] * M >= min(0.05 * M, max(0.008 * M, 5.0)):
            score += 1.0
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
                            f"theta_gs={theta_gs:.3f}, theta_mod={theta_mod}, "
                            f"diag={diag}")
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


def generate_data(dgp_cfg, healthy_cfg, seeds, verbose=True):
    """Top-level DGP entry point.

    Returns:
        theta_star, obs_bundles, dgp_data, diagnostics
    """
    theta_star, bundles, dgp_data, diag = search_healthy_theta(
        dgp_cfg, healthy_cfg, seeds, verbose=verbose)

    obs_bundles = np.stack(bundles)
    return theta_star, obs_bundles, dgp_data, diag
