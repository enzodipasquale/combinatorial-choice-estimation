"""DGP for airline / gross-substitutes scenario.

Generates city geography, populations, airline hubs, route covariates,
and runs healthy-theta search to find an informative true parameter.
"""

import numpy as np
from combest.utils import get_logger

logger = get_logger(__name__)


def build_geography(C, pop_log_std, rng, pop_dist="lognormal",
                    pareto_alpha=1.5):
    """Draw C cities with random locations and populations.

    Args:
        pop_dist: "lognormal" (default) or "pareto".
        pop_log_std: used when pop_dist == "lognormal".
        pareto_alpha: shape parameter for Pareto (smaller -> heavier tail).
                      Only used when pop_dist == "pareto".
    """
    locations = rng.uniform(0, 1, (C, 2))
    dists = np.sqrt(((locations[:, None] - locations[None, :]) ** 2).sum(-1))
    if pop_dist == "lognormal":
        log_pop = rng.normal(0, pop_log_std, C)
        populations = np.exp(log_pop)
    elif pop_dist == "pareto":
        # numpy's pareto samples from 1 + Pareto(alpha)-1 (Lomax); shift so min = 1
        populations = 1.0 + rng.pareto(pareto_alpha, C)
    else:
        raise ValueError(f"Unknown pop_dist={pop_dist}")
    return locations, dists, populations


def build_edges(C):
    """Build UNDIRECTED edge list on C cities.

    Returns:
        edges: list of (a, b) tuples with a<b, length M = C(C-1)/2
        endpoints_a: (M,) first endpoint (smaller index)
        endpoints_b: (M,) second endpoint (larger index)
        M: number of edges
    """
    edges = [(a, b) for a in range(C) for b in range(a + 1, C)]
    M = len(edges)
    endpoints_a = np.array([e[0] for e in edges])
    endpoints_b = np.array([e[1] for e in edges])
    return edges, endpoints_a, endpoints_b, M


def edge_categories(hubs_i, endpoints_a, endpoints_b):
    """Partition edges into three categories for airline i.

    Returns three boolean masks over edges, each of shape (M,):
        hub_hub_mask:     both endpoints are hubs of i (forced open)
        hub_nonhub_mask:  exactly one endpoint is a hub (choice set)
        nonhub_nonhub_mask: neither endpoint is a hub (forbidden)

    Each edge falls into exactly one category.
    """
    hubs_set = set(hubs_i)
    a_is_hub = np.array([a in hubs_set for a in endpoints_a])
    b_is_hub = np.array([b in hubs_set for b in endpoints_b])
    hub_hub_mask = a_is_hub & b_is_hub
    hub_nonhub_mask = a_is_hub ^ b_is_hub
    nonhub_nonhub_mask = ~(a_is_hub | b_is_hub)
    return hub_hub_mask, hub_nonhub_mask, nonhub_nonhub_mask


def hub_of_spoke(edge_idx, hubs_i, endpoints_a, endpoints_b):
    """Given a hub-nonhub edge index, return which endpoint is the hub."""
    a, b = int(endpoints_a[edge_idx]), int(endpoints_b[edge_idx])
    return a if a in hubs_i else b


def build_hubs(N, C, populations, min_hubs, max_hubs, hub_pool_frac, rng,
               hub_pool_size=None):
    """Assign hub sets for N airlines from the largest cities.

    If hub_pool_size is given (e.g., 10), use that directly.
    Otherwise fall back to max(max_hubs, int(C * hub_pool_frac)).
    """
    if hub_pool_size is None:
        hub_pool_size = max(max_hubs, int(C * hub_pool_frac))
    hub_pool_size = min(hub_pool_size, C)
    hub_pool = np.argsort(populations)[-hub_pool_size:]
    hubs = []
    for _ in range(N):
        n_hubs = rng.integers(min_hubs, min(max_hubs, len(hub_pool)) + 1)
        hub_set = set(rng.choice(hub_pool, size=n_hubs, replace=False).tolist())
        hubs.append(hub_set)
    return hubs


def build_covariates(C, M, endpoints_a, endpoints_b, dists, populations, fe_mode):
    """Build shared item covariate matrix phi: (M, n_shared).

    Columns:
      0: gravity = pop_a * pop_b / dist_ab   (undirected)

    fe_mode="none" only supported (no origin FEs with undirected edges).
    """
    d = dists[endpoints_a, endpoints_b]
    d_safe = np.maximum(d, 1e-6)
    gravity = populations[endpoints_a] * populations[endpoints_b] / d_safe
    phi = gravity[:, None]  # (M, 1)

    if fe_mode != "none":
        raise ValueError(
            f"fe_mode={fe_mode} not supported with undirected edges")

    return phi


def n_shared_covariates(C, fe_mode):
    """Number of shared (non-firm-specific) modular covariates."""
    base = 1  # gravity only
    if fe_mode == "none":
        return base
    else:
        raise ValueError(f"Unknown fe_mode: {fe_mode}")


def n_params(C, N, fe_mode):
    """Total parameters: n_shared + N (firm FCs) + 1 (theta_gs)."""
    return n_shared_covariates(C, fe_mode) + N + 1


def compute_utility(bundle, phi, theta_rev, theta_fc_i, theta_gs, hubs_i,
                    endpoints_a, endpoints_b):
    """Utility V_i(b) for firm i with undirected edges.

    Congestion counts edges of the form {h, c} with c not in hubs_i,
    summed over hubs h, squared per hub, summed across hubs.
    """
    n_active = int(bundle.sum())
    modular = (phi[bundle] @ theta_rev).sum() - theta_fc_i * n_active
    congestion = 0.0
    if theta_gs > 0 and len(hubs_i) > 0:
        _, spoke_mask, _ = edge_categories(hubs_i, endpoints_a, endpoints_b)
        # For each hub, count spokes incident to it
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
    """Greedy with hub-clique seeded open and forbidden non-hub-non-hub edges.

    - Edges with both endpoints in hubs_i: forced open (contribute to
      revenue - FC + error but NOT congestion). Added to bundle upfront.
    - Edges with exactly one endpoint in hubs_i: greedy choice set.
    - Edges with neither endpoint in hubs_i: forbidden.
    """
    hh_mask, spoke_mask, _ = edge_categories(hubs_i, endpoints_a, endpoints_b)
    bundle = hh_mask.copy()  # seed with hub-hub clique

    base_marginals = (phi @ theta_rev).ravel() - theta_fc_i + errors_i

    # Per-hub count of CURRENTLY OPEN spokes incident to that hub
    hubs_set = set(hubs_i)
    hub_counts = {h: 0 for h in hubs_i}

    # Precompute which hub each spoke edge is incident to
    # (for every edge: if spoke, which endpoint is the hub)
    spoke_hub = -np.ones(M, dtype=int)
    for j in np.where(spoke_mask)[0]:
        a, b = int(endpoints_a[j]), int(endpoints_b[j])
        spoke_hub[j] = a if a in hubs_set else b

    while True:
        marginals = base_marginals.copy()
        # Forbid non-spoke items (already-open hub-hub, or forbidden nonhub-nonhub)
        marginals[~spoke_mask] = -np.inf
        marginals[bundle] = -np.inf  # already selected spokes

        # Apply congestion penalty per spoke based on which hub it touches
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

    return bundle


def check_healthy_dgp(bundles, M, N, C, fe_mode, endpoints_a, endpoints_b,
                      hubs, healthy_cfg):
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
    # Require every firm to have at least ~5 spokes beyond its clique for FC identification
    min_bundle = int(sizes.min())
    if min_bundle < 8:
        ok = False

    # Cross-airline heterogeneity — only meaningful for N > 1
    if N > 1 and std_size < 2.0:
        ok = False

    # Congestion variation (undirected: count hub-non-hub spokes per hub)
    hub_route_counts = []
    for i, b in enumerate(bundles):
        _, spoke_mask, _ = edge_categories(hubs[i], endpoints_a, endpoints_b)
        hubs_set = set(hubs[i])
        n_by_hub = {h: 0 for h in hubs[i]}
        for j in np.where(b & spoke_mask)[0]:
            a_, b_ = int(endpoints_a[j]), int(endpoints_b[j])
            h = a_ if a_ in hubs_set else b_
            n_by_hub[h] += 1
        hub_route_counts.append(sum(n ** 2 for n in n_by_hub.values()))
    hub_route_counts = np.array(hub_route_counts, dtype=float)
    frac_with_hub_routes = (hub_route_counts > 0).mean()
    cong_std = hub_route_counts.std()
    diag["frac_with_hub_routes"] = float(frac_with_hub_routes)
    diag["congestion_std"] = float(cong_std)
    if frac_with_hub_routes < min_frac_hub:
        ok = False
    # cong_std is only meaningful for N > 1
    if N > 1 and cong_std < min_cong_std:
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
    pop_dist = dgp_cfg.get('pop_dist', 'lognormal')
    pareto_alpha = dgp_cfg.get('pareto_alpha', 1.5)
    hub_pool_size = dgp_cfg.get('hub_pool_size', None)

    max_candidates = healthy_cfg['max_candidates']
    theta_rev_range = healthy_cfg['theta_rev_range']
    theta_fc_range = healthy_cfg['theta_fc_range']
    theta_gs_range = healthy_cfg['theta_gs_range']

    rng_dgp = np.random.default_rng(seeds['dgp'])
    rng_search = np.random.default_rng(seeds['search'])

    _, endpoints_a, endpoints_b, M = build_edges(C)
    locations, dists, populations = build_geography(
        C, pop_log_std, rng_dgp,
        pop_dist=pop_dist, pareto_alpha=pareto_alpha)
    phi = build_covariates(C, M, endpoints_a, endpoints_b, dists, populations,
                           fe_mode)
    hubs = build_hubs(N, C, populations, min_hubs, max_hubs, hub_pool_frac,
                      rng_dgp, hub_pool_size=hub_pool_size)

    n_shared = n_shared_covariates(C, fe_mode)

    rng_err = np.random.default_rng(seeds['dgp'] + 999)
    errors = rng_err.normal(0, sigma, (N, M))

    gravity_median = float(np.median(phi[:, 0]))

    best_theta, best_bundles, best_diag, best_score = None, None, None, -np.inf

    for trial in range(max_candidates):
        # --- Calibrate to a target "unit" revenue ---
        # rev_unit := theta_rev * median_gravity  (we want this ~ 1-3 times sigma)
        # theta_fc in units of rev_unit
        # theta_gs set so marginal cost at k_target spokes equals rev_unit
        rev_unit = rng_search.uniform(*theta_rev_range) * sigma
        theta_rev = np.zeros(n_shared)
        theta_rev[0] = rev_unit / gravity_median
        if fe_mode == "origin":
            theta_rev[1:] = rng_search.uniform(-1.0, 1.0, C)

        # FC: fraction of rev_unit. With fc_frac in [0.2, 1.2], roughly half
        # the spokes are net-positive before congestion.
        fc_frac = rng_search.uniform(*theta_fc_range)
        fc_mean = fc_frac * rev_unit
        fc_disp = healthy_cfg.get('theta_fc_dispersion', 0.3)
        theta_fc = np.abs(rng_search.normal(fc_mean, fc_disp * fc_mean, N))

        # Congestion: theta_gs such that at k_target spokes per hub, the
        # marginal congestion cost equals gs_frac * rev_unit.
        # theta_gs * (2*k_target - 1) = gs_frac * rev_unit
        gs_frac = rng_search.uniform(*theta_gs_range)
        k_target = 10
        theta_gs = gs_frac * rev_unit / (2 * k_target - 1)

        bundles = []
        for i in range(N):
            b = greedy_demand(phi, theta_rev, theta_fc[i], theta_gs,
                              hubs[i], endpoints_a, endpoints_b,
                              errors[i], M)
            bundles.append(b)

        ok, diag = check_healthy_dgp(bundles, M, N, C, fe_mode,
                                     endpoints_a, endpoints_b, hubs,
                                     healthy_cfg)

        score = 0.0
        mf = diag["mean_bundle_frac"]
        if 0.005 < mf < 0.2:
            score += 1.0
        else:
            score -= abs(mf - 0.05)
        if diag["std_bundle_frac"] * M >= 2.0:
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
            # theta_star = [theta_rev..., theta_fc_0, ..., theta_fc_{N-1}, theta_gs]
            theta_star = np.concatenate([theta_rev, theta_fc, [theta_gs]])
            if verbose:
                logger.info(f"  Healthy theta found at trial {trial}: "
                            f"theta_rev={theta_rev}, theta_fc={theta_fc}, "
                            f"theta_gs={theta_gs:.3f}, diag={diag}")
            dgp_data = {
                "locations": locations, "dists": dists, "populations": populations,
                "phi": phi, "hubs": hubs, "errors": errors,
                "endpoints_a": endpoints_a, "endpoints_b": endpoints_b,
                "C": C, "M": M, "N": N, "fe_mode": fe_mode,
            }
            return theta_star, bundles, dgp_data, diag

        if score > best_score:
            best_score = score
            best_theta = np.concatenate([theta_rev, theta_fc, [theta_gs]])
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
