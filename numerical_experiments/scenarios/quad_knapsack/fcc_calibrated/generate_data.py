"""DGP for FCC-calibrated quadratic knapsack scenario.

Synthetic data matching the economic structure of the FCC C-block
spectrum auction: lognormal populations/capacities, geographic
adjacency with row-normalization, 6 modular regressors built from
eligibility, assets, and distance to headquarters.
"""

import numpy as np
from scipy import sparse
from combest.utils import get_logger

logger = get_logger(__name__)

def _demean(x):
    """Demean along first axis."""
    return x - x.mean(axis=0)


def build_blp_data(M, blp_cfg, rng):
    """Generate BLP data with demeaned-lognormal instruments and errors.

    Draws z_j and xi_j from demeaned lognormals (skewed, mean-zero)
    rather than Gaussians, matching the heavy-tailed structure of
    real auction unobservables.

    The exclusion restriction E[z_j xi_j] = 0 holds by construction
    since z and xi are drawn independently and both demeaned.
    """
    K_phi = blp_cfg['K_phi']
    rho = blp_cfg['rho']
    delta_std = blp_cfg['delta_std']

    # Instruments and structural error: demeaned lognormals
    z_raw = rng.lognormal(0, 1, (M, K_phi))
    z = _demean(z_raw)

    xi_raw = rng.lognormal(0, 1, M)
    xi = xi_raw - xi_raw.mean()

    # Item characteristics (endogenous): phi correlated with xi via rho
    phi = z + rho * xi[:, None]

    # Raw beta and raw delta
    beta_raw = rng.normal(0, 1, K_phi)
    delta_raw = phi @ beta_raw + xi

    # Demean and rescale to target std
    delta_raw -= delta_raw.mean()
    raw_std = max(delta_raw.std(), 1e-8)
    scale_factor = delta_std / raw_std
    delta = delta_raw * scale_factor

    # Effective beta and structural error after rescaling
    beta_star = beta_raw * scale_factor
    mu_orig = (phi @ beta_raw + xi).mean()
    xi_star = scale_factor * (xi - mu_orig)

    # Prices
    pi_z = rng.normal(0, blp_cfg['pi_z_std'], K_phi)
    prices = (blp_cfg['pi_0']
              + z @ pi_z
              + blp_cfg['pi_xi'] * xi
              + rng.normal(0, blp_cfg['price_noise_std'], M))

    return {
        'delta': delta,
        'phi': phi,
        'z': z,
        'xi': xi_star,
        'beta_star': beta_star,
        'beta_raw': beta_raw,
        'scale_factor': scale_factor,
        'mu_orig': mu_orig,
        'prices': prices,
        'pi_z': pi_z,
    }

N_MODULAR = 6
REGRESSOR_NAMES = [
    'elig_pop', 'assets_pop', 'log_dist_hq',
    'elig_log_dist_hq', 'log_dist_hq_pop', 'elig_log_dist_hq_pop',
]


def n_params(M):
    """Total parameters: 6 modular + M deltas + 1 lambda."""
    return N_MODULAR + M + 1


# ---------------------------------------------------------------------------
# Items (BTA-like)
# ---------------------------------------------------------------------------

def build_items(M, rng, item_cfg):
    """Place M items on [0,1]^2 with lognormal populations.

    Returns: locations (M,2), pop (M,), dists (M,M)
    """
    locations = rng.uniform(0, 1, (M, 2))
    dists = np.sqrt(((locations[:, None] - locations[None, :]) ** 2).sum(-1))

    raw = rng.lognormal(item_cfg['pop_log_mu'], item_cfg['pop_log_sigma'], M)
    pop = np.clip(raw, item_cfg['pop_min'], item_cfg['pop_max']).astype(int)
    pop = np.maximum(pop, item_cfg['pop_min'])  # ensure floor

    return locations, pop, dists


# ---------------------------------------------------------------------------
# Adjacency — matching registries.normalize_interaction_matrix
# ---------------------------------------------------------------------------

def normalize_interaction_matrix(matrix, pop_norm):
    """Row-normalize and pop-scale, matching the FCC application exactly.

    Args:
        matrix: (M, M) raw interaction matrix (e.g. binary adjacency).
        pop_norm: (M,) normalized population (sums to ~1).

    Returns: (M, M) normalized matrix.
    """
    matrix = matrix.copy().astype(float)
    np.fill_diagonal(matrix, 0)
    outflow = matrix.sum(axis=1)
    mask = outflow > 0
    matrix[mask] /= outflow[mask][:, None]
    matrix *= pop_norm[:, None]
    np.fill_diagonal(matrix, 0)
    return matrix


def build_adjacency(M, dists, pop_norm, geo_cfg):
    """Build geographic adjacency, then normalize like the FCC application.

    Returns:
        Q_sparse: CSR sparse of the normalized matrix.
        Q_dense:  dense (M, M) normalized matrix.
        Q_binary: dense (M, M) raw binary adjacency.
        avg_deg:  realized average degree of binary adjacency.
        nnz:      nonzero count in binary adjacency.
    """
    avg_degree = geo_cfg['avg_degree']
    max_nnz = geo_cfg['max_nnz_per_item']

    r = np.sqrt(avg_degree / (np.pi * max(M - 1, 1)))
    d = dists.copy()
    np.fill_diagonal(d, np.inf)
    Q_binary = (d <= r).astype(float)
    np.fill_diagonal(Q_binary, 0.0)
    Q_binary = np.maximum(Q_binary, Q_binary.T)

    nnz = int(Q_binary.sum())
    avg_deg = nnz / max(M, 1)
    if nnz > max_nnz * M:
        raise RuntimeError(
            f"Adjacency too dense: nnz={nnz} > {max_nnz}*M={max_nnz * M}")

    Q_dense = normalize_interaction_matrix(Q_binary, pop_norm)
    Q_sparse = sparse.csr_matrix(Q_dense)

    return Q_sparse, Q_dense, Q_binary, avg_deg, nnz


# ---------------------------------------------------------------------------
# Agents (bidder-like)
# ---------------------------------------------------------------------------

def build_agents(N, M, pop, rng, agent_cfg):
    """Draw agent attributes from FCC-calibrated distributions.

    Returns dict with: capacity, elig, assets, hq_idx.
    """
    total_w = int(pop.sum())

    # Capacity: lognormal, clipped
    cap_raw = rng.lognormal(agent_cfg['cap_log_mu'],
                            agent_cfg['cap_log_sigma'], N)
    capacity = np.clip(cap_raw, agent_cfg['cap_min'], total_w).astype(int)
    capacity = np.maximum(capacity, agent_cfg['cap_min'])

    # Eligibility: normalized capacity
    elig = capacity.astype(float) / total_w

    # Assets: Bernoulli-lognormal mixture, normalized to [0,1]
    is_zero = rng.random(N) < agent_cfg['assets_zero_prob']
    assets_raw = rng.lognormal(agent_cfg['assets_log_mu'],
                               agent_cfg['assets_log_sigma'], N)
    assets_raw[is_zero] = 0.0
    amax = max(assets_raw.max(), 1e-8)
    assets = assets_raw / amax

    # HQ: random item
    hq_idx = rng.integers(0, M, N)

    return {
        'capacity': capacity,
        'elig': elig,
        'assets': assets,
        'hq_idx': hq_idx,
    }


# ---------------------------------------------------------------------------
# Modular features (6 regressors matching FCC application)
# ---------------------------------------------------------------------------

def build_modular_features(N, M, elig, assets, pop_norm, dists, hq_idx):
    """Build 6 modular regressors matching registries.py exactly.

    Args:
        pop_norm: (M,) population normalized to sum to 1.
        dists:    (M, M) pairwise distances.
        hq_idx:   (N,) headquarters item index per agent.

    Returns: (N, M, 6) array.
    """
    # log_dist_hq[i, j] = log(1 + dists[hq[i], j])
    log_dist_hq = np.log1p(dists[hq_idx])  # (N, M)

    x = np.zeros((N, M, N_MODULAR))
    x[:, :, 0] = elig[:, None] * pop_norm[None, :]          # elig_pop
    x[:, :, 1] = assets[:, None] * pop_norm[None, :]        # assets_pop
    x[:, :, 2] = log_dist_hq                                # log_dist_hq
    x[:, :, 3] = elig[:, None] * log_dist_hq                # elig_log_dist_hq
    x[:, :, 4] = log_dist_hq * pop_norm[None, :]            # log_dist_hq_pop
    x[:, :, 5] = elig[:, None] * log_dist_hq * pop_norm[None, :]  # elig_log_dist_hq_pop

    return x


# ---------------------------------------------------------------------------
# Solve QKP for all agents
# ---------------------------------------------------------------------------

def solve_all_agents(N, M, x_modular, delta, Q_dense, theta_mod, lambda_,
                     errors, weights, capacities, solver_cfg):
    """Solve each agent's QKP exactly via Gurobi.

    Utility: x_modular[i] @ theta_mod - delta + errors[i]  (linear)
             + 0.5 * lambda * b^T Q b                       (quadratic)
    s.t.     weights^T b <= capacity[i]
    """
    import gurobipy as gp

    bundles = np.zeros((N, M), dtype=bool)
    solve_times = np.zeros(N)
    gaps = np.zeros(N)

    for i in range(N):
        model = gp.Model()
        model.setParam('OutputFlag', solver_cfg.get('OutputFlag', 0))
        model.setParam('MIPGap', solver_cfg.get('MIPGap', 0))
        if 'TimeLimit' in solver_cfg:
            model.setParam('TimeLimit', solver_cfg['TimeLimit'])

        b = model.addMVar(M, vtype=gp.GRB.BINARY, name='b')
        model.addConstr(weights @ b <= capacities[i])

        linear = x_modular[i] @ theta_mod - delta + errors[i]
        model.setMObjective(
            Q=0.5 * lambda_ * Q_dense,
            c=linear,
            constant=0.0,
            sense=gp.GRB.MAXIMIZE
        )

        model.optimize()
        solve_times[i] = model.Runtime
        gaps[i] = model.MIPGap if model.SolCount > 0 else float('inf')

        if model.SolCount > 0:
            bundles[i] = np.array(model.x, dtype=bool)

    return bundles, solve_times, gaps


# ---------------------------------------------------------------------------
# Healthy-DGP checks
# ---------------------------------------------------------------------------

def check_healthy_dgp(bundles, N, M, Q_dense, x_modular, delta, theta_mod,
                      lambda_, weights, capacities, errors, avg_deg, nnz,
                      solve_times, gaps, healthy_cfg, solver_timeout):
    """Healthy-DGP checks adapted for sparse FCC-like bundles."""
    diag = {}
    ok = True
    is_tiny = M <= 20

    # 1. Optimization gap
    max_gap = float(gaps.max())
    diag['gurobi_max_gap'] = max_gap
    if max_gap > 0:
        ok = False

    # 2. Wall time
    diag['gurobi_max_wall_time_s'] = float(solve_times.max())
    diag['gurobi_median_wall_time_s'] = float(np.median(solve_times))
    if solve_times.max() >= solver_timeout:
        ok = False

    # 3. Bundle sparsity
    sizes = bundles.sum(axis=1).astype(float)
    mean_size = sizes.mean()
    diag['mean_bundle_size'] = float(mean_size)
    diag['median_bundle_size'] = float(np.median(sizes))
    diag['max_bundle_size'] = int(sizes.max())
    diag['min_bundle_size'] = int(sizes.min())
    diag['n_empty_bundles'] = int((sizes == 0).sum())
    min_frac = healthy_cfg['min_mean_bundle_frac']
    max_frac = healthy_cfg['max_mean_bundle_frac']
    if not (min_frac * M <= mean_size <= max_frac * M):
        ok = False

    # 4. Heterogeneity
    std_size = sizes.std()
    diag['std_bundle_size'] = float(std_size)
    if std_size < healthy_cfg['min_std_bundle_size']:
        ok = False

    # 5. Item identification (soft threshold for realistic sparse bundles)
    item_present = bundles.any(axis=0)
    item_absent = (~bundles).any(axis=0)
    n_never_chosen = int((~item_present).sum())
    n_always_chosen = int((~item_absent).sum())
    frac_unidentified = n_never_chosen / max(M, 1)
    diag['item_identification_ok'] = bool(n_never_chosen == 0 and n_always_chosen == 0)
    diag['n_items_never_chosen'] = n_never_chosen
    diag['n_items_always_chosen'] = n_always_chosen
    diag['frac_items_identified'] = float(1 - frac_unidentified)
    max_unid = healthy_cfg.get('max_unidentified_frac', 0.50)
    if not is_tiny and frac_unidentified > max_unid:
        ok = False

    # 6. Quadratic contribution
    quad_contribs = np.zeros(N)
    modular_contribs = np.zeros(N)
    for i in range(N):
        b = bundles[i].astype(float)
        quad_contribs[i] = 0.5 * lambda_ * b @ Q_dense @ b
        lin_val = (x_modular[i] @ theta_mod - delta)[bundles[i]]
        modular_contribs[i] = abs(lin_val.sum()) if bundles[i].any() else 0
    mean_quad = quad_contribs.mean()
    mean_modular = max(modular_contribs.mean(), 1e-8)
    quad_frac = mean_quad / mean_modular
    diag['quad_contribution_fraction'] = float(quad_frac)
    if not is_tiny and quad_frac < healthy_cfg['min_quad_contribution_frac']:
        ok = False

    # 7. Counterfactual at lambda=0
    import gurobipy as gp
    jaccards = []
    for i in range(N):
        if not bundles[i].any():
            continue
        model = gp.Model()
        model.setParam('OutputFlag', 0)
        model.setParam('MIPGap', 0)
        if solver_timeout:
            model.setParam('TimeLimit', solver_timeout)
        bvar = model.addMVar(M, vtype=gp.GRB.BINARY, name='b')
        model.addConstr(weights @ bvar <= capacities[i])
        lin_0 = x_modular[i] @ theta_mod - delta + errors[i]
        model.setObjective(lin_0 @ bvar, gp.GRB.MAXIMIZE)
        model.optimize()
        if model.SolCount > 0:
            b0 = np.array(model.x, dtype=bool)
            inter = (bundles[i] & b0).sum()
            union = (bundles[i] | b0).sum()
            jaccards.append(inter / max(union, 1))
    mean_jaccard = float(np.mean(jaccards)) if jaccards else 1.0
    diag['bundle_jaccard_vs_lambda0'] = mean_jaccard
    diag['n_agents_for_jaccard'] = len(jaccards)
    if not is_tiny and mean_jaccard > healthy_cfg['max_jaccard_vs_lambda0']:
        ok = False

    # 8. Sparsity
    diag['avg_degree'] = float(avg_deg)
    diag['nnz_Q'] = int(nnz)
    if not (healthy_cfg['min_avg_degree'] <= avg_deg <= healthy_cfg['max_avg_degree']):
        ok = False

    # 9. Capacity binding
    load_fracs = np.zeros(N)
    for i in range(N):
        load_fracs[i] = (weights * bundles[i]).sum() / max(capacities[i], 1e-8)
    threshold = healthy_cfg['capacity_binding_threshold']
    binding_frac = (load_fracs >= threshold).mean()
    diag['capacity_binding_fraction'] = float(binding_frac)
    if not is_tiny and binding_frac < healthy_cfg['min_capacity_binding_frac']:
        ok = False

    diag['all_checks_passed'] = ok
    return ok, diag


# ---------------------------------------------------------------------------
# Healthy-theta search
# ---------------------------------------------------------------------------

def search_healthy_theta(size_cfg, cfg, seeds, verbose=True):
    """8D random search over (theta_mod_1..6, lambda, delta_scale).

    Returns (theta_star, bundles, dgp_data, diagnostics).
    """
    N = size_cfg['N']
    M = size_cfg['M']
    timeout = size_cfg['gurobi_timeout']
    item_cfg = cfg['items']
    agent_cfg = cfg['agents']
    geo_cfg = cfg['geo']
    blp_cfg = cfg['blp']
    healthy_cfg = cfg['healthy_dgp']
    solver_cfg = cfg['solver']
    sigma = cfg['sigma']

    rng_dgp = np.random.default_rng(seeds['dgp'])
    rng_search = np.random.default_rng(seeds['search'])

    # Fixed geography
    locations, pop, dists = build_items(M, rng_dgp, item_cfg)
    weights = pop.copy()
    pop_norm = pop.astype(float) / pop.sum()

    Q_sparse, Q_dense, Q_binary, avg_deg, nnz = build_adjacency(
        M, dists, pop_norm, geo_cfg)

    # Fixed agents
    agents = build_agents(N, M, pop, rng_dgp, agent_cfg)
    capacities = agents['capacity']

    # Fixed features
    x_modular = build_modular_features(
        N, M, agents['elig'], agents['assets'], pop_norm,
        dists, agents['hq_idx'])

    # Fixed BLP data
    blp = build_blp_data(M, blp_cfg, rng_dgp)

    # Fixed errors
    rng_err = np.random.default_rng(seeds['dgp'] + 999)
    errors = rng_err.normal(0, sigma, (N, M))

    max_candidates = healthy_cfg['max_candidates']
    mod_ranges = healthy_cfg['modular_ranges']
    lambda_range = healthy_cfg['lambda_range']
    delta_scale_range = healthy_cfg['delta_scale_range']

    solver_cfg_run = dict(solver_cfg)
    solver_cfg_run['TimeLimit'] = timeout

    best = {'score': -np.inf}

    for trial in range(max_candidates):
        # Draw 6 modular coefficients
        theta_mod = np.array([
            rng_search.uniform(*mod_ranges['elig_pop']),
            rng_search.uniform(*mod_ranges['assets_pop']),
            rng_search.uniform(*mod_ranges['log_dist_hq']),
            rng_search.uniform(*mod_ranges['elig_log_dist_hq']),
            rng_search.uniform(*mod_ranges['log_dist_hq_pop']),
            rng_search.uniform(*mod_ranges['elig_log_dist_hq_pop']),
        ])
        lambda_ = rng_search.uniform(*lambda_range)
        delta_scale = rng_search.uniform(*delta_scale_range)
        delta_trial = blp['delta'] * (delta_scale / max(blp_cfg['delta_std'], 1e-8))

        bundles, solve_times, gaps_arr = solve_all_agents(
            N, M, x_modular, delta_trial, Q_dense, theta_mod, lambda_,
            errors, weights, capacities, solver_cfg_run)

        ok, diag = check_healthy_dgp(
            bundles, N, M, Q_dense, x_modular, delta_trial, theta_mod,
            lambda_, weights, capacities, errors, avg_deg, nnz,
            solve_times, gaps_arr, healthy_cfg, timeout)

        if verbose:
            logger.info(
                f"  Trial {trial}: mean_size={diag['mean_bundle_size']:.1f}, "
                f"empty={diag['n_empty_bundles']}, "
                f"never_chosen={diag.get('n_items_never_chosen', '?')}, "
                f"ok={ok}")

        if ok:
            theta_star = np.concatenate([theta_mod, delta_trial, [lambda_]])

            # Rescale BLP consistently
            ratio = delta_scale / max(blp_cfg['delta_std'], 1e-8)
            blp_final = dict(blp)
            blp_final['delta'] = delta_trial
            blp_final['beta_star'] = blp['beta_star'] * ratio
            blp_final['xi'] = blp['xi'] * ratio

            dgp_data = {
                'locations': locations, 'pop': pop, 'pop_norm': pop_norm,
                'dists': dists, 'weights': weights,
                'Q_sparse': Q_sparse, 'Q_dense': Q_dense,
                'avg_deg': avg_deg, 'nnz': nnz,
                'agents': agents, 'x_modular': x_modular,
                'errors': errors, 'blp': blp_final,
                'N': N, 'M': M,
            }
            return theta_star, bundles, dgp_data, diag

        # Score partial candidates
        score = 0.0
        ms = diag['mean_bundle_size']
        if healthy_cfg['min_mean_bundle_frac'] * M <= ms <= healthy_cfg['max_mean_bundle_frac'] * M:
            score += 2.0
        if diag.get('item_identification_ok', False):
            score += 2.0
        if diag['gurobi_max_gap'] == 0:
            score += 1.0
        if diag['n_empty_bundles'] < N * 0.5:
            score += 1.0
        if score > best['score']:
            best = {'score': score, 'trial': trial, 'diag': diag}

    raise RuntimeError(
        f"Healthy-DGP search failed after {max_candidates} candidates. "
        f"Best: {best}")


# ---------------------------------------------------------------------------
# Top-level entry
# ---------------------------------------------------------------------------

def generate_data(size_cfg, cfg, seeds, verbose=True):
    """Top-level entry. Returns (theta_star, bundles, dgp_data, diagnostics)."""
    return search_healthy_theta(size_cfg, cfg, seeds, verbose=verbose)
