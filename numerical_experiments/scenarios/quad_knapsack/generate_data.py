"""DGP for quadratic knapsack / auction scenario.

Generates item locations, geographic adjacency Q, agent features,
BLP-style delta/instruments, and runs healthy-theta search.
"""

import numpy as np
from scipy import sparse
from combest.utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Geography and adjacency
# ---------------------------------------------------------------------------

def build_items(M, rng):
    """Place M items uniformly on [0,1]^2, compute pairwise distances."""
    locations = rng.uniform(0, 1, (M, 2))
    dists = np.sqrt(((locations[:, None] - locations[None, :]) ** 2).sum(-1))
    return locations, dists


def build_adjacency(M, dists, avg_degree, max_nnz_per_item):
    """Build sparse symmetric binary adjacency Q from geographic proximity.

    Radius chosen so pi*r^2*(M-1) ~ avg_degree.
    Aborts if nnz > max_nnz_per_item * M.

    Returns:
        Q: (M, M) sparse CSR matrix, symmetric, binary, zero diagonal.
        avg_deg: realized average degree.
        nnz: number of nonzero entries (counting both i,j and j,i).
    """
    r = np.sqrt(avg_degree / (np.pi * max(M - 1, 1)))
    np.fill_diagonal(dists, np.inf)
    Q = (dists <= r).astype(np.float64)
    np.fill_diagonal(Q, 0.0)
    # Ensure symmetric (distances are symmetric, but guard against float issues)
    Q = np.maximum(Q, Q.T)

    nnz = int(Q.sum())
    avg_deg = nnz / max(M, 1)
    if nnz > max_nnz_per_item * M:
        raise RuntimeError(
            f"Adjacency too dense: nnz={nnz} > {max_nnz_per_item}*{M}={max_nnz_per_item * M}. "
            f"avg_degree={avg_deg:.1f}, radius={r:.4f}")

    Q_sparse = sparse.csr_matrix(Q)
    return Q_sparse, Q, avg_deg, nnz


# ---------------------------------------------------------------------------
# Weights, capacities, features
# ---------------------------------------------------------------------------

def build_weights_capacities(M, N, dgp_cfg, rng):
    """Item weights ~ U(lo, hi), agent capacities ~ U(frac_lo*W, frac_hi*W)."""
    weights = rng.uniform(dgp_cfg['weight_lo'], dgp_cfg['weight_hi'], M)
    total_w = weights.sum()
    capacities = rng.uniform(
        dgp_cfg['capacity_frac_lo'] * total_w,
        dgp_cfg['capacity_frac_hi'] * total_w,
        N)
    return weights, capacities


def build_modular_features(N, M, rng):
    """Agent-item modular feature x_ij ~ N(0,1)."""
    return rng.normal(0, 1, (N, M))


# ---------------------------------------------------------------------------
# BLP data: delta, instruments, prices
# ---------------------------------------------------------------------------

def build_blp_data(M, dgp_cfg, rng):
    """Generate delta, instruments, and prices per SPEC.

    Returns dict with keys: delta, phi, z, xi, beta_star, prices,
    beta_raw, scale_factor.
    """
    K_phi = dgp_cfg['K_phi']
    rho = dgp_cfg['rho']
    delta_std = dgp_cfg['delta_std']

    # Draw instruments and structural error
    z = rng.normal(0, 1, (M, K_phi))
    xi = rng.normal(0, 1, M)

    # Item characteristics (endogenous)
    phi = z + rho * xi[:, None]

    # Raw beta and raw delta
    beta_raw = rng.normal(0, 1, K_phi)
    delta_raw = phi @ beta_raw + xi

    # Demean and rescale to target std
    delta_raw -= delta_raw.mean()
    raw_std = max(delta_raw.std(), 1e-8)
    scale_factor = delta_std / raw_std
    delta = delta_raw * scale_factor

    # Effective beta: delta = phi @ beta_star + xi_star
    # where beta_star = s * beta_raw, xi_star = s * (xi - mu_orig)
    beta_star = beta_raw * scale_factor
    mu_orig = (phi @ beta_raw + xi).mean()
    xi_star = scale_factor * (xi - mu_orig)

    # Prices: p_j = pi_0 + pi_z^T z_j + pi_xi * xi_j + u_j
    pi_z = rng.normal(0, dgp_cfg['pi_z_std'], K_phi)
    prices = (dgp_cfg['pi_0']
              + z @ pi_z
              + dgp_cfg['pi_xi'] * xi
              + rng.normal(0, dgp_cfg['price_noise_std'], M))

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


# ---------------------------------------------------------------------------
# Solve QKP for all agents
# ---------------------------------------------------------------------------

def solve_all_agents(N, M, x, delta, Q_dense, lambda_, alpha, errors,
                     weights, capacities, solver_cfg):
    """Solve each agent's QKP exactly via Gurobi. Returns bundles + diagnostics."""
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

        # Capacity constraint
        model.addConstr(weights @ b <= capacities[i])

        # Objective: alpha * x_i^T b - delta^T b
        #            + lambda * sum_{j<j'} Q_{jj'} b_j b_{j'} + errors_i^T b
        # Gurobi computes b^T Q b = 2 * sum_{j<j'}, so pass 0.5 * lambda * Q.
        linear = alpha * x[i] - delta + errors[i]
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

def check_healthy_dgp(bundles, N, M, Q_dense, x, delta, lambda_, alpha,
                      weights, capacities, errors, avg_deg, nnz,
                      solve_times, gaps, healthy_cfg, solver_timeout):
    """Run healthy-DGP checks from SPEC. Returns (ok, diagnostics).

    At tiny size (M <= 20), only checks 1-5 and 8 are enforced.
    Checks 6 (quad contribution), 7 (Jaccard vs lambda=0), and
    9 (capacity binding) are computed but not enforced, because with
    small M and N the quadratic term has minimal effect and the capacity
    constraint rarely binds tightly.
    """
    diag = {}
    ok = True
    is_tiny = M <= 20

    # 1. Optimization gap is zero
    max_gap = float(gaps.max())
    diag['gurobi_max_gap'] = max_gap
    if max_gap > 0:
        ok = False

    # 2. Per-agent wall time
    diag['gurobi_max_wall_time_s'] = float(solve_times.max())
    diag['gurobi_median_wall_time_s'] = float(np.median(solve_times))
    if solve_times.max() >= solver_timeout:
        ok = False

    # 3. Bundle sparsity
    sizes = bundles.sum(axis=1).astype(float)
    mean_size = sizes.mean()
    diag['mean_bundle_size'] = float(mean_size)
    min_frac = healthy_cfg['min_mean_bundle_frac']
    max_frac = healthy_cfg['max_mean_bundle_frac']
    if not (min_frac * M <= mean_size <= max_frac * M):
        ok = False

    # 4. Cross-agent heterogeneity
    std_size = sizes.std()
    diag['std_bundle_size'] = float(std_size)
    if is_tiny:
        min_std = max(1.0, 0.05 * M)
    else:
        min_std = max(healthy_cfg['min_std_bundle_size'], 0.03 * M)
    if std_size < min_std:
        ok = False

    # 5. Item-level identification
    item_present = bundles.any(axis=0)
    item_absent = (~bundles).any(axis=0)
    item_id_ok = bool(item_present.all() and item_absent.all())
    diag['item_identification_ok'] = item_id_ok
    if not item_id_ok:
        ok = False

    # 6. Quadratic term binds (computed always, enforced at pilot+ only)
    # Quadratic contribution: lambda * sum_{j<j'} Q b_j b_{j'} = 0.5 * lambda * b^T Q b
    quad_contribs = np.zeros(N)
    modular_contribs = np.zeros(N)
    for i in range(N):
        b = bundles[i].astype(float)
        quad_contribs[i] = 0.5 * lambda_ * b @ Q_dense @ b
        modular_contribs[i] = abs((alpha * x[i] - delta)[bundles[i]].sum())
    mean_quad = quad_contribs.mean()
    mean_modular = max(modular_contribs.mean(), 1e-8)
    quad_frac = mean_quad / mean_modular
    diag['quad_contribution_fraction'] = float(quad_frac)
    if not is_tiny and quad_frac < healthy_cfg['min_quad_contribution_frac']:
        ok = False

    # 7. Counterfactual at lambda=0 (computed always, enforced at pilot+ only)
    import gurobipy as gp
    jaccards = np.zeros(N)
    for i in range(N):
        model = gp.Model()
        model.setParam('OutputFlag', 0)
        model.setParam('MIPGap', 0)
        bvar = model.addMVar(M, vtype=gp.GRB.BINARY, name='b')
        model.addConstr(weights @ bvar <= capacities[i])
        linear_0 = alpha * x[i] - delta + errors[i]
        model.setObjective(linear_0 @ bvar, gp.GRB.MAXIMIZE)
        model.optimize()
        if model.SolCount > 0:
            b0 = np.array(model.x, dtype=bool)
            inter = (bundles[i] & b0).sum()
            union = (bundles[i] | b0).sum()
            jaccards[i] = inter / max(union, 1)
        else:
            jaccards[i] = 1.0
    mean_jaccard = float(jaccards.mean())
    diag['bundle_jaccard_vs_lambda0'] = mean_jaccard
    if not is_tiny and mean_jaccard > healthy_cfg['max_jaccard_vs_lambda0']:
        ok = False

    # 8. Sparsity hit
    diag['avg_degree'] = float(avg_deg)
    diag['nnz_Q'] = int(nnz)
    if not (healthy_cfg['min_avg_degree'] <= avg_deg <= healthy_cfg['max_avg_degree']):
        ok = False

    # 9. Capacity binds (computed always, enforced at pilot+ only)
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

def search_healthy_theta(size_cfg, dgp_cfg, healthy_cfg, solver_cfg, seeds,
                         verbose=True):
    """Random search over (lambda, delta_scale) to find healthy DGP.

    Returns (theta_star, bundles, dgp_data, diagnostics).
    """
    N = size_cfg['N']
    M = size_cfg['M']
    timeout = size_cfg['gurobi_timeout']

    rng_dgp = np.random.default_rng(seeds['dgp'])
    rng_search = np.random.default_rng(seeds['search'])

    # Build geography (fixed across search)
    locations, dists = build_items(M, rng_dgp)
    Q_sparse, Q_dense, avg_deg, nnz = build_adjacency(
        M, dists, dgp_cfg['avg_degree'], dgp_cfg['max_nnz_per_item'])

    # Weights and capacities (fixed)
    weights, capacities = build_weights_capacities(M, N, dgp_cfg, rng_dgp)

    # Modular features (fixed)
    x = build_modular_features(N, M, rng_dgp)

    # BLP data (fixed)
    blp = build_blp_data(M, dgp_cfg, rng_dgp)

    # Modular errors (fixed)
    rng_err = np.random.default_rng(seeds['dgp'] + 999)
    errors = rng_err.normal(0, dgp_cfg['sigma'], (N, M))

    max_candidates = healthy_cfg['max_candidates']
    lambda_range = healthy_cfg['lambda_range']
    alpha_range = healthy_cfg.get('alpha_range', [0.05, 0.5])
    delta_scale_range = healthy_cfg['delta_scale_range']

    solver_cfg_with_timeout = dict(solver_cfg)
    solver_cfg_with_timeout['TimeLimit'] = timeout

    best = {'score': -np.inf}

    for trial in range(max_candidates):
        lambda_ = rng_search.uniform(*lambda_range)
        alpha = rng_search.uniform(*alpha_range)
        delta_scale = rng_search.uniform(*delta_scale_range)
        delta_trial = blp['delta'] * (delta_scale / max(dgp_cfg['delta_std'], 1e-8))

        bundles, solve_times, gaps_arr = solve_all_agents(
            N, M, x, delta_trial, Q_dense, lambda_, alpha, errors,
            weights, capacities, solver_cfg_with_timeout)

        ok, diag = check_healthy_dgp(
            bundles, N, M, Q_dense, x, delta_trial, lambda_, alpha,
            weights, capacities, errors, avg_deg, nnz,
            solve_times, gaps_arr, healthy_cfg, timeout)

        if verbose:
            logger.info(f"  Trial {trial}: lambda={lambda_:.4f}, "
                        f"delta_scale={delta_scale:.3f}, "
                        f"ok={ok}, mean_size={diag['mean_bundle_size']:.1f}")

        if ok:
            # Build theta_star = [alpha, delta_1..delta_M, lambda]
            theta_star = np.concatenate([[alpha], delta_trial, [lambda_]])

            # Rescale BLP data consistently.
            # delta_trial = blp['delta'] * ratio where ratio = delta_scale / delta_std.
            # blp['delta'] = scale_factor * (phi@beta_raw + xi - mu_orig) = phi@beta_star + xi_star
            # So delta_trial = ratio * (phi@beta_star + xi_star)
            #                = phi @ (ratio*beta_star) + ratio*xi_star
            blp_final = dict(blp)
            blp_final['delta'] = delta_trial
            ratio = delta_scale / max(dgp_cfg['delta_std'], 1e-8)
            blp_final['beta_star'] = blp['beta_star'] * ratio
            blp_final['xi'] = blp['xi'] * ratio

            dgp_data = {
                'locations': locations, 'Q_sparse': Q_sparse, 'Q_dense': Q_dense,
                'weights': weights, 'capacities': capacities,
                'x': x, 'errors': errors, 'blp': blp_final,
                'avg_deg': avg_deg, 'nnz': nnz,
                'N': N, 'M': M,
            }
            return theta_star, bundles, dgp_data, diag

        # Track best partial candidate
        score = 0.0
        ms = diag['mean_bundle_size']
        if 0.15 * M <= ms <= 0.45 * M:
            score += 2.0
        if diag.get('item_identification_ok', False):
            score += 1.0
        if diag['gurobi_max_gap'] == 0:
            score += 1.0
        if score > best['score']:
            best = {'score': score, 'lambda': lambda_,
                    'delta_scale': delta_scale, 'diag': diag}

    raise RuntimeError(
        f"Healthy-DGP search failed after {max_candidates} candidates. "
        f"Best: {best}")


# ---------------------------------------------------------------------------
# Top-level entry
# ---------------------------------------------------------------------------

def generate_data(size_cfg, dgp_cfg, healthy_cfg, solver_cfg, seeds,
                  defaults=None, verbose=True):
    """Top-level DGP entry point.

    Returns: (theta_star, bundles, dgp_data, diagnostics)
    """
    return search_healthy_theta(
        size_cfg, dgp_cfg, healthy_cfg, solver_cfg, seeds, verbose=verbose)
