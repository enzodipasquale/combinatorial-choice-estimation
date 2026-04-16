"""BGK moment inequality estimator (Bontemps, Gualdani, Remmy 2025).

Implements the support-function approach from their Section 5.
The identified set is defined by averaging moment inequalities
across markets (routes), using instruments to select subsets.

Key equation (22)/(25): for each firm f and instrument r,
average the one-link deviation inequalities across markets,
so that E[eta]=0 makes them informative.
"""

import sys
from pathlib import Path
import numpy as np
import gurobipy as gp
from gurobipy import GRB

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from generate_data import (build_geography, build_edges, build_covariates,
                           build_hubs, greedy_demand)


def build_moment_inequalities(obs_bundles, phi, hubs, origin_of, N, M,
                              theta_rev_true, instruments=None):
    """Build averaged moment inequalities as in BGK eq. (22)/(25).

    For each firm f and instrument r, we average across routes:

    UPPER BOUND (from unserved routes, eq.11 averaged):
      (1/|unserved_r|) sum_{j unserved, Z=1} [revenue_j]
          <= gamma_1f + gamma_2 * (1/|unserved_r|) sum_{j unserved, Z=1} [DeltaQ_j]

    LOWER BOUND (from served routes, eq.12 averaged):
      (1/|served_r|) sum_{j served, Z=1} [revenue_j]
          >= gamma_1f + gamma_2 * (1/|served_r|) sum_{j served, Z=1} [DeltaQ_j]

    With E[eta]=0, the revenue averages converge to the true conditional
    expectation, making bounds informative.

    Args:
        instruments: list of functions Z_r(f, j, data) -> bool, or None
                     for default instruments (all routes, hub routes, non-hub)

    Returns:
        list of (b_vec, a_scalar): b @ gamma <= a
    """
    gravity = phi[:, 0]
    n_gamma = N + 1  # [theta_fc_0, ..., theta_fc_{N-1}, theta_gs]

    # Default instruments: partition routes by hub status
    if instruments is None:
        instruments = _default_instruments(hubs, origin_of, N, M)

    ineqs = []

    for f in range(N):
        bundle = obs_bundles[f]
        hubs_f = hubs[f]

        # Hub outbound counts in observed bundle
        hub_counts = {}
        for h in hubs_f:
            hub_counts[h] = int((bundle & (origin_of == h)).sum())

        for inst_name, inst_mask in instruments[f].items():
            # --- Unserved routes with Z=1: upper bound on gamma ---
            unserved_z = (~bundle) & inst_mask
            n_unserved = unserved_z.sum()

            if n_unserved > 0:
                # Average revenue of unserved routes selected by instrument
                avg_rev = theta_rev_true * gravity[unserved_z].mean()

                # Average DeltaQ for adding each unserved route
                delta_q_vals = np.zeros(M)
                for j in np.where(unserved_z)[0]:
                    o = origin_of[j]
                    if o in hub_counts:
                        delta_q_vals[j] = 2 * hub_counts[o] + 1
                avg_delta_q = delta_q_vals[unserved_z].mean()

                # Inequality: avg_rev <= gamma_1f + gamma_2 * avg_delta_q
                # => -gamma_1f - gamma_2 * avg_delta_q <= -avg_rev
                b = np.zeros(n_gamma)
                b[f] = -1.0
                b[-1] = -avg_delta_q
                ineqs.append((b, -avg_rev))

            # --- Served routes with Z=1: lower bound on gamma ---
            served_z = bundle & inst_mask
            n_served = served_z.sum()

            if n_served > 0:
                # Average revenue of served routes selected by instrument
                avg_rev = theta_rev_true * gravity[served_z].mean()

                # Average DeltaQ for removing each served route
                delta_q_vals = np.zeros(M)
                for j in np.where(served_z)[0]:
                    o = origin_of[j]
                    if o in hub_counts:
                        delta_q_vals[j] = 2 * hub_counts[o] - 1
                avg_delta_q = delta_q_vals[served_z].mean()

                # Inequality: avg_rev >= gamma_1f + gamma_2 * avg_delta_q
                # => gamma_1f + gamma_2 * avg_delta_q <= avg_rev
                b = np.zeros(n_gamma)
                b[f] = 1.0
                b[-1] = avg_delta_q
                ineqs.append((b, avg_rev))

    return ineqs


def _default_instruments(hubs, origin_of, N, M):
    """Default instruments: partition routes by characteristics.

    For each firm, creates instrument masks selecting subsets of routes:
    1. All routes
    2. Routes from hub origins
    3. Routes from non-hub origins
    4. Short-distance routes (below median distance)
    5. Long-distance routes (above median distance)
    """
    instruments = []
    for f in range(N):
        hubs_f = hubs[f]
        hub_origin = np.array([origin_of[j] in hubs_f for j in range(M)])
        non_hub_origin = ~hub_origin

        masks = {
            'all': np.ones(M, dtype=bool),
            'hub_origin': hub_origin,
            'non_hub_origin': non_hub_origin,
        }
        instruments.append(masks)
    return instruments


def solve_support_function(ineqs, n_gamma, direction, gamma_bounds=None):
    """Solve support function LP: max/min q^T gamma s.t. moment ineqs."""
    results = {}

    for sense_name, sense in [('upper', GRB.MAXIMIZE), ('lower', GRB.MINIMIZE)]:
        m = gp.Model()
        m.Params.OutputFlag = 0

        lb = [-100.0] * n_gamma
        ub = [100.0] * n_gamma
        if gamma_bounds:
            for i in range(n_gamma):
                if 'lbs' in gamma_bounds and i in gamma_bounds['lbs']:
                    lb[i] = max(lb[i], gamma_bounds['lbs'][i])
                if 'ubs' in gamma_bounds and i in gamma_bounds['ubs']:
                    ub[i] = min(ub[i], gamma_bounds['ubs'][i])

        gamma = m.addMVar(n_gamma, lb=lb, ub=ub, name='gamma')
        m.setObjective(direction @ gamma, sense)

        if len(ineqs) > 0:
            B = np.array([iq[0] for iq in ineqs])
            a = np.array([iq[1] for iq in ineqs])
            m.addConstr(B @ gamma <= a)

        m.optimize()

        if m.Status == GRB.OPTIMAL:
            results[sense_name] = m.ObjVal
        else:
            results[sense_name] = None

    return results.get('lower'), results.get('upper')


def estimate_identified_set(obs_bundles, phi, hubs, origin_of, N, M,
                            theta_rev_true, gamma_bounds=None):
    """Compute identified set projections for each component of gamma."""
    ineqs = build_moment_inequalities(
        obs_bundles, phi, hubs, origin_of, N, M, theta_rev_true)

    n_gamma = N + 1
    param_names = [f"theta_fc_{i}" for i in range(N)] + ["theta_gs"]

    if gamma_bounds is None:
        lbs = {i: 0 for i in range(n_gamma)}
        gamma_bounds = {'lbs': lbs}

    results = {}
    for p, name in enumerate(param_names):
        q = np.zeros(n_gamma)
        q[p] = 1.0
        lb, ub = solve_support_function(ineqs, n_gamma, q, gamma_bounds)
        results[name] = {'lb': lb, 'ub': ub}

    return results, len(ineqs)
