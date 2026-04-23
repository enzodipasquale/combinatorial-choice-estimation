"""Single point estimation at N=1 with verbose row generation.

Uses the same theta* and first rep's geography as monte_carlo_N1.json.
Run:  python numerical_experiments/scenarios/airline/point_estimate_N1.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import yaml

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

from generate_data import (build_covariates, build_edges, build_geography,
                           build_hubs, greedy_demand, n_shared_covariates)
from monte_carlo import (AirlineGreedySolver, dgp_err_seed, dgp_geo_seed,
                         sim_err_seed)
from oracle import build_covariates_oracle

import combest as ce


def main(rep: int = 0):
    with open(BASE / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    dgp_cfg = dict(cfg["dgp"])
    dgp_cfg["N"] = 1
    est_cfg = cfg["estimation"]
    seeds = cfg["seeds"]

    # Reuse theta* from N=10 MC (fc = average of 10 firms)
    mc10 = json.load(open(BASE / "results" / "monte_carlo.json"))
    theta10 = np.array(mc10["theta_true"])
    theta_rev = float(theta10[0])
    theta_fc = float(theta10[1:11].mean())
    theta_gs = float(theta10[-1])
    theta_star = np.array([theta_rev, theta_fc, theta_gs])
    print(f"theta* (N=1): rev={theta_rev:.4f}, fc={theta_fc:.4f}, "
          f"gs={theta_gs:.4f}")

    C = dgp_cfg["C"]
    N = 1
    n_shared = n_shared_covariates(C, dgp_cfg["fe_mode"])
    n_p = n_shared + 1 + 1  # rev + 1 FC + gs

    # Fresh geography + hubs + errors (rep index passed in)
    rng_geo = np.random.default_rng(dgp_geo_seed(seeds["dgp"], rep))
    _, dists, populations = build_geography(
        C, dgp_cfg["pop_log_std"], rng_geo,
        pop_dist=dgp_cfg.get("pop_dist", "lognormal"),
        pareto_alpha=dgp_cfg.get("pareto_alpha", 1.5))
    _, endpoints_a, endpoints_b, M = build_edges(C)
    phi = build_covariates(C, M, endpoints_a, endpoints_b, dists, populations,
                           dgp_cfg["fe_mode"])
    hubs = build_hubs(N, C, populations,
                      dgp_cfg["min_hubs"], dgp_cfg["max_hubs"],
                      dgp_cfg["hub_pool_frac"], rng_geo,
                      hub_pool_size=dgp_cfg.get("hub_pool_size", None))

    rng_err = np.random.default_rng(dgp_err_seed(seeds["dgp"], rep))
    errors = rng_err.normal(0, dgp_cfg["sigma"], (N, M))

    obs = np.zeros((N, M), dtype=bool)
    obs[0] = greedy_demand(phi, np.array([theta_rev]), theta_fc, theta_gs,
                           hubs[0], endpoints_a, endpoints_b,
                           errors[0], M)
    print(f"Bundle size: {obs[0].sum()} / {M}  "
          f"(hubs: {sorted(hubs[0])})")

    # Estimation
    model = ce.Model()
    input_data = {
        "id_data": {"obs_bundles": obs,
                    "hubs": [list(h) for h in hubs]},
        "item_data": {"phi": phi,
                      "endpoints_a": endpoints_a,
                      "endpoints_b": endpoints_b,
                      "N_firms": N},
    }
    lbs = {0: 0, n_shared: 0, n_shared + 1: 0}
    model.load_config({
        "dimensions": {"n_obs": N, "n_items": M, "n_covariates": n_p,
                       "n_simulations": est_cfg["n_simulations"]},
        "subproblem": {"gurobi_params": {
            "TimeLimit": est_cfg["gurobi_timeout"], "OutputFlag": 0}},
        "row_generation": {"max_iters": est_cfg["max_iters"],
                           "tolerance": est_cfg["tolerance"],
                           "theta_bounds": {"lb": -20, "ub": 20, "lbs": lbs}},
    })
    model.data.load_and_distribute_input_data(input_data)
    ld = model.data.local_data
    ld.id_data["hubs"] = [set(h) for h in ld.id_data["hubs"]]
    ld.item_data["obs_ids"] = model.comm_manager.obs_ids
    model.features.build_local_modular_error_oracle(
        seed=sim_err_seed(seeds["error"], rep), sigma=dgp_cfg["sigma"])
    model.features.set_covariates_oracle(build_covariates_oracle(N))
    model.subproblems.load_solver(AirlineGreedySolver)
    model.subproblems.initialize_solver()

    result = model.point_estimation.n_slack.solve(
        initialize_solver=False, initialize_master=True, verbose=True)

    if result is None:
        print("No result.")
        return

    theta_hat = result.theta_hat
    err = theta_hat - theta_star
    err_pct = err / np.abs(theta_star) * 100
    print()
    print(f"{'Param':<14} {'True':>10} {'Hat':>10} {'Error':>10} {'Err%':>8}")
    print("-" * 54)
    for i, name in enumerate(["theta_rev", "theta_fc", "theta_cong"]):
        print(f"{name:<14} {theta_star[i]:>10.4f} {theta_hat[i]:>10.4f} "
              f"{err[i]:>+10.4f} {err_pct[i]:>+7.1f}%")
    print()
    print(f"converged={result.converged}, iters={result.num_iterations}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--rep", type=int, default=0,
                   help="Replication index (picks which geography/errors to use)")
    args = p.parse_args()
    main(args.rep)
