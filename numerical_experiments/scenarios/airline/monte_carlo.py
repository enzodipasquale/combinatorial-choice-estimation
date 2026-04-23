"""Monte Carlo for the airline / gross-substitutes scenario.

Research-grade specification:
- theta* is fixed across replications (drawn once from a healthy-DGP search
  on a dedicated calibration seed, INDEPENDENT of rep seeds)
- Each replication draws fresh geography, populations, hubs, AND modular
  errors, using disjoint random streams
- Simulation errors for the estimator come from a third, disjoint stream

Seed structure (all np.random.default_rng with tuple seeds):
  Calibration (theta* search): (seeds['dgp'], 'calibration', 0, ...)
  Rep r DGP geography/hubs:    (seeds['dgp'], 99999, r)
  Rep r DGP errors:            (seeds['dgp'], 88888, r)
  Rep r simulation errors:     (seeds['error'], 77777, r)

No seed overlap, so DGP errors and simulation errors are independent draws.

Output:
- Reports bias, Std, RMSE, MAE%, MdAE% per parameter
- Filters to converged replications (non-converged are reported separately)
- Aborts on degenerate bundles (no silent skipping)
- Saves full theta_hat trajectory for downstream analysis
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_data import (build_covariates, build_edges, build_geography,
                           build_hubs, generate_data, greedy_demand,
                           n_params, n_shared_covariates)
from oracle import build_covariates_oracle, make_find_best_item

import combest as ce
from combest.subproblems.registry.greedy import GreedySolver
from combest.utils import get_logger

logger = get_logger(__name__)

BASE = Path(__file__).resolve().parent
with open(BASE / "config.yaml") as f:
    CFG = yaml.safe_load(f)


class AirlineGreedySolver(GreedySolver):
    find_best_item = staticmethod(make_find_best_item())


# ---------------------------------------------------------------------------
# Seed construction (disjoint streams for each component)
# ---------------------------------------------------------------------------

SEED_TAG_DGP_GEO = 99999       # geography, populations, hubs
SEED_TAG_DGP_ERR = 88888       # modular errors in the DGP
SEED_TAG_SIM_ERR = 77777       # simulation errors in the estimator


def dgp_geo_seed(dgp_seed: int, rep: int):
    return (dgp_seed, SEED_TAG_DGP_GEO, rep)


def dgp_err_seed(dgp_seed: int, rep: int):
    return (dgp_seed, SEED_TAG_DGP_ERR, rep)


def sim_err_seed(error_seed: int, rep: int):
    return (error_seed, SEED_TAG_SIM_ERR, rep)


def audit_seeds(base_cfg, n_reps: int):
    """Verify no seed collision across rep streams."""
    dgp = base_cfg["seeds"]["dgp"]
    err = base_cfg["seeds"]["error"]
    all_seeds = []
    for r in range(n_reps):
        all_seeds.append(("dgp_geo", dgp_geo_seed(dgp, r)))
        all_seeds.append(("dgp_err", dgp_err_seed(dgp, r)))
        all_seeds.append(("sim_err", sim_err_seed(err, r)))
    # Tuples are unique by construction (distinct middle tags).
    # Verify by hashing.
    hashes = {s for _, s in all_seeds}
    assert len(hashes) == 3 * n_reps, "Seed collision detected!"
    return all_seeds


# ---------------------------------------------------------------------------
# Single replication
# ---------------------------------------------------------------------------

def run_one_rep(rep, theta_star, base_cfg):
    """One MC replication. Returns dict with diagnostics and theta_hat."""
    dgp_cfg = base_cfg["dgp"]
    est_cfg = base_cfg["estimation"]
    seeds = base_cfg["seeds"]

    C = dgp_cfg["C"]
    N = dgp_cfg["N"]
    fe_mode = dgp_cfg["fe_mode"]
    sigma = dgp_cfg["sigma"]
    M = C * (C - 1) // 2  # undirected
    n_shared = n_shared_covariates(C, fe_mode)

    # --- Fresh geography + hubs ---
    rng_geo = np.random.default_rng(dgp_geo_seed(seeds["dgp"], rep))
    locations, dists, populations = build_geography(
        C, dgp_cfg["pop_log_std"], rng_geo,
        pop_dist=dgp_cfg.get("pop_dist", "lognormal"),
        pareto_alpha=dgp_cfg.get("pareto_alpha", 1.5))
    _, endpoints_a, endpoints_b, _ = build_edges(C)
    phi = build_covariates(C, M, endpoints_a, endpoints_b, dists, populations,
                           fe_mode)
    hubs = build_hubs(N, C, populations,
                     dgp_cfg["min_hubs"], dgp_cfg["max_hubs"],
                     dgp_cfg["hub_pool_frac"], rng_geo,
                     hub_pool_size=dgp_cfg.get("hub_pool_size", None))

    # --- Fresh DGP errors (independent stream) ---
    rng_err = np.random.default_rng(dgp_err_seed(seeds["dgp"], rep))
    errors = rng_err.normal(0, sigma, (N, M))

    # --- Observed bundles at theta* ---
    theta_rev = theta_star[:n_shared]
    theta_fc = theta_star[n_shared:n_shared + N]
    theta_gs = theta_star[-1]

    obs_bundles = np.zeros((N, M), dtype=bool)
    for i in range(N):
        obs_bundles[i] = greedy_demand(
            phi, theta_rev, theta_fc[i], theta_gs,
            hubs[i], endpoints_a, endpoints_b, errors[i], M)

    sizes = obs_bundles.sum(axis=1)

    # --- Health check ---
    healthy = (sizes.min() > 0) and (sizes.max() < M)
    if not healthy:
        return {
            "rep": rep,
            "healthy": False,
            "bundle_sizes": sizes.tolist(),
            "theta_hat": None,
            "converged": False,
            "num_iterations": 0,
        }

    # --- Estimation ---
    model = ce.Model()
    input_data = {
        "id_data": {
            "obs_bundles": obs_bundles,
            "hubs": [list(h) for h in hubs],
        },
        "item_data": {
            "phi": phi,
            "endpoints_a": endpoints_a,
            "endpoints_b": endpoints_b,
            "N_firms": N,
        },
    }

    n_p = n_params(C, N, fe_mode)
    lbs = {0: 0}
    for i in range(N):
        lbs[n_shared + i] = 0
    lbs[n_shared + N] = 0

    model_cfg = {
        "dimensions": {
            "n_obs": N, "n_items": M, "n_covariates": n_p,
            "n_simulations": est_cfg["n_simulations"],
        },
        "subproblem": {
            "gurobi_params": {
                "TimeLimit": est_cfg["gurobi_timeout"],
                "OutputFlag": 0,
            },
        },
        "row_generation": {
            "max_iters": est_cfg["max_iters"],
            "tolerance": est_cfg["tolerance"],
            "theta_bounds": {"lb": -20, "ub": 20, "lbs": lbs},
        },
    }
    model.load_config(model_cfg)
    model.data.load_and_distribute_input_data(input_data)

    ld = model.data.local_data
    ld.id_data["hubs"] = [set(h) for h in ld.id_data["hubs"]]
    ld.item_data["obs_ids"] = model.comm_manager.obs_ids

    # Simulation errors: third disjoint stream
    model.features.build_local_modular_error_oracle(
        seed=sim_err_seed(seeds["error"], rep), sigma=sigma)
    model.features.set_covariates_oracle(build_covariates_oracle(N))

    model.subproblems.load_solver(AirlineGreedySolver)
    model.subproblems.initialize_solver()

    row_gen = model.point_estimation.n_slack
    result = row_gen.solve(initialize_solver=False, initialize_master=True,
                           verbose=False)

    if result is None:
        return {
            "rep": rep,
            "healthy": True,
            "bundle_sizes": sizes.tolist(),
            "theta_hat": None,
            "converged": False,
            "num_iterations": 0,
        }

    return {
        "rep": rep,
        "healthy": True,
        "bundle_sizes": sizes.tolist(),
        "theta_hat": result.theta_hat.tolist(),
        "converged": bool(result.converged),
        "num_iterations": int(result.num_iterations),
    }


# ---------------------------------------------------------------------------
# Monte Carlo driver
# ---------------------------------------------------------------------------

def calibrate_theta_star(base_cfg):
    """Draw theta* via healthy-DGP search on a CALIBRATION geography.

    The calibration seed is distinct from any rep seed, so theta* is
    independent of the rep draws.
    """
    dgp_cfg = base_cfg["dgp"]
    healthy_cfg = base_cfg["healthy_dgp"]
    # Dedicated calibration seeds (not used by any replication)
    calib_seeds = {
        "dgp": base_cfg["seeds"]["dgp"],  # geography for the search
        "search": base_cfg["seeds"]["search"],
    }
    theta_star, _, _, _ = generate_data(
        dgp_cfg, healthy_cfg, calib_seeds, verbose=True)
    return theta_star


def run_monte_carlo(n_reps=30, base_cfg=None, theta_star=None, out_dir=None):
    if base_cfg is None:
        base_cfg = CFG
    if out_dir is None:
        out_dir = BASE / "results"

    out_dir.mkdir(parents=True, exist_ok=True)

    dgp_cfg = base_cfg["dgp"]
    est_cfg = base_cfg["estimation"]

    C = dgp_cfg["C"]
    N = dgp_cfg["N"]
    fe_mode = dgp_cfg["fe_mode"]
    sigma = dgp_cfg["sigma"]
    M = C * (C - 1) // 2  # undirected
    n_shared = n_shared_covariates(C, fe_mode)
    n_p = n_params(C, N, fe_mode)

    if theta_star is None:
        theta_star = calibrate_theta_star(base_cfg)

    # ---- Seed audit ----
    all_seeds = audit_seeds(base_cfg, n_reps)
    logger.info("SEED AUDIT")
    logger.info(f"  dgp seed:   {base_cfg['seeds']['dgp']}")
    logger.info(f"  error seed: {base_cfg['seeds']['error']}")
    logger.info(f"  Disjoint streams per rep: geo, dgp_err, sim_err "
                f"(tags {SEED_TAG_DGP_GEO}, {SEED_TAG_DGP_ERR}, {SEED_TAG_SIM_ERR})")
    logger.info(f"  Total unique seed tuples: {3 * n_reps}")

    # ---- Theta* summary ----
    theta_rev = theta_star[:n_shared]
    theta_fc = theta_star[n_shared:n_shared + N]
    theta_gs = theta_star[-1]
    logger.info(f"\nMONTE CARLO: C={C}, M={M}, N={N}, S={est_cfg['n_simulations']}, "
                f"sigma={sigma}, n_params={n_p}, n_reps={n_reps}")
    logger.info(f"theta_rev:    {theta_rev}")
    logger.info(f"theta_fc:     {theta_fc}")
    logger.info(f"theta_gs:     {theta_gs:.4f}")

    # ---- Run replications ----
    t0 = time.perf_counter()
    rep_results = []
    n_unhealthy = 0
    n_non_converged = 0

    for rep in range(n_reps):
        t_rep = time.perf_counter()
        res = run_one_rep(rep, theta_star, base_cfg)
        dt = time.perf_counter() - t_rep
        res["time_s"] = dt

        if not res["healthy"]:
            n_unhealthy += 1
            logger.info(f"  rep {rep:>3d}: UNHEALTHY (bundles "
                        f"min={min(res['bundle_sizes'])}, "
                        f"max={max(res['bundle_sizes'])})")
        elif res["theta_hat"] is None:
            n_non_converged += 1
            logger.info(f"  rep {rep:>3d}: ESTIMATION FAILED")
        else:
            theta_hat = np.array(res["theta_hat"])
            err_pct = np.abs(theta_hat - theta_star) / np.abs(theta_star) * 100
            conv_str = "conv" if res["converged"] else "NOT-CONV"
            if not res["converged"]:
                n_non_converged += 1
            logger.info(f"  rep {rep:>3d}: {conv_str}, "
                        f"iters={res['num_iterations']:>3d}, "
                        f"bundles={min(res['bundle_sizes'])}-"
                        f"{max(res['bundle_sizes'])}, "
                        f"max_err={err_pct.max():.1f}%, "
                        f"time={dt:.1f}s")
        rep_results.append(res)

    t_total = time.perf_counter() - t0

    # ---- Analyze converged replications only ----
    good = [r for r in rep_results
            if r["healthy"] and r["theta_hat"] is not None and r["converged"]]
    n_good = len(good)

    if n_good == 0:
        logger.error("No converged replications — cannot compute statistics.")
        return None

    estimates = np.array([r["theta_hat"] for r in good])
    deviations = estimates - theta_star

    hat_means = estimates.mean(axis=0)
    hat_stds = estimates.std(axis=0, ddof=1) if n_good > 1 else np.zeros(n_p)
    hat_ses = hat_stds / np.sqrt(n_good) if n_good > 1 else np.zeros(n_p)
    bias = deviations.mean(axis=0)
    rmse = np.sqrt((deviations ** 2).mean(axis=0))
    abs_err_pct = np.abs(deviations) / np.abs(theta_star) * 100
    mae_pct = abs_err_pct.mean(axis=0)
    mdae_pct = np.median(abs_err_pct, axis=0)

    all_iters = [r["num_iterations"] for r in good]

    # ---- Parameter names ----
    shared_names = ["theta_rev"]
    if fe_mode == "origin":
        shared_names += [f"theta_fe_{j}" for j in range(C)]
    fc_names = [f"theta_fc_{i}" for i in range(N)]
    param_names = shared_names + fc_names + ["theta_gs"]

    # ---- Table ----
    W = 90
    print(f"\n{'=' * W}")
    print(f"  MONTE CARLO RESULTS")
    print(f"  C={C}, M={M}, N={N}, n_params={n_p}, "
          f"S={est_cfg['n_simulations']}, sigma={sigma}")
    print(f"  Replications: {n_good}/{n_reps} converged  "
          f"({n_unhealthy} unhealthy, {n_non_converged} non-converged)")
    print(f"  Runtime: {t_total:.0f}s total, "
          f"{t_total / n_reps:.1f}s/rep  ||  "
          f"Mean iters: {np.mean(all_iters):.1f}")
    print(f"{'=' * W}")
    header = (f"  {'Param':<14} {'True':>10} {'Mean':>10} {'Bias':>10} "
              f"{'Std':>10} {'RMSE':>10} {'MAE%':>7} {'MdAE%':>7}")
    print(header)
    print(f"  {'-' * (len(header) - 2)}")

    for j, name in enumerate(param_names):
        print(f"  {name:<14} {theta_star[j]:>10.4f} {hat_means[j]:>10.4f} "
              f"{bias[j]:>+10.4f} {hat_stds[j]:>10.4f} "
              f"{rmse[j]:>10.4f} {mae_pct[j]:>6.1f}% {mdae_pct[j]:>6.1f}%")

    if N > 1:
        print(f"  {'-' * (len(header) - 2)}")
        fc_sl = slice(n_shared, n_shared + N)
        print(f"  {'avg(fc)':<14} {'':>10} {'':>10} "
              f"{bias[fc_sl].mean():>+10.4f} {hat_stds[fc_sl].mean():>10.4f} "
              f"{rmse[fc_sl].mean():>10.4f} "
              f"{mae_pct[fc_sl].mean():>6.1f}% "
              f"{mdae_pct[fc_sl].mean():>6.1f}%")
    print()

    # ---- Save ----
    out = {
        "config": {
            "C": C, "M": M, "N": N, "fe_mode": fe_mode, "sigma": sigma,
            "n_params": n_p,
            "n_simulations": est_cfg["n_simulations"],
            "max_iters": est_cfg["max_iters"],
            "tolerance": est_cfg["tolerance"],
            "dgp_seed": base_cfg["seeds"]["dgp"],
            "error_seed": base_cfg["seeds"]["error"],
            "search_seed": base_cfg["seeds"]["search"],
        },
        "summary": {
            "n_reps_requested": n_reps,
            "n_reps_converged": n_good,
            "n_reps_unhealthy": n_unhealthy,
            "n_reps_non_converged": n_non_converged,
            "runtime_total_s": round(t_total, 1),
            "runtime_per_rep_s": round(t_total / n_reps, 2),
            "mean_row_gen_iters": float(np.mean(all_iters)),
        },
        "theta_true": theta_star.tolist(),
        "param_names": param_names,
        "statistics": {
            "mean": hat_means.tolist(),
            "bias": bias.tolist(),
            "std": hat_stds.tolist(),
            "se_of_mean": hat_ses.tolist(),
            "rmse": rmse.tolist(),
            "mae_pct": mae_pct.tolist(),
            "median_ae_pct": mdae_pct.tolist(),
        },
        "theta_hats": estimates.tolist(),
        "rep_details": [
            {k: v for k, v in r.items() if k != "theta_hat"}
            for r in rep_results
        ],
    }
    out_path = out_dir / "monte_carlo.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Saved to {out_path}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-reps", type=int,
                        default=CFG.get("replications", {}).get("n_reps", 30))
    parser.add_argument("--N", type=int, default=None)
    parser.add_argument("--C", type=int, default=None)
    parser.add_argument("--S", type=int, default=None)
    parser.add_argument("--sigma", type=float, default=None)
    args = parser.parse_args()

    cfg = {k: dict(v) if isinstance(v, dict) else v
           for k, v in CFG.items()}
    if args.N is not None:
        cfg["dgp"] = dict(cfg["dgp"]); cfg["dgp"]["N"] = args.N
    if args.C is not None:
        cfg["dgp"] = dict(cfg["dgp"]); cfg["dgp"]["C"] = args.C
    if args.S is not None:
        cfg["estimation"] = dict(cfg["estimation"])
        cfg["estimation"]["n_simulations"] = args.S
    if args.sigma is not None:
        cfg["dgp"] = dict(cfg["dgp"]); cfg["dgp"]["sigma"] = args.sigma

    run_monte_carlo(n_reps=args.n_reps, base_cfg=cfg)
