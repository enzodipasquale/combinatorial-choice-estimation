#!/usr/bin/env python3
"""Bootstrap welfare comparison: BTA vs MTA counterfactual."""
import json, sys, argparse, time
import numpy as np
from pathlib import Path

CBLOCK_DIR = Path(__file__).parent
APP_DIR = CBLOCK_DIR.parent.parent
sys.path.insert(0, str(APP_DIR.parent.parent))

from applications.combinatorial_auction.data.loaders import (
    load_bta_data, build_context, load_aggregation_matrix, continental_mta_nums,
)
from applications.combinatorial_auction.scripts.c_block.analyze import tsls, _build_distant_stats

POP_THRESHOLD = 500_000


def run_2sls(delta, price, pop, hhinc, geo, sample="full"):
    if sample == "rural":
        mask = pop < POP_THRESHOLD
    else:
        mask = np.ones(len(pop), dtype=bool)

    n = mask.sum()
    d_s, p_s = delta[mask], price[mask]

    zm = np.array([pop[geo[j] > 500].mean() if (geo[j] > 500).any() else 0
                    for j in range(len(pop))])[mask]
    zh = np.array([hhinc[geo[j] > 500].mean() if (geo[j] > 500).any() else 0
                    for j in range(len(hhinc))])[mask]

    X = np.column_stack([np.ones(n), -p_s])
    Z = np.column_stack([zm, zh])
    b_iv, _, _, _ = tsls(X, d_s, Z)
    return b_iv[0], b_iv[1]  # alpha_0, alpha_1


def run_single_counterfactual(beta, gamma_id, gamma_item, alpha_0, alpha_1,
                               xi, input_data_template, meta, config_template,
                               n_sim_cf, error_seed, error_scaling=None,
                               error_correlation=None):
    import combest as ce
    from combest.estimation.callbacks import adaptive_gurobi_timeout

    A = meta["A"]
    n_mtas = A.shape[0]
    n_btas = A.shape[1]
    mta_sizes = A.sum(1)
    xi_m = A @ xi
    offset_m = mta_sizes * alpha_0 + xi_m

    if input_data_template is not None:
        input_data = {
            "id_data": {k: v.copy() if isinstance(v, np.ndarray) else v
                         for k, v in input_data_template["id_data"].items()},
            "item_data": {k: v.copy() if isinstance(v, np.ndarray) else v
                           for k, v in input_data_template["item_data"].items()},
        }
        input_data["item_data"]["modular"] = -alpha_1 * np.eye(n_mtas, dtype=np.float64)
    else:
        input_data = None

    config = json.loads(json.dumps(config_template))
    bounds = config["row_generation"]["theta_bounds"]
    lbs = bounds.setdefault("lbs", {})
    ubs = bounds.setdefault("ubs", {})
    names = meta["covariate_names"]

    for i in range(len(beta)):
        lbs[names[i]] = float(beta[i])
        ubs[names[i]] = float(beta[i])
    off = meta["n_id_mod"] + meta["n_item_mod"]
    for i in range(len(gamma_id)):
        lbs[names[off + i]] = float(gamma_id[i])
        ubs[names[off + i]] = float(gamma_id[i])
    off += meta["n_id_quad"]
    for i in range(len(gamma_item)):
        lbs[names[off + i]] = float(gamma_item[i])
        ubs[names[off + i]] = float(gamma_item[i])

    config["dimensions"].update(
        n_obs=meta["n_obs"], n_items=meta["n_items"],
        n_covariates=meta["n_covariates"],
        covariate_names=meta["covariate_names"],
    )
    config["application"].update(
        n_id_mod=meta["n_id_mod"], n_item_mod=meta["n_item_mod"],
        n_id_quad=meta["n_id_quad"], n_item_quad=meta["n_item_quad"],
    )

    model = ce.Model()
    model.load_config(config)
    model.data.load_and_distribute_input_data(input_data)
    model.features.build_quadratic_covariates_from_data()

    from applications.combinatorial_auction.data.errors import (
        build_cholesky_factor, build_counterfactual_errors,
    )
    L_corr = build_cholesky_factor(error_correlation)
    local_errors = build_counterfactual_errors(
        model.features.comm_manager, n_btas, A, offset_m, error_seed,
        elig=meta.get("elig"), error_scaling=error_scaling, L_corr=L_corr,
    )
    model.features.local_modular_errors = local_errors
    model.features._error_oracle = lambda b, ids: (model.features.local_modular_errors[ids] * b).sum(-1)
    model.features._error_oracle_takes_data = False

    model.subproblems.load_solver()

    callbacks = config.get("callbacks", {})
    pt_cb, _ = adaptive_gurobi_timeout(callbacks["row_gen"])
    result = model.row_generation.solve(iteration_callback=pt_cb, verbose=False)

    if result is None:
        return None, None

    prices = result.theta_hat[meta["n_id_mod"]:meta["n_id_mod"] + meta["n_item_mod"]]
    revenue = prices.sum()
    u_hat = result.u_hat
    n_obs = meta["n_obs"]
    n_sim = len(u_hat) // n_obs
    net_surplus = u_hat.reshape(n_obs, n_sim).mean(1).sum() / alpha_1

    return revenue, net_surplus


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bootstrap_result", help="Path to bootstrap_result.json")
    parser.add_argument("--boot-config", default=None,
                        help="YAML config used for this bootstrap run (used to derive spec). "
                             "Falls back to 'config' field inside bootstrap_result.json.")
    parser.add_argument("--n-sim-cf", type=int, default=5,
                        help="Number of simulations for counterfactual (default: 5)")
    parser.add_argument("--iv-sample", default="full", choices=["full", "rural"],
                        help="IV sample (default: full)")
    parser.add_argument("--error-seed", type=int, default=24)
    args = parser.parse_args()

    try:
        from mpi4py import MPI
        comm, rank = MPI.COMM_WORLD, MPI.COMM_WORLD.Get_rank()
    except ImportError:
        comm, rank = None, 0

    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    boot = json.load(open(CBLOCK_DIR / args.bootstrap_result))
    boot_thetas = [np.array(t) for t in boot["bootstrap_thetas"]]
    boot_u_hats = [np.array(u) for u in boot["bootstrap_u_hat"]]
    n_boot = len(boot_thetas)

    # Derive spec from --boot-config YAML, or from 'config' field saved in boot JSON
    import yaml as _yaml
    if args.boot_config is not None:
        _cfg = _yaml.safe_load(open(args.boot_config))
    elif "config" in boot:
        _cfg = boot["config"]
    else:
        raise ValueError("No spec info: pass --boot-config <yaml> or re-run with updated estimate.py that saves config.")

    _app = _cfg["application"]
    modular_regressors    = _app.get("modular_regressors", ["elig_pop"])
    quadratic_regressors  = _app.get("quadratic_regressors", [])
    quadratic_id_regressors = _app.get("quadratic_id_regressors", [])
    error_correlation     = _app.get("error_correlation", None)
    n_id_mod  = len(modular_regressors)
    n_btas    = 480
    n_id_quad = len(quadratic_id_regressors)

    # Synthetic est_result dict for prepare_counterfactual (no file needed)
    est_result_dict = {
        "theta_hat": boot["theta_hat"],
        "n_id_mod": n_id_mod,
        "n_btas": n_btas,
        "n_id_quad": n_id_quad,
        "specification": {
            "modular": modular_regressors,
            "quadratic": quadratic_regressors,
            "quadratic_id": quadratic_id_regressors,
        },
    }

    raw = load_bta_data()
    ctx = build_context(raw)
    pop = raw["bta_data"]["pop90"].to_numpy().astype(float)
    hhinc = raw["bta_data"]["hhinc35k"].to_numpy().astype(float)
    price_bta = raw["bta_data"]["bid"].to_numpy().astype(float) / 1e9
    geo = raw["geo_distance"]
    b_obs = ctx["c_obs_bundles"]
    bta_revenue = (b_obs @ price_bta).sum()

    if rank == 0:
        from applications.combinatorial_auction.scripts.c_block.counterfactual.prepare import (
            prepare_counterfactual,
        )
        # Use point estimate theta to build template structure; alpha values overridden per draw
        theta_pt = np.array(boot["theta_hat"])
        delta_pt = -theta_pt[n_id_mod:n_id_mod + n_btas]
        a0_pt, a1_pt = run_2sls(delta_pt, price_bta, pop, hhinc, geo, args.iv_sample)

        input_data_template, meta = prepare_counterfactual(
            est_result_path_or_dict=est_result_dict,
            alpha_0=a0_pt, alpha_1=a1_pt,
            modular_regressors=modular_regressors,
            quadratic_regressors=quadratic_regressors,
            quadratic_id_regressors=quadratic_id_regressors,
        )
    else:
        input_data_template, meta = None, {}

    if comm is not None:
        meta = comm.bcast(meta, root=0)

    # CF config template
    cf_config = {
        "application": {"mode": "estimation"},
        "dimensions": {"n_simulations": args.n_sim_cf},
        "subproblem": {"name": "QuadraticKnapsackGRB",
                       "gurobi_params": {"TimeLimit": 1.0}},
        "row_generation": {
            "max_iters": 200, "tolerance": 0.01,
            "master_gurobi_params": {"Method": 0},
            "theta_bounds": {"lb": 0, "ub": 1.0},
        },
        "callbacks": {
            "row_gen": [{"iters": 50, "timeout": 1.0},
                        {"timeout": 10.0, "retire": True}],
        },
    }

    if rank == 0:
        n_obs = meta["n_obs"]
        print(f"Point estimate: a0={a0_pt:.4f}, a1={a1_pt:.4f}")
        print(f"  BTA revenue = ${bta_revenue:.4f}B")

    results = []
    for b in range(n_boot):
        t0 = time.perf_counter()
        theta_b = boot_thetas[b]

        # Extract components
        beta_b = theta_b[:n_id_mod]
        delta_b = -theta_b[n_id_mod:n_id_mod + n_btas]
        gamma_id_b = theta_b[n_id_mod + n_btas:n_id_mod + n_btas + n_id_quad]
        gamma_item_b = theta_b[n_id_mod + n_btas + n_id_quad:]

        # 2SLS
        a0_b, a1_b = run_2sls(delta_b, price_bta, pop, hhinc, geo, args.iv_sample)

        # BTA surplus from per-draw u_hat
        if rank == 0:
            u_hat_b = boot_u_hats[b]
            n_sim_est = len(u_hat_b) // n_obs
            bta_surplus_b = u_hat_b.reshape(n_obs, n_sim_est).mean(1).sum() / a1_b

        # Xi for this bootstrap draw
        xi_b = delta_b - a0_b + a1_b * price_bta

        # Run counterfactual
        cf_rev_b, cf_surplus_b = run_single_counterfactual(
            beta_b, gamma_id_b, gamma_item_b, a0_b, a1_b,
            xi_b, input_data_template, meta, cf_config,
            args.n_sim_cf, args.error_seed,
            error_correlation=error_correlation,
        )

        elapsed = time.perf_counter() - t0
        if rank == 0 and cf_rev_b is not None:
            results.append({
                "a0": a0_b, "a1": a1_b,
                "bta_surplus": bta_surplus_b,
                "bta_revenue": bta_revenue,
                "cf_revenue": cf_rev_b,
                "cf_surplus": cf_surplus_b,
            })
            print(f"  Boot {b+1}/{n_boot}: a1={a1_b:.2f}, "
                  f"BTA_S={bta_surplus_b:.2f}, CF_R={cf_rev_b:.2f}, "
                  f"CF_S={cf_surplus_b:.2f}  ({elapsed:.1f}s)")

    if rank == 0 and results:
        bta_surp = np.array([r["bta_surplus"] for r in results])
        bta_welf = np.array([r["bta_revenue"] + r["bta_surplus"] for r in results])
        cf_rev = np.array([r["cf_revenue"] for r in results])
        cf_surp = np.array([r["cf_surplus"] for r in results])
        cf_welf = cf_rev + cf_surp
        d_rev = (cf_rev / bta_revenue - 1) * 100
        d_surp = (cf_surp / bta_surp - 1) * 100
        d_welf = (cf_welf / bta_welf - 1) * 100

        print(f"\n{'='*70}")
        print(f"BOOTSTRAP WELFARE COMPARISON ({len(results)} samples)")
        print(f"{'='*70}")
        print(f"  {'':30s} {'Mean':>10s} {'SE':>10s}")
        print(f"  {'-'*52}")
        print(f"  {'BTA net surplus ($B)':30s} {bta_surp.mean():>10.2f} {bta_surp.std():>10.2f}")
        print(f"  {'BTA revenue ($B)':30s} {bta_revenue:>10.2f} {'---':>10s}")
        print(f"  {'BTA welfare ($B)':30s} {bta_welf.mean():>10.2f} {bta_welf.std():>10.2f}")
        print(f"  {'CF revenue ($B)':30s} {cf_rev.mean():>10.2f} {cf_rev.std():>10.2f}")
        print(f"  {'CF net surplus ($B)':30s} {cf_surp.mean():>10.2f} {cf_surp.std():>10.2f}")
        print(f"  {'CF welfare ($B)':30s} {cf_welf.mean():>10.2f} {cf_welf.std():>10.2f}")
        print(f"  {'-'*52}")
        print(f"  {'Delta revenue (%)':30s} {d_rev.mean():>10.1f} {d_rev.std():>10.1f}")
        print(f"  {'Delta surplus (%)':30s} {d_surp.mean():>10.1f} {d_surp.std():>10.1f}")
        print(f"  {'Delta welfare (%)':30s} {d_welf.mean():>10.1f} {d_welf.std():>10.1f}")

        # Save
        out = {
            "n_boot": len(results),
            "iv_sample": args.iv_sample,
            "results": results,
            "summary": {
                "bta_surplus": {"mean": float(bta_surp.mean()), "se": float(bta_surp.std())},
                "cf_revenue": {"mean": float(cf_rev.mean()), "se": float(cf_rev.std())},
                "cf_surplus": {"mean": float(cf_surp.mean()), "se": float(cf_surp.std())},
            },
        }
        outpath = CBLOCK_DIR / "bootstrap_welfare.json"
        json.dump(out, open(outpath, "w"), indent=2)
        print(f"\nSaved -> {outpath}")


if __name__ == "__main__":
    main()
