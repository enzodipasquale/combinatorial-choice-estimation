#!/usr/bin/env python3
"""Per-draw CF welfare. For each bootstrap sample b, re-run 2SLS to get
(α₀_b, α₁_b), solve the CF once (with_xi offset — algebraically equivalent to
"A·δ + α₁·A·p", invariant to the demand-controls decomposition), and
accumulate BTA vs MTA welfare statistics.

Usage:  mpirun -n N python -m applications.combinatorial_auction.pipeline.counterfactual.bootstrap_welfare SPEC

Writes  results/<SPEC>/bootstrap_welfare.json.
"""
import sys, json, yaml, argparse, time
from pathlib import Path
import numpy as np

APP_ROOT  = Path(__file__).resolve().parent.parent.parent
REPO_ROOT = APP_ROOT.parent.parent
sys.path.insert(0, str(REPO_ROOT))

try:
    from mpi4py import MPI
    _comm = MPI.COMM_WORLD
    _rank = _comm.Get_rank()
except ImportError:
    _comm = None
    _rank = 0

from applications.combinatorial_auction.data.loaders import load_raw, build_context
from applications.combinatorial_auction.pipeline.second_stage.iv import (
    simple_instruments, second_stage as run_2sls,
)
from applications.combinatorial_auction.pipeline import errors as E
from applications.combinatorial_auction.pipeline.counterfactual.run import solve_cf


def _welfare_from_result(result, meta, alpha_1):
    """(revenue, net_surplus) in $B."""
    n_id_mod, n_mtas = meta["n_id_mod"], meta["n_mtas"]
    prices = result.theta_hat[n_id_mod:n_id_mod + n_mtas]
    u = result.u_hat
    n_obs = meta["n_obs"]
    n_sim = len(u) // n_obs
    net_surplus = u.reshape(n_obs, n_sim).mean(1).sum() / alpha_1
    return float(prices.sum()), float(net_surplus)


def main(spec, *, configs_dir=None, results_dir=None, out_dir=None):
    cfg_dir = Path(configs_dir) if configs_dir else APP_ROOT / "configs"
    res_dir = Path(results_dir) if results_dir else APP_ROOT / "results"
    app = yaml.safe_load(open(cfg_dir / f"{spec}.yaml"))["application"]
    out_dir = Path(out_dir) if out_dir else (res_dir / spec)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Rank-0 setup: load bootstrap draws, 2SLS at point estimate, BTA revenue.
    if _rank == 0:
        import warnings; warnings.filterwarnings("ignore", category=RuntimeWarning)
        boot = json.load(open(res_dir / spec / "bootstrap_result.json"))
        boot_thetas = np.asarray(boot["bootstrap_thetas"])
        boot_u_hats = np.asarray(boot["bootstrap_u_hat"])
        raw = load_raw()
        ctx = build_context(raw)
        price = raw["bta_data"]["bid"].to_numpy(dtype=float) / 1e9
        b_obs = ctx["c_obs_bundles"]
        bta_revenue = float((b_obs @ price).sum())
        use_blp = app.get("error_scaling") == "pop"
        si_cache = None if use_blp else simple_instruments(raw)
        bta_cov = E.covariance(ctx, app)
    else:
        boot_thetas = boot_u_hats = None
        price = bta_revenue = bta_cov = si_cache = None
        use_blp = None

    if _comm is not None:
        boot_thetas = _comm.bcast(boot_thetas, root=0)
        boot_u_hats = _comm.bcast(boot_u_hats, root=0)
        price       = _comm.bcast(price,       root=0)
        bta_revenue = _comm.bcast(bta_revenue, root=0)
        bta_cov     = _comm.bcast(bta_cov,     root=0)
        use_blp     = _comm.bcast(use_blp,     root=0)
        si_cache    = _comm.bcast(si_cache,    root=0)

    # n_id_mod and n_btas are recoverable from θ via app regressors + price length.
    n_id_mod = len(app.get("modular_regressors", []))
    n_btas = len(price)
    n_boot = len(boot_thetas)

    rows = []
    for b in range(n_boot):
        t0 = time.perf_counter()
        theta_b = boot_thetas[b]
        delta_b = -theta_b[n_id_mod:n_id_mod + n_btas]

        # Rank 0 runs 2SLS and broadcasts (simple IV — already cached; BLP IV
        # is deterministic but non-trivial, so cache-and-broadcast is cheap).
        if _rank == 0:
            iv = run_2sls(delta_b, price, load_raw(), use_blp=use_blp,
                          simple_instruments_cached=si_cache,
                          pop_threshold=app.get("pop_threshold", 500_000))
            a0_b, a1_b = iv["a0"], iv["a1"]
            dc_b = iv["demand_controls"]
        else:
            a0_b = a1_b = None; dc_b = None
        if _comm is not None:
            a0_b, a1_b, dc_b = _comm.bcast((a0_b, a1_b, dc_b), root=0)

        result, meta = solve_cf(
            theta_b, app,
            alpha_0=a0_b, alpha_1=a1_b, demand_controls=dc_b,
            bta_cov=bta_cov, include_xi=True, verbose=False,
        )
        if _rank == 0 and result is not None:
            cf_rev, cf_surp = _welfare_from_result(result, meta, a1_b)
            u_b = boot_u_hats[b]
            n_sim_est = len(u_b) // meta["n_obs"]   # CF n_obs == est n_obs
            bta_surp = float(u_b.reshape(meta["n_obs"], n_sim_est).mean(1).sum() / a1_b)
            rows.append(dict(a0=a0_b, a1=a1_b,
                             bta_surplus=bta_surp, bta_revenue=bta_revenue,
                             cf_revenue=cf_rev, cf_surplus=cf_surp))
            print(f"  Boot {b+1}/{n_boot}: a1={a1_b:.2f}  "
                  f"BTA_S={bta_surp:.2f}  CF_R={cf_rev:.2f}  CF_S={cf_surp:.2f}  "
                  f"({time.perf_counter()-t0:.1f}s)")

    if _rank == 0 and rows:
        R = {k: np.array([r[k] for r in rows]) for k in ("bta_surplus", "cf_revenue", "cf_surplus")}
        bta_welf = R["bta_surplus"] + bta_revenue
        cf_welf  = R["cf_revenue"]  + R["cf_surplus"]
        summary = {
            "bta_surplus":  {"mean": float(R["bta_surplus"].mean()), "se": float(R["bta_surplus"].std())},
            "bta_revenue":  {"mean": bta_revenue, "se": 0.0},
            "bta_welfare":  {"mean": float(bta_welf.mean()),        "se": float(bta_welf.std())},
            "cf_revenue":   {"mean": float(R["cf_revenue"].mean()), "se": float(R["cf_revenue"].std())},
            "cf_surplus":   {"mean": float(R["cf_surplus"].mean()), "se": float(R["cf_surplus"].std())},
            "cf_welfare":   {"mean": float(cf_welf.mean()),         "se": float(cf_welf.std())},
            "delta_revenue_pct": {"mean": float((R["cf_revenue"]/bta_revenue - 1).mean() * 100)},
            "delta_welfare_pct": {"mean": float((cf_welf/bta_welf - 1).mean() * 100)},
        }
        out = {"n_boot": len(rows), "results": rows, "summary": summary}
        out_path = out_dir / "bootstrap_welfare.json"
        json.dump(out, open(out_path, "w"), indent=2)
        print(f"\nSaved -> {out_path}")
        for k, v in summary.items():
            if "se" in v:
                print(f"  {k:>24}  {v['mean']:>8.2f} ± {v['se']:.2f}")
            else:
                print(f"  {k:>24}  {v['mean']:>8.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    main(ap.parse_args().spec)
