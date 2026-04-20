#!/usr/bin/env python3
"""Per-draw CF welfare. For each bootstrap sample b, run 2SLS → (α₀_b, α₁_b),
solve the CF once (with_xi — algebraically "A·δ + α₁·A·p", invariant to the
demand-controls decomposition), and accumulate BTA vs MTA welfare statistics.

Usage:  mpirun -n N python -m applications.combinatorial_auction.scripts.counterfactual.bootstrap_welfare SPEC

Reads   results/<SPEC>/bootstrap/bootstrap_result.json.
Writes  results/<SPEC>/counterfactual/bootstrap_welfare.json.
"""
import sys, json, yaml, argparse, time
from pathlib import Path
import numpy as np

APP_ROOT  = Path(__file__).resolve().parent.parent.parent
REPO_ROOT = APP_ROOT.parent.parent
sys.path.insert(0, str(REPO_ROOT))

try:
    from mpi4py import MPI
    _comm, _rank = MPI.COMM_WORLD, MPI.COMM_WORLD.Get_rank()
except ImportError:
    _comm, _rank = None, 0

from applications.combinatorial_auction.data.loaders import load_raw, build_context
from applications.combinatorial_auction.scripts.second_stage.iv import run_2sls
from applications.combinatorial_auction.scripts.counterfactual.run import solve_cf


def _welfare(result, meta, alpha_1):
    """(revenue, net_surplus) in $B from a CF solve result."""
    prices = result.theta_hat[meta["n_id_mod"]:meta["n_id_mod"] + meta["n_mtas"]]
    u, n_obs = result.u_hat, meta["n_obs"]
    net_surplus = u.reshape(n_obs, len(u) // n_obs).mean(1).sum() / alpha_1
    return float(prices.sum()), float(net_surplus)


def main(spec, *, configs_dir=None, results_dir=None, out_dir=None):
    cfg_dir = Path(configs_dir) if configs_dir else APP_ROOT / "configs"
    res_dir = Path(results_dir) if results_dir else APP_ROOT / "results"
    app = yaml.safe_load(open(cfg_dir / f"{spec}.yaml"))["application"]
    out_dir = Path(out_dir) if out_dir else (res_dir / spec / "counterfactual")
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = load_raw() if _rank == 0 else None  # cached for the 2SLS loop below
    if _rank == 0:
        import warnings; warnings.filterwarnings("ignore", category=RuntimeWarning)
        boot = json.load(open(res_dir / spec / "bootstrap" / "bootstrap_result.json"))
        boot_thetas = np.asarray(boot["bootstrap_thetas"])
        boot_u_hats = np.asarray(boot["bootstrap_u_hat"])
        ctx         = build_context(raw)
        price       = raw["bta_data"]["bid"].to_numpy(dtype=float) / 1e9
        bta_revenue = float((ctx["c_obs_bundles"] @ price).sum())
    else:
        boot_thetas = boot_u_hats = price = bta_revenue = None

    if _comm is not None:
        boot_thetas, boot_u_hats, price, bta_revenue = _comm.bcast(
            (boot_thetas, boot_u_hats, price, bta_revenue), root=0)

    n_id_mod, n_btas = len(app.get("modular_regressors", [])), len(price)
    rows = []

    for b in range(len(boot_thetas)):
        t0 = time.perf_counter()
        theta_b = boot_thetas[b]
        delta_b = -theta_b[n_id_mod:n_id_mod + n_btas]

        if _rank == 0:
            iv = run_2sls(delta_b, raw, app)
            a0_b, a1_b, dc_b = iv["a0"], iv["a1"], iv["demand_controls"]
        else:
            a0_b = a1_b = dc_b = None
        if _comm is not None:
            a0_b, a1_b, dc_b = _comm.bcast((a0_b, a1_b, dc_b), root=0)

        result, meta = solve_cf(theta_b, app,
                                alpha_0=a0_b, alpha_1=a1_b, demand_controls=dc_b,
                                include_xi=True, verbose=False)
        if _rank == 0 and result is not None:
            cf_rev, cf_surp = _welfare(result, meta, a1_b)
            u_b = boot_u_hats[b]
            n_sim_est = len(u_b) // meta["n_obs"]
            bta_surp = float(u_b.reshape(meta["n_obs"], n_sim_est).mean(1).sum() / a1_b)
            rows.append(dict(a0=a0_b, a1=a1_b, bta_surplus=bta_surp,
                             bta_revenue=bta_revenue,
                             cf_revenue=cf_rev, cf_surplus=cf_surp))
            print(f"  Boot {b+1}/{len(boot_thetas)}: a1={a1_b:.2f}  "
                  f"BTA_S={bta_surp:.2f}  CF_R={cf_rev:.2f}  CF_S={cf_surp:.2f}  "
                  f"({time.perf_counter()-t0:.1f}s)")

    if _rank == 0 and rows:
        R = {k: np.array([r[k] for r in rows]) for k in ("bta_surplus", "cf_revenue", "cf_surplus")}
        bta_welf = R["bta_surplus"] + bta_revenue
        cf_welf  = R["cf_revenue"]  + R["cf_surplus"]
        def _ms(v): return {"mean": float(v.mean()), "se": float(v.std())}
        summary = {
            "bta_surplus": _ms(R["bta_surplus"]),
            "bta_revenue": {"mean": bta_revenue, "se": 0.0},
            "bta_welfare": _ms(bta_welf),
            "cf_revenue":  _ms(R["cf_revenue"]),
            "cf_surplus":  _ms(R["cf_surplus"]),
            "cf_welfare":  _ms(cf_welf),
            "delta_revenue_pct": {"mean": float(((R["cf_revenue"] / bta_revenue - 1) * 100).mean())},
            "delta_welfare_pct": {"mean": float(((cf_welf / bta_welf - 1) * 100).mean())},
        }
        out_path = out_dir / "bootstrap_welfare.json"
        json.dump({"n_boot": len(rows), "results": rows, "summary": summary},
                  open(out_path, "w"), indent=2)
        print(f"\nSaved -> {out_path}")
        for k, v in summary.items():
            print(f"  {k:>24}  {v['mean']:>8.2f}" + (f" ± {v['se']:.2f}" if v.get('se') else ""))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    main(ap.parse_args().spec)
