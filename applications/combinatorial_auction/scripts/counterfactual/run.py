#!/usr/bin/env python3
"""Counterfactual: C-block bidders buying MTAs, α₀/α₁ from 2SLS.

Usage:  mpirun -n N python -m applications.combinatorial_auction.scripts.counterfactual.run SPEC

Reads   configs/<SPEC>.yaml, results/<SPEC>/point_estimate/result.json.
Writes  results/<SPEC>/counterfactual/cf_with_xi.json and cf_no_xi.json.

The 2SLS (IV choice, regressors, instruments, sample, thresholds) is fully
driven by the `counterfactual:` block of the spec YAML — see
`scripts.second_stage.iv` for the config keys.
"""
import sys, json, yaml, copy, argparse
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
from applications.combinatorial_auction.scripts.counterfactual.prepare import (
    prepare_counterfactual, freeze_bounds,
)
from applications.combinatorial_auction.scripts import errors

# Solver settings for every CF run. The estimation spec drives regressors and
# error model; the CF's own 2SLS knobs live in the counterfactual: config block.
CF_CONFIG = {
    "application":    {"mode": "estimation"},
    "dimensions":     {"n_simulations": 5},
    "subproblem":     {"name": "QuadraticKnapsackGRB",
                       "gurobi_params": {"TimeLimit": 1.0}},
    "row_generation": {"max_iters": 200, "tolerance": 0.01,
                       "master_gurobi_params": {"Method": 0},
                       "theta_bounds": {"lb": 0, "ub": 1.0}},
    "callbacks":      {"row_gen": [{"iters": 50, "timeout": 1.0},
                                   {"timeout": 10.0, "retire": True}]},
}
# Fixed seed for the CF errors; CF is deterministic for a given point estimate.
CF_ERROR_SEED = 24


def solve_cf(theta, app, *, alpha_0, alpha_1, demand_controls, bta_cov,
             include_xi, verbose=False):
    """Solve one CF (given (θ, α₀, α₁, demand_controls)); return (Result, meta).

    Shared by the point-estimate run (both xi variants) and by the per-draw
    bootstrap welfare loop.
    """
    import combest as ce
    from combest.estimation.callbacks import adaptive_gurobi_timeout

    input_data, meta, cf = prepare_counterfactual(
        theta, app, alpha_0=alpha_0, alpha_1=alpha_1, demand_controls=demand_controls,
    )
    config = copy.deepcopy(CF_CONFIG)
    freeze_bounds(config, meta, cf)
    config["dimensions"].update(
        n_obs=meta["n_obs"], n_items=meta["n_items"],
        n_covariates=meta["n_covariates"], covariate_names=meta["covariate_names"],
    )
    config["application"].update(
        n_id_mod=meta["n_id_mod"], n_item_mod=meta["n_item_mod"],
        n_id_quad=meta["n_id_quad"], n_item_quad=meta["n_item_quad"],
    )

    model = ce.Model()
    model.load_config(config)
    model.data.load_and_distribute_input_data(input_data)
    model.features.build_quadratic_covariates_from_data()
    errors.install_aggregated(
        model, seed=CF_ERROR_SEED, A=cf["A"], bta_cov=bta_cov,
        offset=cf["offset_m"] if include_xi else cf["offset_m_no_xi"],
        scaling=app.get("error_scaling"), pop=cf["pop"], elig=cf["elig"],
    )
    model.subproblems.load_solver()

    pt_cb, _ = adaptive_gurobi_timeout(config["callbacks"]["row_gen"])
    return model.row_generation.solve(iteration_callback=pt_cb, verbose=verbose), meta


def _derive_alphas(theta, raw, app):
    """δ = −θ_fe → 2SLS → (α₀, α₁, demand_controls)."""
    n_id_mod = len(app.get("modular_regressors", []))
    delta = -np.asarray(theta)[n_id_mod:n_id_mod + raw["bta_data"].shape[0]]
    iv = run_2sls(delta, raw, app)
    return iv["a0"], iv["a1"], iv["demand_controls"]


def _save(result, meta, alpha_0, alpha_1, path):
    theta  = result.theta_hat
    prices = theta[meta["n_id_mod"]:meta["n_id_mod"] + meta["n_mtas"]]
    out = {
        "theta_hat":            theta.tolist(),
        "prices":               prices.tolist(),
        "continental_mta_nums": [int(m) for m in meta["continental_mta_nums"]],
        "n_mtas":               int(meta["n_mtas"]),
        "n_obs":                int(meta["n_obs"]),
        "alpha_0":              float(alpha_0),
        "alpha_1":              float(alpha_1),
        "converged":            bool(result.converged),
        "objective":            float(result.final_objective),
    }
    if result.u_hat is not None:
        out["u_hat"] = result.u_hat.tolist()
    json.dump(out, open(path, "w"), indent=2)
    print(f"Saved -> {path}")


def main(spec, *, configs_dir=None, results_dir=None, out_dir=None):
    cfg_dir = Path(configs_dir) if configs_dir else APP_ROOT / "configs"
    res_dir = Path(results_dir) if results_dir else APP_ROOT / "results"
    cfg_path = cfg_dir / f"{spec}.yaml"
    pt_path  = res_dir / spec / "point_estimate" / "result.json"
    est_app  = yaml.safe_load(open(cfg_path))["application"]
    out_dir  = Path(out_dir) if out_dir else (res_dir / spec / "counterfactual")
    out_dir.mkdir(parents=True, exist_ok=True)

    if _rank == 0:
        import warnings; warnings.filterwarnings("ignore", category=RuntimeWarning)
        theta = np.array(json.load(open(pt_path))["theta_hat"])
        raw = load_raw()
        a0, a1, dc = _derive_alphas(theta, raw, est_app)
        bta_cov = errors.covariance(build_context(raw), est_app)
        print(f"Counterfactual[{spec}]: α₀={a0:.6f}  α₁={a1:.4f}  controls={dc}")
    else:
        theta = bta_cov = a0 = a1 = dc = None

    if _comm is not None:
        theta, bta_cov, a0, a1, dc = _comm.bcast((theta, bta_cov, a0, a1, dc), root=0)

    for include_xi, label in [(True, "with_xi"), (False, "no_xi")]:
        if _rank == 0:
            print(f"\n── Counterfactual: {label} ──")
        result, meta = solve_cf(theta, est_app,
                                alpha_0=a0, alpha_1=a1, demand_controls=dc,
                                bta_cov=bta_cov, include_xi=include_xi, verbose=True)
        if _rank == 0 and result is not None:
            _save(result, meta, a0, a1, out_dir / f"cf_{label}.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    main(ap.parse_args().spec)
