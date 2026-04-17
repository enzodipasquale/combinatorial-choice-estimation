#!/usr/bin/env python3
"""Point estimation and bootstrap for the C-block auction (MPI entry).

Usage:  mpirun -n N python -m applications.combinatorial_auction.scripts.estimate CONFIG.yaml

Config keys (application block):
    mode                'estimation' | 'bootstrap'
    modular_regressors / quadratic_regressors / quadratic_id_regressors  (lists)
    winners_only        bool
    capacity_source     'initial' | 'last_round'
    error_seed, error_correlation, spatial_rho, error_scaling  (see scripts.errors)

Results written to  <repo>/results/<config-stem>/point_estimate/result.json  or
                    <repo>/results/<config-stem>/bootstrap/bootstrap_result.json .
"""
import sys, json, yaml, argparse, time
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent       # .../combinatorial_auction
REPO_ROOT = APP_ROOT.parent.parent                       # repo root
sys.path.insert(0, str(REPO_ROOT))

try:
    from mpi4py import MPI
    _comm, _rank = MPI.COMM_WORLD, MPI.COMM_WORLD.Get_rank()
except ImportError:
    _comm, _rank = None, 0

from applications.combinatorial_auction.data.prepare import prepare
from applications.combinatorial_auction.data.loaders import load_raw, build_context
from applications.combinatorial_auction.scripts import errors


def _save(path, payload, config, meta):
    """Write a results JSON with config + meta attached."""
    json.dump({**payload, "config": config, "meta": meta,
               "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")},
              open(path, "w"), indent=2)


def main(config_path):
    from combest.estimation.callbacks import adaptive_gurobi_timeout
    import combest as ce

    config = yaml.safe_load(open(config_path))
    app    = config["application"]
    mode   = app.get("mode", "estimation")
    kind   = "point_estimate" if mode == "estimation" else "bootstrap"
    out_dir = APP_ROOT / "results" / Path(config_path).stem / kind
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Rank-0: build inputs and error artifacts (broadcast below) ────
    if _rank == 0:
        import warnings; warnings.filterwarnings("ignore", category=RuntimeWarning)
        input_data, meta = prepare(
            modular_regressors      = app.get("modular_regressors", []),
            quadratic_regressors    = app.get("quadratic_regressors", []),
            quadratic_id_regressors = app.get("quadratic_id_regressors", []),
            winners_only            = app.get("winners_only", False),
            capacity_source         = app.get("capacity_source", "initial"),
        )
        ctx     = build_context(load_raw())
        cov     = errors.covariance(ctx, app)
        pop_vec = errors.pop_vector(ctx) if app.get("error_scaling") == "pop" else None
        print(f"{meta['n_obs']} obs, {meta['n_items']} items, {meta['n_covariates']} cov")
    else:
        input_data = meta = cov = pop_vec = None

    if _comm is not None:
        config, meta, cov, pop_vec = _comm.bcast((config, meta, cov, pop_vec), root=0)

    # ── Every rank: build combest model, install errors ──────────────
    config["dimensions"].update(
        n_obs=meta["n_obs"], n_items=meta["n_items"],
        n_covariates=meta["n_covariates"], covariate_names=meta["covariate_names"],
    )
    app.update(n_id_mod=meta["n_id_mod"], n_item_mod=meta["n_item_mod"],
               n_id_quad=meta["n_id_quad"], n_item_quad=meta["n_item_quad"])

    model = ce.Model()
    model.load_config(config)
    model.data.load_and_distribute_input_data(input_data)
    model.features.build_quadratic_covariates_from_data()
    errors.install(model, seed=app["error_seed"], cov=cov,
                   scaling=app.get("error_scaling"), pop=pop_vec)
    model.subproblems.load_solver()

    callbacks = config.get("callbacks", {})
    pt_cb, _ = adaptive_gurobi_timeout(callbacks["row_gen"])

    # ── Solve ────────────────────────────────────────────────────────
    if mode == "estimation":
        result = model.row_generation.solve(iteration_callback=pt_cb, verbose=True)
        if result is None:
            return
        _save(out_dir / "result.json", {
            "theta_hat":  result.theta_hat.tolist(),
            "u_hat":      result.u_hat.tolist() if result.u_hat is not None else None,
            "xbar":       result.xbar.tolist()  if result.xbar  is not None else None,
            "converged":  bool(result.converged),
            "objective":  float(result.final_objective),
            "iterations": int(result.num_iterations),
            "runtime":    float(getattr(result, "runtime", 0.0)),
        }, config, meta)
        print(f"Saved -> {out_dir / 'result.json'}")
        return

    if mode != "bootstrap":
        raise ValueError(f"mode must be 'estimation' or 'bootstrap', got {mode!r}")

    # Bootstrap: apply SE overrides, build boot callback, run, save.
    boot_cfg = config.get("bootstrap", {})
    overrides = {k: boot_cfg[k] for k in
                 ("rowgen_max_iters", "rowgen_tol", "rowgen_min_iters") if k in boot_cfg}
    if overrides:
        d = model.config.standard_errors.__dict__.copy(); d.update(overrides)
        model.config.standard_errors = type(model.config.standard_errors)(**d)

    _, dist_cb = adaptive_gurobi_timeout(callbacks["boot"])
    strip = callbacks.get("boot_strip")
    def boot_cb(it, boot, master):
        dist_cb(it, boot, master)
        if master is not None and it == 0 and strip is not None:
            master.strip_slack_constraints(percentile=strip["percentile"],
                                           hard_threshold=strip["hard_threshold"])

    se = model.standard_errors.compute_distributed_bootstrap(
        num_bootstrap         = boot_cfg.get("num_samples", 2),
        seed                  = boot_cfg.get("seed", 54),
        verbose               = True,
        pt_estimate_callbacks = (None, pt_cb),
        bootstrap_callback    = boot_cb,
        method                = "bayesian",
        save_model_dir        = str(out_dir / "checkpoints"),
        load_model_dir        = str(out_dir / "checkpoints"),
    )
    if se is None:
        return
    payload = {
        "theta_hat":        se.mean.tolist(),
        "se":               se.se.tolist(),
        "bootstrap_thetas": se.samples.tolist(),
        "bootstrap_u_hat":  se.u_samples.tolist(),
        "converged":        se.converged.tolist(),
    }
    # Carry xbar forward from a prior point-estimate run (for second stage).
    pt = out_dir.parent / "point_estimate" / "result.json"
    if pt.exists():
        xbar = json.load(open(pt)).get("xbar")
        if xbar is not None:
            payload["xbar"] = xbar
    _save(out_dir / "bootstrap_result.json", payload, config, meta)
    print(f"Saved -> {out_dir / 'bootstrap_result.json'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("config", help="Path to experiment config.yaml")
    main(p.parse_args().config)
