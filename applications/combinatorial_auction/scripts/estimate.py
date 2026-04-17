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
import numpy as np

APP_ROOT = Path(__file__).resolve().parent.parent       # .../combinatorial_auction
REPO_ROOT = APP_ROOT.parent.parent                       # repo root
sys.path.insert(0, str(REPO_ROOT))

try:
    from mpi4py import MPI
    _comm = MPI.COMM_WORLD
    _rank = _comm.Get_rank()
except ImportError:
    _comm = None
    _rank = 0

from applications.combinatorial_auction.data.prepare import prepare
from applications.combinatorial_auction.data.loaders import load_raw, build_context
from applications.combinatorial_auction.scripts import errors


def _results_dir(config_path, kind):
    """kind: 'point_estimate' or 'bootstrap'."""
    d = APP_ROOT / "results" / Path(config_path).stem / kind
    d.mkdir(parents=True, exist_ok=True)
    return d


def _prepare_inputs(app):
    """Rank-0 work: run prepare() and build the error-artifact trio (ctx, cov, pop).

    Returns (input_data, meta, cov, pop_vec). pop_vec is None unless
    error_scaling=='pop'.
    """
    input_data, meta = prepare(
        modular_regressors       = app.get("modular_regressors", []),
        quadratic_regressors     = app.get("quadratic_regressors", []),
        quadratic_id_regressors  = app.get("quadratic_id_regressors", []),
        winners_only             = app.get("winners_only", False),
        capacity_source          = app.get("capacity_source", "initial"),
    )
    ctx = build_context(load_raw())
    cov = errors.covariance(ctx, app)
    pop_vec = errors.pop_vector(ctx) if app.get("error_scaling") == "pop" else None
    return input_data, meta, cov, pop_vec


def _build_model(config, input_data, meta, cov, pop_vec):
    """Every-rank work: construct combest model, install errors, load solver."""
    app = config["application"]
    # Let combest see dimensions/names (it prints them and uses them for bounds).
    config["dimensions"].update(
        n_obs=meta["n_obs"], n_items=meta["n_items"],
        n_covariates=meta["n_covariates"],
        covariate_names=meta["covariate_names"],
    )
    app.update(n_id_mod=meta["n_id_mod"], n_item_mod=meta["n_item_mod"],
               n_id_quad=meta["n_id_quad"], n_item_quad=meta["n_item_quad"])

    import combest as ce
    model = ce.Model()
    model.load_config(config)
    model.data.load_and_distribute_input_data(input_data)
    model.features.build_quadratic_covariates_from_data()

    errors.install(model, seed=app["error_seed"], cov=cov,
              scaling=app.get("error_scaling"), pop=pop_vec)

    model.subproblems.load_solver()
    return model


def _row_gen_callback(callbacks_cfg):
    from combest.estimation.callbacks import adaptive_gurobi_timeout
    cb, _ = adaptive_gurobi_timeout(callbacks_cfg["row_gen"])
    return cb


def _bootstrap_callback(callbacks_cfg):
    from combest.estimation.callbacks import adaptive_gurobi_timeout
    _, dist_cb = adaptive_gurobi_timeout(callbacks_cfg["boot"])
    strip = callbacks_cfg.get("boot_strip")

    def callback(it, boot, master):
        dist_cb(it, boot, master)
        if master is not None and it == 0 and strip is not None:
            master.strip_slack_constraints(
                percentile=strip["percentile"],
                hard_threshold=strip["hard_threshold"],
            )
    return callback


def _apply_se_overrides(model, boot_cfg):
    """Apply per-bootstrap row-generation overrides into the combest config."""
    overrides = {k: boot_cfg[k] for k in
                 ("rowgen_max_iters", "rowgen_tol", "rowgen_min_iters")
                 if k in boot_cfg}
    if not overrides:
        return
    d = model.config.standard_errors.__dict__.copy()
    d.update(overrides)
    model.config.standard_errors = type(model.config.standard_errors)(**d)


def _save(path, payload, config, meta):
    """Write a results JSON with config + meta attached."""
    payload = dict(payload)
    payload["config"] = config
    payload["meta"] = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                       for k, v in meta.items()}
    payload["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    json.dump(payload, open(path, "w"), indent=2)


def _run_point(model, config, meta, out_path):
    cb = _row_gen_callback(config.get("callbacks", {}))
    result = model.row_generation.solve(iteration_callback=cb, verbose=True)
    if result is None:
        return
    payload = {
        "theta_hat":  result.theta_hat.tolist(),
        "u_hat":      result.u_hat.tolist() if result.u_hat is not None else None,
        "xbar":       result.xbar.tolist()  if result.xbar  is not None else None,
        "converged":  bool(result.converged),
        "objective":  float(result.final_objective),
        "iterations": int(result.num_iterations),
        "runtime":    float(getattr(result, "runtime", 0.0)),
    }
    _save(out_path, payload, config, meta)
    print(f"Saved -> {out_path}")


def _run_bootstrap(model, config, meta, out_dir):
    boot_cfg = config.get("bootstrap", {})
    _apply_se_overrides(model, boot_cfg)
    pt_cb   = _row_gen_callback(config.get("callbacks", {}))
    boot_cb = _bootstrap_callback(config.get("callbacks", {}))

    se = model.standard_errors.compute_distributed_bootstrap(
        num_bootstrap           = boot_cfg.get("num_samples", 2),
        seed                    = boot_cfg.get("seed", 54),
        verbose                 = True,
        pt_estimate_callbacks   = (None, pt_cb),
        bootstrap_callback      = boot_cb,
        method                  = "bayesian",
        save_model_dir          = str(out_dir / "checkpoints"),
        load_model_dir          = str(out_dir / "checkpoints"),
    )
    if se is None:
        return
    payload = {
        "theta_hat":         se.mean.tolist(),
        "se":                se.se.tolist(),
        "bootstrap_thetas":  se.samples.tolist(),
        "bootstrap_u_hat":   se.u_samples.tolist(),
        "converged":         se.converged.tolist(),
    }
    # xbar, if a prior point-estimate was saved, goes along for second-stage.
    pt = out_dir.parent / "point_estimate" / "result.json"
    if pt.exists():
        with open(pt) as f:
            d = json.load(f)
        if d.get("xbar") is not None:
            payload["xbar"] = d["xbar"]
    _save(out_dir / "bootstrap_result.json", payload, config, meta)
    print(f"Saved -> {out_dir / 'bootstrap_result.json'}")


def main(config_path):
    config = yaml.safe_load(open(config_path))
    app = config["application"]
    mode = app.get("mode", "estimation")
    kind = "point_estimate" if mode == "estimation" else "bootstrap"
    out_dir = _results_dir(config_path, kind)

    if _rank == 0:
        import warnings; warnings.filterwarnings("ignore", category=RuntimeWarning)
        input_data, meta, cov, pop_vec = _prepare_inputs(app)
        print(f"{meta['n_obs']} obs, {meta['n_items']} items, "
              f"{meta['n_covariates']} cov")
    else:
        input_data = meta = cov = pop_vec = None

    if _comm is not None:
        config  = _comm.bcast(config,  root=0)
        meta    = _comm.bcast(meta,    root=0)
        cov     = _comm.bcast(cov,     root=0)
        pop_vec = _comm.bcast(pop_vec, root=0)

    model = _build_model(config, input_data, meta, cov, pop_vec)

    if mode == "estimation":
        _run_point(model, config, meta, out_dir / "result.json")
    elif mode == "bootstrap":
        _run_bootstrap(model, config, meta, out_dir)
    else:
        raise ValueError(f"mode must be 'estimation' or 'bootstrap', got {mode!r}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to experiment config.yaml")
    main(parser.parse_args().config)
