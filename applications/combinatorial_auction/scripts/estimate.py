#!/usr/bin/env python3
import json, sys, yaml, argparse
import numpy as np
from pathlib import Path

APP_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(APP_DIR.parent.parent))

try:
    from mpi4py import MPI
    comm, rank = MPI.COMM_WORLD, MPI.COMM_WORLD.Get_rank()
except ImportError:
    comm, rank = None, 0

from applications.combinatorial_auction.data.prepare import prepare


def main(config_path):
    config = yaml.safe_load(open(config_path))
    experiment_dir = Path(config_path).resolve().parent
    app = config["application"]

    if rank == 0:
        import warnings; warnings.filterwarnings("ignore", category=RuntimeWarning)
        input_data, meta = prepare(
            dataset=app["dataset"],
            modular_regressors=app.get("modular_regressors", ["elig_pop"]),
            quadratic_regressors=app.get("quadratic_regressors", ["adjacency", "pop_centroid_delta4", "travel_survey", "air_travel"]),
            quadratic_id_regressors=app.get("quadratic_id_regressors", []),
            item_modular=app.get("item_modular", "fe"),
            separate_ab_quadratics=app.get("separate_ab_quadratics", False),
        )
        meta.pop("raw", None)

        config["dimensions"].update(n_obs=meta["n_obs"], n_items=meta["n_items"],
                                    n_covariates=meta["n_covariates"],
                                    covariate_names=meta["covariate_names"])
        app.update(n_id_mod=meta["n_id_mod"], n_item_mod=meta["n_item_mod"],
                   n_id_quad=meta["n_id_quad"], n_item_quad=meta["n_item_quad"])
        if app["dataset"] == "joint":
            app["n_obs_c"] = meta.get("n_obs_c", 0)

        print(f"{app['dataset']} ({app.get('item_modular','fe')}): {meta['n_obs']} obs, {meta['n_items']} items, {meta['n_covariates']} cov")
        print(f"  {meta['covariate_names']}")
    else:
        input_data, meta = None, {}

    if comm is not None:
        config = comm.bcast(config, root=0)
        meta = comm.bcast(meta, root=0)

    import combest as ce
    from combest.estimation.callbacks import adaptive_gurobi_timeout

    model = ce.Model()
    model.load_config(config)
    model.data.load_and_distribute_input_data(input_data)
    model.features.build_quadratic_covariates_from_data()
    _build_error_oracle(model, app["dataset"], meta, app.get("error_seed", 1998),
                        error_scaling=app.get("error_scaling"),
                        error_correlation=app.get("error_correlation"))
    model.subproblems.load_solver()

    callbacks = config.get("callbacks", {})
    mode = app.get("mode", "estimation")

    if mode == "estimation":
        pt_cb, _ = adaptive_gurobi_timeout(callbacks["row_gen"])
        result = model.row_generation.solve(iteration_callback=pt_cb, verbose=True)
        if rank == 0 and result is not None:
            config_name = Path(config_path).stem
            suffix = config_name if config_name != "config" else ("FE" if app.get("item_modular", "fe") == "fe" else "noFE")
            _save(result, config, meta, experiment_dir / f"result_{suffix}.json")

    elif mode == "bootstrap":
        boot_cfg = config.get("bootstrap", {})
        # merge bootstrap rowgen settings into standard_errors config
        se_overrides = {}
        for key in ("rowgen_max_iters", "rowgen_tol", "rowgen_min_iters"):
            if key in boot_cfg:
                se_overrides[key] = boot_cfg[key]
        if se_overrides:
            se_dict = model.config.standard_errors.__dict__.copy()
            se_dict.update(se_overrides)
            model.config.standard_errors = type(model.config.standard_errors)(**se_dict)
        pt_cb, _ = adaptive_gurobi_timeout(callbacks["row_gen"])
        _, dist_cb = adaptive_gurobi_timeout(callbacks["boot"])

        def boot_callback(it, boot, master):
            dist_cb(it, boot, master)
            strip = callbacks.get("boot_strip")
            if master is not None and it == 0 and strip:
                master.strip_slack_constraints(percentile=strip["percentile"],
                                               hard_threshold=strip["hard_threshold"])

        se = model.standard_errors.compute_distributed_bootstrap(
            num_bootstrap=boot_cfg.get("num_samples", 2),
            seed=boot_cfg.get("seed", 54),
            verbose=True,
            pt_estimate_callbacks=(None, pt_cb),
            bootstrap_callback=boot_callback,
            method="bayesian",
            save_model_dir=str(experiment_dir / f"master_{Path(config_path).stem}"),
            load_model_dir=str(experiment_dir / f"master_{Path(config_path).stem}"),
        )
        if rank == 0 and se is not None:
            config_name = Path(config_path).stem
            boot_suffix = f"_{config_name}" if config_name != "config" else ""
            out = {
                "theta_hat": se.mean.tolist(),
                "se": se.se.tolist(),
                "bootstrap_thetas": se.samples.tolist(),
                "bootstrap_u_hat": se.u_samples.tolist(),
                "converged": se.converged.tolist(),
                "config": config,
            }
            json.dump(out, open(experiment_dir / f"bootstrap_result{boot_suffix}.json", "w"), indent=2)


def _build_error_oracle(model, dataset, meta, seed, error_scaling=None,
                        error_correlation=None):
    if dataset == "c_block":
        cov = None
        if error_correlation is not None:
            from applications.combinatorial_auction.data.registries import QUADRATIC
            from applications.combinatorial_auction.data.loaders import load_bta_data, build_context
            raw = load_bta_data()
            ctx = build_context(raw)
            Q = QUADRATIC[error_correlation](ctx)
            cov = (Q + Q.T) / 2
            np.fill_diagonal(cov, 1.0)
        model.features.build_local_modular_error_oracle(seed=seed, covariance_matrix=cov)
        if error_scaling == "elig":
            elig = model.data.local_data.id_data['elig']
            model.features.local_modular_errors *= elig[:, None]
        return

    A = meta["A"]
    n_bta = A.shape[1]
    n_obs_c = meta.get("n_obs_c", 0)
    cm = model.features.comm_manager

    local_errors = np.zeros((cm.num_local_agent, model.n_items))
    for i, gid in enumerate(cm.agent_ids):
        rng = np.random.default_rng((seed, gid))
        bta_err = rng.normal(0, 1, n_bta)
        if dataset == "joint" and gid % model.n_obs < n_obs_c:
            local_errors[i, :n_bta] = bta_err
        else:
            offset = n_bta if dataset == "joint" else 0
            local_errors[i, offset:offset + A.shape[0]] = bta_err @ A.T

    model.features.local_modular_errors = local_errors
    model.features._error_oracle = lambda b, ids: (model.features.local_modular_errors[ids] * b).sum(-1)
    model.features._error_oracle_takes_data = False


def _save(result, config, meta, path):
    import time as _time
    app = config["application"]
    theta = result.theta_hat
    names = config["dimensions"]["covariate_names"]

    print(f"\ntheta ({len(theta)} params):")
    for idx in sorted(names):
        print(f"  {names[idx]}: {theta[int(idx)]:.4f}")

    out = {
        # --- Estimates ---
        "theta_hat": theta.tolist(),
        "covariate_names": {str(k): v for k, v in names.items()},
        "converged": bool(result.converged),
        "objective": float(result.final_objective),
        "iterations": int(result.num_iterations),
        "runtime": float(result.runtime) if hasattr(result, "runtime") else None,
        # --- Dimensions ---
        "n_items": config["dimensions"]["n_items"],
        "n_obs": config["dimensions"]["n_obs"],
        "n_covariates": config["dimensions"]["n_covariates"],
        "n_simulations": config["dimensions"].get("n_simulations", 1),
        # --- Specification ---
        "dataset": app["dataset"],
        "item_modular": app.get("item_modular", "fe"),
        "specification": {k: app.get(f"{k}_regressors", []) for k in ["modular", "quadratic", "quadratic_id"]},
        "n_id_mod": app["n_id_mod"],
        "n_id_quad": app["n_id_quad"],
        # --- Error oracle ---
        "error_seed": app.get("error_seed"),
        "error_scaling": app.get("error_scaling"),
        "error_correlation": app.get("error_correlation"),
        # --- Estimation config ---
        "subproblem": config.get("subproblem", {}),
        "row_generation": config.get("row_generation", {}),
        "callbacks": config.get("callbacks", {}),
        "mode": app.get("mode", "estimation"),
        # --- Timestamp ---
        "timestamp": _time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if result.u_hat is not None:
        out["u_hat"] = result.u_hat.tolist()
    for k in ["n_btas", "n_mtas", "n_obs_c", "n_obs_ab", "continental_mta_nums"]:
        if k in meta:
            out[k] = [int(x) for x in meta[k]] if isinstance(meta[k], (list, np.ndarray)) and k == "continental_mta_nums" else meta[k]

    json.dump(out, open(path, "w"), indent=2)
    print(f"Saved -> {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to experiment config.yaml")
    main(parser.parse_args().config)
