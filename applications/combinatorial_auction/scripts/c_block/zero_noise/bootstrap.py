#!/usr/bin/env python3
"""Bayesian bootstrap with deterministic elig_pop errors (no noise, n_sim=1)."""
import json, sys, yaml, argparse
import numpy as np
from pathlib import Path

ZERO_DIR = Path(__file__).parent
APP_DIR = ZERO_DIR.parent.parent.parent
sys.path.insert(0, str(APP_DIR.parent.parent))

try:
    from mpi4py import MPI
    comm, rank = MPI.COMM_WORLD, MPI.COMM_WORLD.Get_rank()
except ImportError:
    comm, rank = None, 0

from applications.combinatorial_auction.data.prepare import prepare


def main(config_path):
    config = yaml.safe_load(open(config_path))
    app = config["application"]
    out_dir = ZERO_DIR / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    if rank == 0:
        import warnings; warnings.filterwarnings("ignore", category=RuntimeWarning)
        input_data, meta = prepare(
            dataset=app["dataset"],
            modular_regressors=app.get("modular_regressors", []),
            quadratic_regressors=app.get("quadratic_regressors", []),
            quadratic_id_regressors=app.get("quadratic_id_regressors", []),
            item_modular=app.get("item_modular", "fe"),
        )
        meta.pop("raw", None)

        config["dimensions"].update(
            n_obs=meta["n_obs"], n_items=meta["n_items"],
            n_covariates=meta["n_covariates"],
            covariate_names=meta["covariate_names"],
        )
        app.update(
            n_id_mod=meta["n_id_mod"], n_item_mod=meta["n_item_mod"],
            n_id_quad=meta["n_id_quad"], n_item_quad=meta["n_item_quad"],
        )
        print(f"zero_noise bootstrap ({app['error_mode']}): {meta['n_obs']} obs, "
              f"{meta['n_items']} items, {meta['n_covariates']} cov")
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

    # deterministic errors: elig_i * pop_j
    cm = model.features.comm_manager
    n_items = model.n_items
    local_errors = np.zeros((cm.num_local_agent, n_items))
    if app["error_mode"] == "elig_pop":
        elig = model.data.local_data.id_data["elig"]
        pop = model.data.local_data.item_data["weight"].astype(float)
        pop = pop / pop.sum()
        for i in range(cm.num_local_agent):
            local_errors[i] = elig[i] * pop
    model.features.local_modular_errors = local_errors
    model.features._error_oracle = lambda b, ids: (model.features.local_modular_errors[ids] * b).sum(-1)
    model.features._error_oracle_takes_data = False

    model.subproblems.load_solver()

    callbacks = config.get("callbacks", {})
    boot_cfg = config.get("bootstrap", {})

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
            master.strip_slack_constraints(
                percentile=strip["percentile"],
                hard_threshold=strip["hard_threshold"],
            )

    se = model.standard_errors.compute_distributed_bootstrap(
        num_bootstrap=boot_cfg.get("num_samples", 2),
        seed=boot_cfg.get("seed", 54),
        verbose=True,
        pt_estimate_callbacks=(None, pt_cb),
        bootstrap_callback=boot_callback,
        method="bayesian",
        save_model_dir=str(out_dir / "checkpoints"),
        load_model_dir=str(out_dir / "checkpoints"),
    )

    if rank == 0 and se is not None:
        out = {
            "theta_hat": se.mean.tolist(),
            "se": se.se.tolist(),
            "bootstrap_thetas": se.samples.tolist(),
            "bootstrap_u_hat": se.u_samples.tolist(),
            "converged": se.converged.tolist(),
            "config": config,
        }
        json.dump(out, open(out_dir / "bootstrap_result.json", "w"), indent=2)
        print(f"Saved -> {out_dir / 'bootstrap_result.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default=str(ZERO_DIR / "config.yaml"))
    main(parser.parse_args().config)
