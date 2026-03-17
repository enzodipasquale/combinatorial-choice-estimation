#!/usr/bin/env python3
"""Counterfactual: C-block bidders on MTAs."""
import json, sys, yaml, argparse
import numpy as np
from pathlib import Path

APP_DIR = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(APP_DIR.parent.parent))

try:
    from mpi4py import MPI
    comm, rank = MPI.COMM_WORLD, MPI.COMM_WORLD.Get_rank()
except ImportError:
    comm, rank = None, 0

from applications.combinatorial_auction.specs.c_block.counterfactual.prepare import (
    prepare_counterfactual,
)


def main(config_path):
    config = yaml.safe_load(open(config_path))
    experiment_dir = Path(config_path).resolve().parent
    app = config["application"]

    if rank == 0:
        import warnings; warnings.filterwarnings("ignore", category=RuntimeWarning)

        est_path = (experiment_dir / app["est_result"]).resolve()
        input_data, meta = prepare_counterfactual(
            est_result_path=est_path,
            alpha_0=app["alpha_0"],
            alpha_1=app["alpha_1"],
            modular_regressors=app.get("modular_regressors", ["elig_pop"]),
            quadratic_regressors=app.get("quadratic_regressors",
                ["adjacency", "pop_centroid_delta4", "travel_survey", "air_travel"]),
        )

        # inject fixed-parameter bounds from estimation
        bounds = config["row_generation"].setdefault("theta_bounds", {})
        lbs = bounds.setdefault("lbs", {})
        ubs = bounds.setdefault("ubs", {})
        for i, name in enumerate(app.get("modular_regressors", [])):
            lbs[name] = float(meta["beta"][i])
            ubs[name] = float(meta["beta"][i])
        for i, name in enumerate(app.get("quadratic_regressors", [])):
            lbs[name] = float(meta["gamma"][i])
            ubs[name] = float(meta["gamma"][i])

        config["dimensions"].update(
            n_obs=meta["n_obs"], n_items=meta["n_items"],
            n_covariates=meta["n_covariates"],
            covariate_names=meta["covariate_names"],
        )
        app.update(
            n_id_mod=meta["n_id_mod"], n_item_mod=meta["n_item_mod"],
            n_id_quad=meta["n_id_quad"], n_item_quad=meta["n_item_quad"],
        )

        print(f"Counterfactual: {meta['n_obs']} C-block bidders, {meta['n_mtas']} MTAs")
        print(f"  alpha_0={app['alpha_0']:.4f}, alpha_1={app['alpha_1']:.4f}")
        print(f"  Fixed: beta={meta['beta'].tolist()}, gamma={meta['gamma'].tolist()}")
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

    # error oracle: BTA errors aggregated via A + structural offset
    _build_counterfactual_errors(model, meta, app.get("error_seed", 1998))

    model.subproblems.load_solver()

    callbacks = config.get("callbacks", {})
    pt_cb, _ = adaptive_gurobi_timeout(callbacks["row_gen"])
    result = model.row_generation.solve(iteration_callback=pt_cb, verbose=True)

    if rank == 0 and result is not None:
        _save(result, config, meta, experiment_dir / "result_counterfactual.json")


def _build_counterfactual_errors(model, meta, seed):
    A = meta["A"]
    n_bta = A.shape[1]
    offset_m = meta["offset_m"]
    cm = model.features.comm_manager

    local_errors = np.zeros((cm.num_local_agent, model.n_items))
    for i, gid in enumerate(cm.agent_ids):
        rng = np.random.default_rng((seed, gid))
        bta_err = rng.normal(0, 1, n_bta)
        local_errors[i] = bta_err @ A.T + offset_m

    model.features.local_modular_errors = local_errors
    model.features._error_oracle = lambda b, ids: (model.features.local_modular_errors[ids] * b).sum(-1)
    model.features._error_oracle_takes_data = False


def _save(result, config, meta, path):
    app = config["application"]
    theta = result.theta_hat
    names = config["dimensions"]["covariate_names"]
    n_id_mod = app["n_id_mod"]
    n_item_mod = app["n_item_mod"]

    prices = theta[n_id_mod : n_id_mod + n_item_mod]

    print(f"\nCounterfactual MTA prices (n={len(prices)}):")
    print(f"  mean={prices.mean():.6f}, min={prices.min():.6f}, max={prices.max():.6f}")
    print(f"  total revenue = ${prices.sum():.4f}B")

    for idx in sorted(names):
        print(f"  {names[idx]}: {theta[int(idx)]:.4f}")

    out = {
        "theta_hat": theta.tolist(),
        "prices": prices.tolist(),
        "n_mtas": meta["n_mtas"],
        "n_obs": config["dimensions"]["n_obs"],
        "n_covariates": config["dimensions"]["n_covariates"],
        "n_id_mod": n_id_mod,
        "n_item_mod": n_item_mod,
        "alpha_0": app["alpha_0"],
        "alpha_1": app["alpha_1"],
        "converged": bool(result.converged),
        "objective": float(result.final_objective),
        "iterations": int(result.num_iterations),
        "continental_mta_nums": [int(x) for x in meta["continental_mta_nums"]],
    }
    json.dump(out, open(path, "w"), indent=2)
    print(f"Saved -> {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default=str(Path(__file__).parent / "config.yaml"))
    main(parser.parse_args().config)
