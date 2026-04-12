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

from applications.combinatorial_auction.scripts.c_block.counterfactual.prepare import (
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
            modular_regressors=app.get("modular_regressors"),
            quadratic_regressors=app.get("quadratic_regressors"),
            quadratic_id_regressors=app.get("quadratic_id_regressors"),
            elig_scale=app.get("elig_scale", 1.0),
        )

        bounds = config["row_generation"].setdefault("theta_bounds", {})
        lbs = bounds.setdefault("lbs", {})
        ubs = bounds.setdefault("ubs", {})

        for i in range(len(meta["beta"])):
            name = meta["covariate_names"][i]
            lbs[name] = float(meta["beta"][i])
            ubs[name] = float(meta["beta"][i])

        off = meta["n_id_mod"] + meta["n_item_mod"]
        for i in range(len(meta["gamma_id"])):
            name = meta["covariate_names"][off + i]
            lbs[name] = float(meta["gamma_id"][i])
            ubs[name] = float(meta["gamma_id"][i])
        off += meta["n_id_quad"]
        for i in range(len(meta["gamma_item"])):
            name = meta["covariate_names"][off + i]
            lbs[name] = float(meta["gamma_item"][i])
            ubs[name] = float(meta["gamma_item"][i])

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
        print(f"  Fixed: beta={meta['beta'].tolist()}")
        if len(meta["gamma_id"]) > 0:
            print(f"  Fixed: gamma_id={meta['gamma_id'].tolist()}")
        print(f"  Fixed: gamma_item={meta['gamma_item'].tolist()}")
        print(f"  {meta['covariate_names']}")
    else:
        input_data, meta = None, {}

    if comm is not None:
        config = comm.bcast(config, root=0)
        meta = comm.bcast(meta, root=0)

    import combest as ce
    from combest.estimation.callbacks import adaptive_gurobi_timeout

    callbacks = config.get("callbacks", {})
    config_name = Path(config_path).stem
    result_name = app.get("result_name", "result_counterfactual")

    for include_xi, label in [(True, "with_xi"), (False, "no_xi")]:
        if rank == 0:
            print(f"\n{'#'*60}")
            print(f"# Counterfactual: {label}")
            print(f"{'#'*60}")

        model = ce.Model()
        model.load_config(config)
        model.data.load_and_distribute_input_data(input_data)
        model.features.build_quadratic_covariates_from_data()

        _build_counterfactual_errors(model, meta, app.get("error_seed", 1998),
                                     error_scaling=app.get("error_scaling"),
                                     include_xi=include_xi,
                                     error_correlation=app.get("error_correlation"))

        model.subproblems.load_solver()

        pt_cb, _ = adaptive_gurobi_timeout(callbacks["row_gen"])
        result = model.row_generation.solve(iteration_callback=pt_cb, verbose=True)

        if rank == 0 and result is not None:
            out_name = f"{result_name}_{label}.json"
            _save(result, config, meta, experiment_dir / out_name)


def _build_counterfactual_errors(model, meta, seed, error_scaling=None,
                                  include_xi=True, error_correlation=None):
    from applications.combinatorial_auction.data.errors import (
        build_cholesky_factor, build_counterfactual_errors,
    )
    offset = meta["offset_m"] if include_xi else meta["offset_m_no_xi"]
    L_corr = build_cholesky_factor(error_correlation)
    local_errors = build_counterfactual_errors(
        model.features.comm_manager, meta["A"].shape[1], meta["A"], offset,
        seed, elig=meta.get("elig"), error_scaling=error_scaling, L_corr=L_corr,
    )
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
        "n_id_quad": app.get("n_id_quad", 0),
        "n_item_quad": app.get("n_item_quad", 0),
        "alpha_0": app["alpha_0"],
        "alpha_1": app["alpha_1"],
        "converged": bool(result.converged),
        "objective": float(result.final_objective),
        "iterations": int(result.num_iterations),
        "continental_mta_nums": [int(x) for x in meta["continental_mta_nums"]],
    }
    if result.u_hat is not None:
        out["u_hat"] = result.u_hat.tolist()
    json.dump(out, open(path, "w"), indent=2)
    print(f"Saved -> {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default=str(Path(__file__).parent / "config.yaml"))
    main(parser.parse_args().config)
