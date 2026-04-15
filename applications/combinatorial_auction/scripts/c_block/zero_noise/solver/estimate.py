#!/usr/bin/env python3
"""Zero-noise estimation using custom solver with fixed linear/quadratic terms.

The config specifies:
  - error_mode: which term has coefficient fixed at 1
    - "elig_pop": elig_i * pop_j in the linear part
    - "pop_centroid_delta4": Q matrix in the quadratic part
    - "adjacency": Q matrix in the quadratic part
  - modular/quadratic regressors: the estimated covariates (excluding the fixed one)
"""
import json, sys, yaml, argparse
import numpy as np
from pathlib import Path

SOLVER_DIR = Path(__file__).parent
ZERO_DIR = SOLVER_DIR.parent
sys.path.insert(0, str(ZERO_DIR.parent.parent.parent.parent.parent))
sys.path.insert(0, str(SOLVER_DIR))

try:
    from mpi4py import MPI
    comm, rank = MPI.COMM_WORLD, MPI.COMM_WORLD.Get_rank()
except ImportError:
    comm, rank = None, 0

from applications.combinatorial_auction.data.prepare import prepare
from applications.combinatorial_auction.data.registries import MODULAR, QUADRATIC


def _build_fixed_terms(model, ctx, error_mode):
    """Build fixed_linear and fixed_quadratic arrays based on error_mode."""
    cm = model.features.comm_manager
    n_local = cm.num_local_agent
    n_items = model.n_items
    fixed_linear = None
    fixed_quadratic = None

    if error_mode == "elig_pop":
        elig = model.data.local_data.id_data["elig"]
        pop = model.data.local_data.item_data["weight"].astype(float)
        pop = pop / pop.sum()
        fixed_linear = np.zeros((n_local, n_items))
        for i in range(n_local):
            fixed_linear[i] = elig[i] * pop

    elif error_mode in QUADRATIC:
        fixed_quadratic = QUADRATIC[error_mode](ctx)

    return fixed_linear, fixed_quadratic


def main(config_path):
    config = yaml.safe_load(open(config_path))
    app = config["application"]
    out_dir = ZERO_DIR / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    if rank == 0:
        import warnings; warnings.filterwarnings("ignore", category=RuntimeWarning)
        from applications.combinatorial_auction.data.loaders import load_bta_data, build_context

        raw = load_bta_data()
        ctx = build_context(raw)

        input_data, meta = prepare(
            dataset=app["dataset"],
            modular_regressors=app.get("modular_regressors", []),
            quadratic_regressors=app.get("quadratic_regressors", []),
            quadratic_id_regressors=app.get("quadratic_id_regressors", []),
            item_modular=app.get("item_modular", "fe"),
        )
        meta.pop("raw", None)

        if app.get("winners_only"):
            sys.path.insert(0, str(ZERO_DIR))
            from data_utils import filter_winners, last_round_capacity
            input_data, keep = filter_winners(input_data)
            meta["n_obs"] = input_data["id_data"]["obs_bundles"].shape[0]
            if app.get("capacity_source") == "last_round":
                input_data["id_data"]["capacity"] = last_round_capacity(raw["bidder_data"], keep)
            print(f"  filtered to {meta['n_obs']} winners")

        config["dimensions"].update(
            n_obs=meta["n_obs"], n_items=meta["n_items"],
            n_covariates=meta["n_covariates"],
            covariate_names=meta["covariate_names"],
        )
        app.update(
            n_id_mod=meta["n_id_mod"], n_item_mod=meta["n_item_mod"],
            n_id_quad=meta["n_id_quad"], n_item_quad=meta["n_item_quad"],
        )
        print(f"zero_noise solver ({app['error_mode']}): {meta['n_obs']} obs, "
              f"{meta['n_items']} items, {meta['n_covariates']} cov")
        print(f"  {meta['covariate_names']}")
    else:
        input_data, meta, ctx = None, {}, None

    if comm is not None:
        config = comm.bcast(config, root=0)
        meta = comm.bcast(meta, root=0)
        ctx = comm.bcast(ctx, root=0)

    import combest as ce
    from combest.estimation.callbacks import adaptive_gurobi_timeout
    from knapsack import ZeroNoiseKnapsack

    model = ce.Model()
    model.load_config(config)
    model.data.load_and_distribute_input_data(input_data)
    model.features.build_quadratic_covariates_from_data()

    # zero out combest's error oracle
    cm = model.features.comm_manager
    n_items = model.n_items
    model.features.local_modular_errors = np.zeros((cm.num_local_agent, n_items))
    model.features._error_oracle = lambda b, ids: (model.features.local_modular_errors[ids] * b).sum(-1)
    model.features._error_oracle_takes_data = False

    # build fixed terms and custom solver
    fixed_linear, fixed_quadratic = _build_fixed_terms(model, ctx, app["error_mode"])

    solver = ZeroNoiseKnapsack(
        comm_manager=cm,
        data_manager=model.data,
        features_manager=model.features,
        dimensions_cfg=model.config.dimensions,
        gurobi_params=config.get("subproblem", {}).get("gurobi_params"),
        fixed_linear=fixed_linear,
        fixed_quadratic=fixed_quadratic,
    )
    solver.initialize()

    # replace combest's solver with ours
    model.subproblems.subproblem_solver = solver

    callbacks = config.get("callbacks", {})
    pt_cb, _ = adaptive_gurobi_timeout(callbacks["row_gen"])
    result = model.row_generation.solve(iteration_callback=pt_cb, verbose=True)

    if rank == 0 and result is not None:
        import time as _time
        theta = result.theta_hat
        names = config["dimensions"]["covariate_names"]

        print(f"\ntheta ({len(theta)} params):")
        for idx in sorted(names):
            print(f"  {names[idx]}: {theta[int(idx)]:.6f}")

        config_stem = Path(config_path).stem
        out = {
            "theta_hat": theta.tolist(),
            "covariate_names": {str(k): v for k, v in names.items()},
            "converged": bool(result.converged),
            "objective": float(result.final_objective),
            "iterations": int(result.num_iterations),
            "n_items": config["dimensions"]["n_items"],
            "n_obs": config["dimensions"]["n_obs"],
            "n_covariates": config["dimensions"]["n_covariates"],
            "n_id_mod": app.get("n_id_mod", 0),
            "n_id_quad": app.get("n_id_quad", 0),
            "n_btas": meta.get("n_btas", config["dimensions"]["n_items"]),
            "error_mode": app["error_mode"],
            "timestamp": _time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        if result.u_hat is not None:
            out["u_hat"] = result.u_hat.tolist()

        out_path = out_dir / f"result_{config_stem}.json"
        json.dump(out, open(out_path, "w"), indent=2)
        print(f"Saved -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to config YAML")
    main(parser.parse_args().config)
