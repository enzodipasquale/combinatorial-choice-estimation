#!/usr/bin/env python3
"""Point estimation with deterministic errors (no noise, n_sim=1)."""
import json, sys, yaml, argparse
import numpy as np
from pathlib import Path

ZERO_DIR = Path(__file__).parent
CBLOCK_DIR = ZERO_DIR.parent
APP_DIR = CBLOCK_DIR.parent.parent
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
        print(f"zero_noise ({app['error_mode']}): {meta['n_obs']} obs, "
              f"{meta['n_items']} items, {meta['n_covariates']} cov")
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

    # deterministic errors: elig_i * pop_j with coefficient 1
    cm = model.features.comm_manager
    n_items = model.n_items
    local_errors = np.zeros((cm.num_local_agent, n_items))

    error_mode = app["error_mode"]
    if error_mode == "elig_pop":
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
    pt_cb, _ = adaptive_gurobi_timeout(callbacks["row_gen"])
    result = model.row_generation.solve(iteration_callback=pt_cb, verbose=True)

    if rank == 0 and result is not None:
        import time as _time
        theta = result.theta_hat
        names = config["dimensions"]["covariate_names"]

        print(f"\ntheta ({len(theta)} params):")
        for idx in sorted(names):
            print(f"  {names[idx]}: {theta[int(idx)]:.4f}")

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
            "error_mode": error_mode,
            "timestamp": _time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        if result.u_hat is not None:
            out["u_hat"] = result.u_hat.tolist()

        json.dump(out, open(out_dir / "result.json", "w"), indent=2)
        print(f"Saved -> {out_dir / 'result.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default=str(ZERO_DIR / "config.yaml"))
    main(parser.parse_args().config)
