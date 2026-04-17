#!/usr/bin/env python3
"""SAR-error bootstrap estimation runner."""
import gc, json, sys, yaml, argparse
import numpy as np
from pathlib import Path

SAR_DIR    = Path(__file__).parent
CBLOCK_DIR = SAR_DIR.parent
REPO_ROOT  = CBLOCK_DIR.parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

try:
    from mpi4py import MPI
    comm, rank = MPI.COMM_WORLD, MPI.COMM_WORLD.Get_rank()
except ImportError:
    comm, rank = None, 0

from applications.combinatorial_auction.data.prepare import prepare


def _results_dir(config_path):
    d = CBLOCK_DIR / "results" / Path(config_path).stem
    d.mkdir(parents=True, exist_ok=True)
    return d


def main(config_path):
    config = yaml.safe_load(open(config_path))
    out_dir = _results_dir(config_path)
    app    = config["application"]
    rho    = float(app["sar_rho"])

    if rank == 0:
        import warnings; warnings.filterwarnings("ignore", category=RuntimeWarning)

        from applications.combinatorial_auction.data.loaders import load_bta_data, build_context
        from applications.combinatorial_auction.scripts.c_block.sar_robustness.sar_covariance import (
            build_sar_covariance,
        )

        raw = load_bta_data()
        ctx = build_context(raw)
        adj = ctx["bta_adjacency"]
        adj = ((adj + adj.T) > 0).astype(float)
        np.fill_diagonal(adj, 0)

        sar_cov = build_sar_covariance(adj, rho=rho)
        n_btas  = adj.shape[0]

        print(f"SAR covariance built: rho={rho}, shape={sar_cov.shape}")
        off = sar_cov[np.triu_indices(n_btas, k=1)]
        print(f"  max off-diagonal: {off.max():.6f}, mean: {off.mean():.6f}")

        input_data, meta = prepare(
            dataset=app["dataset"],
            modular_regressors=app.get("modular_regressors", ["elig_pop"]),
            quadratic_regressors=app.get("quadratic_regressors", []),
            quadratic_id_regressors=app.get("quadratic_id_regressors", []),
            item_modular=app.get("item_modular", "fe"),
        )
        meta.pop("raw", None)

        if sar_cov.shape[0] != meta["n_items"]:
            raise ValueError(f"SAR cov shape {sar_cov.shape} != n_items={meta['n_items']}")

        config["dimensions"].update(
            n_obs=meta["n_obs"], n_items=meta["n_items"],
            n_covariates=meta["n_covariates"],
            covariate_names=meta["covariate_names"],
        )
        app.update(
            n_id_mod=meta["n_id_mod"], n_item_mod=meta["n_item_mod"],
            n_id_quad=meta["n_id_quad"], n_item_quad=meta["n_item_quad"],
        )
        print(f"{app['dataset']}: {meta['n_obs']} obs, {meta['n_items']} items, {meta['n_covariates']} cov")
    else:
        input_data, meta, sar_cov = None, {}, None

    if comm is not None:
        config  = comm.bcast(config, root=0)
        meta    = comm.bcast(meta,   root=0)
        sar_cov = comm.bcast(sar_cov, root=0)

    import combest as ce
    from combest.estimation.callbacks import adaptive_gurobi_timeout

    model = ce.Model()
    model.load_config(config)
    model.data.load_and_distribute_input_data(input_data)
    model.features.build_quadratic_covariates_from_data()

    model.features.build_local_modular_error_oracle(
        seed=app.get("error_seed", 2006),
        covariance_matrix=sar_cov,
    )

    model.subproblems.load_solver()

    callbacks = config.get("callbacks", {})
    boot_cfg  = config.get("bootstrap", {})

    se_overrides = {}
    for key in ("rowgen_max_iters", "rowgen_tol", "rowgen_min_iters"):
        if key in boot_cfg:
            se_overrides[key] = boot_cfg[key]
    if se_overrides:
        se_dict = model.config.standard_errors.__dict__.copy()
        se_dict.update(se_overrides)
        model.config.standard_errors = type(model.config.standard_errors)(**se_dict)

    pt_cb,   _       = adaptive_gurobi_timeout(callbacks["row_gen"])
    _,       dist_cb = adaptive_gurobi_timeout(callbacks["boot"])

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
            "theta_hat":        se.mean.tolist(),
            "se":               se.se.tolist(),
            "bootstrap_thetas": se.samples.tolist(),
            "bootstrap_u_hat":  se.u_samples.tolist(),
            "converged":        se.converged.tolist(),
            "config":           config,
        }
        json.dump(out, open(out_dir / "bootstrap_result.json", "w"), indent=2)
        print(f"Saved -> {out_dir / 'bootstrap_result.json'}")

    solver = getattr(model.subproblems, "subproblem_solver", None)
    if solver is not None:
        for grb_model in getattr(solver, "local_problems", []):
            try:
                grb_model.dispose()
            except Exception:
                pass

    del model
    del input_data
    del sar_cov
    if "se" in dir():
        del se
    gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to SAR config YAML")
    main(parser.parse_args().config)
