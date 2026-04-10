#!/usr/bin/env python3
"""
SAR-error bootstrap estimation runner.

Usage:
    mpirun -n 4 python applications/combinatorial_auction/specs/c_block/sar_robustness/run.py \
        applications/combinatorial_auction/specs/c_block/sar_robustness/configs/config_sar_rho04.yaml
"""
import json, sys, yaml, argparse
import numpy as np
from pathlib import Path

SAR_DIR   = Path(__file__).parent
# sar_robustness -> c_block -> specs -> combinatorial_auction -> applications -> repo root
REPO_ROOT = SAR_DIR.parent.parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

try:
    from mpi4py import MPI
    comm, rank = MPI.COMM_WORLD, MPI.COMM_WORLD.Get_rank()
except ImportError:
    comm, rank = None, 0

from applications.combinatorial_auction.data.prepare import prepare


def main(config_path):
    config = yaml.safe_load(open(config_path))
    app    = config["application"]
    rho    = float(app["sar_rho"])
    config_name = Path(config_path).stem

    # ── Ensure results/ directory exists ──────────────────────────
    if rank == 0:
        (SAR_DIR / "results").mkdir(parents=True, exist_ok=True)

    # ── Build SAR covariance (rank 0 only, then broadcast) ────────
    if rank == 0:
        import warnings; warnings.filterwarnings("ignore", category=RuntimeWarning)

        from applications.combinatorial_auction.data.loaders import load_bta_data, build_context
        from applications.combinatorial_auction.specs.c_block.sar_robustness.sar_covariance import (
            build_sar_covariance,
        )

        raw = load_bta_data()
        ctx = build_context(raw)
        adj = ctx["bta_adjacency"]
        # symmetrize and binarize
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

        # Verify dimensions match
        if sar_cov.shape[0] != meta["n_items"]:
            raise ValueError(
                f"SAR covariance shape {sar_cov.shape} != n_items={meta['n_items']}. "
                "Adjacency matrix and item count mismatch."
            )

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

    # Inject SAR covariance — bypasses the existing error_correlation path entirely.
    # Always pass sar_cov (including at rho=0 where it is the identity).
    # Criterion 4 requires verifying the Cholesky path is a no-op at rho=0,
    # not that the iid path matches itself.
    model.features.build_local_modular_error_oracle(
        seed=app.get("error_seed", 2006),
        covariance_matrix=sar_cov,
    )

    model.subproblems.load_solver()

    callbacks = config.get("callbacks", {})
    boot_cfg  = config.get("bootstrap", {})

    # ── Bootstrap row-gen overrides ────────────────────────────────
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
        save_model_dir=str(SAR_DIR / f"results/master_{config_name}"),
        load_model_dir=str(SAR_DIR / f"results/master_{config_name}"),
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
        out_path = SAR_DIR / f"results/bootstrap_result_{config_name}.json"
        json.dump(out, open(out_path, "w"), indent=2)
        if rank == 0:
            print(f"Saved -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to SAR config YAML")
    main(parser.parse_args().config)
