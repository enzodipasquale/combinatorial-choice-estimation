"""Compare new build_error_oracle() against main's _build_error_oracle, per spec.
Builds a real combest model (single rank, no MPI) and diffs local_modular_errors."""
import sys, os, yaml, importlib, numpy as np
from pathlib import Path

WORKTREE = Path("/Users/enzo-macbookpro/MyProjects/combinatorial-choice-estimation/.claude/worktrees/zealous-golick")
MAIN     = Path("/Users/enzo-macbookpro/MyProjects/combinatorial-choice-estimation")

SPECS = [
    "boot", "boot_3", "boot_5",
    "boot_pop_scaling", "boot_pop_scaling_large",
    "boot_pop_scaling_winners",
    "sar_rho02",  # from archive
]


def _load_spec(name):
    cfg_dir = WORKTREE / "applications/combinatorial_auction/legacy/old_code/scripts_old/c_block/configs"
    p = cfg_dir / f"{name}.yaml"
    if not p.exists():
        p = cfg_dir / "archive" / f"{name}.yaml"
    return yaml.safe_load(open(p))


def _reset_imports():
    for m in list(sys.modules):
        if m.startswith("applications.combinatorial_auction") or m == "combest" or m.startswith("combest."):
            sys.modules.pop(m, None)


def _build_model(repo, config, input_data):
    sys.path = [s for s in sys.path if s not in (str(WORKTREE), str(MAIN))]
    sys.path.insert(0, str(repo))
    os.chdir(repo)
    _reset_imports()

    import combest as ce
    model = ce.Model()
    model.load_config(config)
    model.data.load_and_distribute_input_data(input_data)
    model.features.build_quadratic_covariates_from_data()
    return model


def _run_main(config):
    spec = config["application"]
    sys.path = [s for s in sys.path if s not in (str(WORKTREE), str(MAIN))]
    sys.path.insert(0, str(MAIN))
    os.chdir(MAIN)
    _reset_imports()

    from applications.combinatorial_auction.data.prepare import prepare
    from applications.combinatorial_auction.data.loaders import (
        load_bta_data, filter_winners, last_round_capacity,
    )
    input_data, meta = prepare(
        dataset=spec.get("dataset", "c_block"),
        modular_regressors=spec.get("modular_regressors", []),
        quadratic_regressors=spec.get("quadratic_regressors", []),
        quadratic_id_regressors=spec.get("quadratic_id_regressors", []),
        item_modular=spec.get("item_modular", "fe"),
        capacity_mode=spec.get("capacity_mode", "initial"),
    )
    meta.pop("raw", None)
    if spec.get("winners_only"):
        raw = load_bta_data()
        input_data, keep = filter_winners(input_data)
        meta["n_obs"] = input_data["id_data"]["obs_bundles"].shape[0]
        if spec.get("capacity_source") == "last_round":
            input_data["id_data"]["capacity"] = last_round_capacity(
                raw["bidder_data"], keep,
            )

    config["dimensions"].update(
        n_obs=meta["n_obs"], n_items=meta["n_items"],
        n_covariates=meta["n_covariates"],
        covariate_names=meta["covariate_names"],
    )
    spec.update(
        n_id_mod=meta["n_id_mod"], n_item_mod=meta["n_item_mod"],
        n_id_quad=meta["n_id_quad"], n_item_quad=meta["n_item_quad"],
    )

    model = _build_model(MAIN, config, input_data)

    # Main's estimate.py _build_error_oracle for c_block: Cholesky + scaling.
    if "sar_rho" in spec:
        from applications.combinatorial_auction.scripts.c_block.sar_robustness.sar_covariance import (
            build_sar_covariance,
        )
        from applications.combinatorial_auction.data.loaders import build_context
        raw = load_bta_data()
        ctx = build_context(raw)
        adj = ((ctx["bta_adjacency"] + ctx["bta_adjacency"].T) > 0).astype(float)
        np.fill_diagonal(adj, 0)
        sar_cov = build_sar_covariance(adj, rho=float(spec["sar_rho"]))
        model.features.build_local_modular_error_oracle(
            seed=spec.get("error_seed", 2006), covariance_matrix=sar_cov,
        )
    else:
        from applications.combinatorial_auction.data.loaders import build_cholesky_factor
        L = build_cholesky_factor(spec.get("error_correlation"))
        cov = L @ L.T if L is not None else None
        model.features.build_local_modular_error_oracle(
            seed=spec.get("error_seed", 2006), covariance_matrix=cov,
        )
        sc = spec.get("error_scaling")
        if sc == "elig":
            elig = model.data.local_data.id_data["elig"]
            model.features.local_modular_errors *= elig[:, None]
        elif sc == "pop":
            pop = load_bta_data()["bta_data"]["pop90"].to_numpy().astype(float)
            pop = pop / pop.sum()
            model.features.local_modular_errors *= pop[None, :]
    return model.features.local_modular_errors.copy()


def _run_new(config):
    spec = config["application"]
    sys.path = [s for s in sys.path if s not in (str(WORKTREE), str(MAIN))]
    sys.path.insert(0, str(WORKTREE))
    os.chdir(WORKTREE)
    _reset_imports()

    from applications.combinatorial_auction.data.prepare import prepare
    from applications.combinatorial_auction.data.loaders import load_raw, build_context
    from applications.combinatorial_auction.pipeline import errors as E

    input_data, meta = prepare(
        modular_regressors=spec.get("modular_regressors", []),
        quadratic_regressors=spec.get("quadratic_regressors", []),
        quadratic_id_regressors=spec.get("quadratic_id_regressors", []),
        item_modular=spec.get("item_modular", "fe"),
        winners_only=spec.get("winners_only", False),
        capacity_source=spec.get("capacity_source", "initial"),
    )
    ctx = build_context(load_raw())

    config["dimensions"].update(
        n_obs=meta["n_obs"], n_items=meta["n_items"],
        n_covariates=meta["n_covariates"],
        covariate_names=meta["covariate_names"],
    )
    spec.update(
        n_id_mod=meta["n_id_mod"], n_item_mod=meta["n_item_mod"],
        n_id_quad=meta["n_id_quad"], n_item_quad=meta["n_item_quad"],
    )

    model = _build_model(WORKTREE, config, input_data)
    # Translate legacy sar_rho -> spatial_rho field.
    if "sar_rho" in spec and "spatial_rho" not in spec:
        spec["spatial_rho"] = spec["sar_rho"]

    cov = E.covariance(ctx, spec)
    pop = E.pop_vector(ctx) if spec.get("error_scaling") == "pop" else None
    E.install(model, seed=spec["error_seed"], cov=cov,
              scaling=spec.get("error_scaling"), pop=pop)
    return model.features.local_modular_errors.copy()


def compare(name):
    cfg = _load_spec(name)
    if cfg["application"].get("dataset", "c_block") != "c_block":
        print(f"[{name}] SKIP non-c_block")
        return True
    # Need independent config copies (dict mutations between runs).
    import copy
    c1, c2 = copy.deepcopy(cfg), copy.deepcopy(cfg)
    main_err = _run_main(c1)
    new_err  = _run_new(c2)
    if main_err.shape != new_err.shape:
        print(f"[{name}] FAIL shape {main_err.shape} vs {new_err.shape}")
        return False
    d = np.abs(main_err - new_err)
    if d.max() > 1e-12:
        print(f"[{name}] FAIL max_abs_diff={d.max():.3e} n_mismatched={(d > 1e-12).sum()}")
        return False
    print(f"[{name}] OK errors shape={main_err.shape} std={new_err.std():.4e}")
    return True


if __name__ == "__main__":
    names = sys.argv[1:] or SPECS
    ok = all(compare(n) for n in names)
    sys.exit(0 if ok else 1)
