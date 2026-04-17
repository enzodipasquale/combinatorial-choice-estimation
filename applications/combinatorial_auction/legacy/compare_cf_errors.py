"""Compare new install_aggregated against main's _build_counterfactual_errors."""
import sys, os, yaml, json, copy
import numpy as np
from pathlib import Path

WORKTREE = Path("/Users/enzo-macbookpro/MyProjects/combinatorial-choice-estimation/.claude/worktrees/zealous-golick")
MAIN     = Path("/Users/enzo-macbookpro/MyProjects/combinatorial-choice-estimation")
MAIN_CFG = MAIN / "applications/combinatorial_auction/scripts/c_block/configs"
MAIN_RES = MAIN / "applications/combinatorial_auction/scripts/c_block/results"


def _fresh(path):
    sys.path = [s for s in sys.path if s not in (str(WORKTREE), str(MAIN))]
    sys.path.insert(0, str(path))
    os.chdir(path)
    for m in list(sys.modules):
        if m.startswith("applications.combinatorial_auction"):
            sys.modules.pop(m, None)


def _find_cfg(stem):
    for d in (MAIN_CFG, MAIN_CFG / "archive"):
        p = d / f"{stem}.yaml"
        if p.exists():
            return p
    raise FileNotFoundError(stem)


def _iv(spec):
    _fresh(WORKTREE)
    from applications.combinatorial_auction.pipeline.second_stage.iv import (
        simple_instruments, second_stage as run_ss,
    )
    from applications.combinatorial_auction.data.loaders import load_raw
    app = yaml.safe_load(open(_find_cfg(spec)))["application"]
    r = json.load(open(MAIN_RES / spec / "result.json"))
    theta = np.asarray(r["theta_hat"])
    n_id_mod = r["n_id_mod"]
    n_btas = r["n_btas"]
    delta = -theta[n_id_mod:n_id_mod + n_btas]
    raw = load_raw()
    price = raw["bta_data"]["bid"].to_numpy(dtype=float) / 1e9
    use_blp = app.get("error_scaling") == "pop"
    si = None if use_blp else simple_instruments(raw)
    iv = run_ss(delta, price, raw, use_blp=use_blp, simple_instruments_cached=si)
    return theta, app, iv["a0"], iv["a1"], iv["demand_controls"]


def _build_min_model(repo, config, input_data):
    _fresh(repo)
    import combest as ce
    model = ce.Model()
    model.load_config(config)
    model.data.load_and_distribute_input_data(input_data)
    model.features.build_quadratic_covariates_from_data()
    return model


def _run_new(spec, include_xi):
    theta, app, a0, a1, dc = _iv(spec)
    _fresh(WORKTREE)
    from applications.combinatorial_auction.pipeline.counterfactual.prepare import (
        prepare_counterfactual, freeze_bounds,
    )
    from applications.combinatorial_auction.pipeline import errors as E
    from applications.combinatorial_auction.data.loaders import load_raw, build_context

    input_data, meta, cf = prepare_counterfactual(theta, app, alpha_0=a0, alpha_1=a1, demand_controls=dc)
    cfg = yaml.safe_load(open(_find_cfg(spec)))
    freeze_bounds(cfg, meta, cf)
    cfg["dimensions"].update(
        n_obs=meta["n_obs"], n_items=meta["n_items"],
        n_covariates=meta["n_covariates"],
        covariate_names=meta["covariate_names"],
    )
    cfg["application"].update(
        n_id_mod=meta["n_id_mod"], n_item_mod=meta["n_item_mod"],
        n_id_quad=meta["n_id_quad"], n_item_quad=meta["n_item_quad"],
    )
    model = _build_min_model(WORKTREE, cfg, input_data)
    bta_cov = E.covariance(build_context(load_raw()), app)
    from applications.combinatorial_auction.pipeline.counterfactual.run import CF_ERROR_SEED
    E.install_aggregated(
        model, seed=CF_ERROR_SEED, A=cf["A"],
        bta_cov=bta_cov,
        offset=cf["offset_m"] if include_xi else cf["offset_m_no_xi"],
        scaling=app.get("error_scaling"), pop=cf["pop"], elig=cf["elig"],
    )
    return model.features.local_modular_errors.copy()


def _run_main(spec, include_xi):
    theta, app, a0, a1, dc = _iv(spec)
    _fresh(MAIN)
    from applications.combinatorial_auction.scripts.c_block.counterfactual.prepare import (
        prepare_counterfactual,
    )
    from applications.combinatorial_auction.scripts.c_block.counterfactual.run import (
        _build_errors as _build_counterfactual_errors, ERROR_SEED as CF_ERROR_SEED,
    )
    r = json.load(open(MAIN_RES / spec / "result.json"))
    input_data, meta = prepare_counterfactual(
        est_result_path_or_dict=r, alpha_0=a0, alpha_1=a1,
        modular_regressors=app.get("modular_regressors", []),
        quadratic_regressors=app.get("quadratic_regressors", []),
        quadratic_id_regressors=app.get("quadratic_id_regressors", []),
        demand_controls=dc,
    )
    cfg = yaml.safe_load(open(_find_cfg(spec)))
    cfg["dimensions"].update(
        n_obs=meta["n_obs"], n_items=meta["n_items"],
        n_covariates=meta["n_covariates"],
        covariate_names=meta["covariate_names"],
    )
    cfg["application"].update(
        n_id_mod=meta["n_id_mod"], n_item_mod=meta["n_item_mod"],
        n_id_quad=meta["n_id_quad"], n_item_quad=meta["n_item_quad"],
    )
    model = _build_min_model(MAIN, cfg, input_data)
    _build_counterfactual_errors(
        model, meta, CF_ERROR_SEED,
        app.get("error_scaling"),
        app.get("error_correlation"),
        include_xi,
    )
    return model.features.local_modular_errors.copy()


SPECS = [("boot", True), ("boot", False),
         ("boot_pop_scaling", True), ("boot_pop_scaling", False)]
ok = True
for spec, with_xi in SPECS:
    new = _run_new(spec, with_xi)
    old = _run_main(spec, with_xi)
    d = np.abs(new - old)
    label = "with_xi" if with_xi else "no_xi"
    if d.max() > 1e-10:
        print(f"[{spec}/{label}] FAIL max_abs={d.max():.3e}  n_off={(d > 1e-10).sum()}")
        ok = False
    else:
        print(f"[{spec}/{label}] OK shape={new.shape}")
sys.exit(0 if ok else 1)
