"""Compare new prepare_counterfactual against main's, using main's result.json."""
import sys, os, yaml, json, copy
import numpy as np
from pathlib import Path

WORKTREE = Path("/Users/enzo-macbookpro/MyProjects/combinatorial-choice-estimation/.claude/worktrees/zealous-golick")
MAIN     = Path("/Users/enzo-macbookpro/MyProjects/combinatorial-choice-estimation")
MAIN_CONFIGS = MAIN / "applications/combinatorial_auction/scripts/c_block/configs"
MAIN_RESULTS = MAIN / "applications/combinatorial_auction/scripts/c_block/results"

SPECS_DEFAULT = ["boot", "boot_pop_scaling"]


def _fresh(path):
    sys.path = [s for s in sys.path if s not in (str(WORKTREE), str(MAIN))]
    sys.path.insert(0, str(path))
    os.chdir(path)
    for m in list(sys.modules):
        if m.startswith("applications.combinatorial_auction"):
            sys.modules.pop(m, None)


def _find_cfg(stem):
    for d in (MAIN_CONFIGS, MAIN_CONFIGS / "archive"):
        p = d / f"{stem}.yaml"
        if p.exists():
            return p
    raise FileNotFoundError(stem)


def _iv_on_pt(stem):
    """Run second_stage 2SLS on the point estimate so both main and new see
    identical (alpha_0, alpha_1, demand_controls)."""
    _fresh(WORKTREE)
    from applications.combinatorial_auction.pipeline.second_stage.iv import (
        simple_instruments, second_stage as run_ss, compute_xi,
    )
    from applications.combinatorial_auction.data.loaders import load_raw
    cfg = yaml.safe_load(open(_find_cfg(stem)))
    app = cfg["application"]
    r = json.load(open(MAIN_RESULTS / stem / "result.json"))
    theta = np.array(r["theta_hat"])
    n_id_mod = r["n_id_mod"]
    n_btas   = r["n_btas"]
    delta = -theta[n_id_mod:n_id_mod + n_btas]
    raw = load_raw()
    price = raw["bta_data"]["bid"].to_numpy(dtype=float) / 1e9
    use_blp = app.get("error_scaling") == "pop"
    si = None if use_blp else simple_instruments(raw)
    iv = run_ss(delta, price, raw, use_blp=use_blp, simple_instruments_cached=si)
    return theta, app, iv["a0"], iv["a1"], iv["demand_controls"]


def _main_prep(stem, theta, app, a0, a1, dc):
    _fresh(MAIN)
    from applications.combinatorial_auction.scripts.c_block.counterfactual.prepare import (
        prepare_counterfactual,
    )
    r = json.load(open(MAIN_RESULTS / stem / "result.json"))
    return prepare_counterfactual(
        est_result_path_or_dict=r,
        alpha_0=a0, alpha_1=a1,
        modular_regressors      = app.get("modular_regressors", []),
        quadratic_regressors    = app.get("quadratic_regressors", []),
        quadratic_id_regressors = app.get("quadratic_id_regressors", []),
        demand_controls=dc,
    )


def _new_prep(theta, app, a0, a1, dc):
    _fresh(WORKTREE)
    from applications.combinatorial_auction.pipeline.counterfactual.prepare import (
        prepare_counterfactual,
    )
    return prepare_counterfactual(theta, app, alpha_0=a0, alpha_1=a1, demand_controls=dc)


def _diff_np(a, b, path, tol=1e-10):
    if a.shape != b.shape:
        return [f"{path}: shape {a.shape} vs {b.shape}"]
    if a.dtype.kind in "fc" or b.dtype.kind in "fc":
        d = np.abs(a.astype(float) - b.astype(float))
        if d.max() > tol:
            return [f"{path}: max_abs_diff={d.max():.3e} n_mismatched={(d > tol).sum()}"]
    else:
        if not np.array_equal(a, b):
            return [f"{path}: int/bool arrays differ n_mismatched={(a != b).sum()}"]
    return []


def _diff_tree(a, b, path):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return _diff_np(np.asarray(a), np.asarray(b), path)
    if isinstance(a, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            if k not in a:
                out.append(f"{path}.{k}: missing in new")
            elif k not in b:
                out.append(f"{path}.{k}: missing in main")
            else:
                out += _diff_tree(a[k], b[k], f"{path}.{k}")
        return out
    if a != b:
        return [f"{path}: {a!r} vs {b!r}"]
    return []


def compare(stem):
    theta, app, a0, a1, dc = _iv_on_pt(stem)
    new_input, new_meta, new_cf = _new_prep(theta, app, a0, a1, dc)
    main_input, main_meta = _main_prep(stem, theta, app, a0, a1, dc)

    errors = []
    # input_data arrays
    errors += _diff_tree(new_input, main_input, "input_data")
    # cf-side: offset_m, offset_m_no_xi, beta, gamma_id, gamma_item
    for k in ("offset_m", "offset_m_no_xi", "beta", "gamma_id", "gamma_item"):
        if k not in main_meta:
            errors.append(f"cf.{k}: missing in main meta")
            continue
        errors += _diff_np(np.asarray(new_cf[k]), np.asarray(main_meta[k]), f"cf.{k}")
    # A (aggregation matrix)
    errors += _diff_np(new_cf["A"], main_meta["A"], "cf.A")

    if errors:
        print(f"[{stem}] FAIL ({len(errors)} diffs)")
        for e in errors[:15]:
            print(f"   {e}")
        return False
    print(f"[{stem}] OK  n_mta={new_meta['n_mtas']}  n_cov={new_meta['n_covariates']}")
    return True


if __name__ == "__main__":
    names = sys.argv[1:] or SPECS_DEFAULT
    ok = all(compare(n) for n in names)
    sys.exit(0 if ok else 1)
