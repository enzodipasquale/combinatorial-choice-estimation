"""Compare new prepare() against main's, per spec.
Exits 0 if everything matches within tolerance, 1 otherwise."""
import sys, os, yaml, numpy as np, importlib
from pathlib import Path

WORKTREE = Path("/Users/enzo-macbookpro/MyProjects/combinatorial-choice-estimation/.claude/worktrees/zealous-golick")
MAIN     = Path("/Users/enzo-macbookpro/MyProjects/combinatorial-choice-estimation")

SPECS = [
    "boot", "boot_3", "boot_5",
    "boot_pop_scaling", "boot_pop_scaling_large",
    "boot_pop_scaling_winners",
    "boot_pop_scaling_winners_item_only",
    "boot_pop_scaling_winners_item_only_large",
]


def _load_spec(name):
    p = WORKTREE / "applications/combinatorial_auction/legacy/old_code/scripts_old/c_block/configs" / f"{name}.yaml"
    return yaml.safe_load(open(p))["application"]


def _run(repo, spec):
    """Import prepare from a given repo and run it on the spec's app dict."""
    # Remove only repo-root entries (not venv site-packages which may contain
    # the project path as a substring).
    sys.path = [s for s in sys.path if s not in (str(WORKTREE), str(MAIN))]
    sys.path.insert(0, str(repo))
    # Force-reload the modules from this repo.
    for m in list(sys.modules):
        if m.startswith("applications.combinatorial_auction"):
            sys.modules.pop(m, None)
    os.chdir(repo)

    if repo == MAIN:
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
        # Replicate estimate.py's post-prepare winners filter.
        if spec.get("winners_only"):
            raw = load_bta_data()
            input_data, keep = filter_winners(input_data)
            meta["n_obs"] = input_data["id_data"]["obs_bundles"].shape[0]
            if spec.get("capacity_source") == "last_round":
                input_data["id_data"]["capacity"] = last_round_capacity(
                    raw["bidder_data"], keep
                )
        return input_data, meta
    else:
        from applications.combinatorial_auction.data.prepare import prepare
        out = prepare(
            modular_regressors=spec.get("modular_regressors", []),
            quadratic_regressors=spec.get("quadratic_regressors", []),
            quadratic_id_regressors=spec.get("quadratic_id_regressors", []),
            item_modular=spec.get("item_modular", "fe"),
            winners_only=spec.get("winners_only", False),
            capacity_source=spec.get("capacity_source", "initial"),
        )
        return out


def _diff_arrays(a, b, path, tol=1e-12):
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        if a.shape != b.shape:
            return [f"{path}: shape {a.shape} vs {b.shape}"]
        if a.dtype != b.dtype:
            # dtype mismatch is OK if values equal
            pass
        if np.issubdtype(a.dtype, np.floating) or np.issubdtype(b.dtype, np.floating):
            if not np.allclose(a, b, atol=tol, rtol=0):
                d = np.abs(a.astype(float) - b.astype(float))
                return [f"{path}: max_abs_diff={d.max():.3e}, n_mismatched={(d > tol).sum()}"]
        else:
            if not np.array_equal(a, b):
                return [f"{path}: integer/bool arrays differ, n_mismatched={(a != b).sum()}"]
        return []
    if isinstance(a, dict):
        if set(a) != set(b):
            return [f"{path}: key set differs: {set(a) ^ set(b)}"]
        out = []
        for k in a:
            out += _diff_arrays(a[k], b[k], f"{path}.{k}")
        return out
    if a != b:
        return [f"{path}: {a!r} vs {b!r}"]
    return []


def compare(spec_name):
    spec = _load_spec(spec_name)
    # Skip joint datasets (not supported in worktree).
    if spec.get("dataset", "c_block") != "c_block":
        print(f"[{spec_name}] SKIP non-c_block dataset={spec.get('dataset')}")
        return True

    new_data, new_meta = _run(WORKTREE, spec)
    main_data, main_meta = _run(MAIN, spec)
    main_meta.pop("raw", None)  # raw is dropped by estimate.py before broadcast

    errors = []
    errors += _diff_arrays(new_data, main_data, "input_data")
    # compare metas key by key; A/continental_mta_nums are not in new meta.
    for k in sorted(set(new_meta) | set(main_meta)):
        if k in {"A", "continental_mta_nums"}:
            continue
        if k not in new_meta:
            errors.append(f"meta.{k}: missing in new (main={type(main_meta[k]).__name__})")
            continue
        if k not in main_meta:
            errors.append(f"meta.{k}: missing in main (new={type(new_meta[k]).__name__})")
            continue
        errors += _diff_arrays(new_meta[k], main_meta[k], f"meta.{k}")

    if errors:
        print(f"[{spec_name}] FAIL ({len(errors)} diffs)")
        for e in errors[:20]:
            print(f"   {e}")
        return False
    print(f"[{spec_name}] OK  n_obs={new_meta['n_obs']}  "
          f"n_items={new_meta['n_items']}  n_cov={new_meta['n_covariates']}")
    return True


if __name__ == "__main__":
    names = sys.argv[1:] or SPECS
    ok = all(compare(n) for n in names)
    sys.exit(0 if ok else 1)
