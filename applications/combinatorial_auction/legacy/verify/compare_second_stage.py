"""Compare new second_stage.decompose() against main's run_spec, per spec.
Uses main's existing bootstrap_result.json files so no new bootstrap is needed."""
import sys, os, importlib, copy
import numpy as np
from pathlib import Path

WORKTREE = Path("/Users/enzo-macbookpro/MyProjects/combinatorial-choice-estimation/.claude/worktrees/zealous-golick")
MAIN     = Path("/Users/enzo-macbookpro/MyProjects/combinatorial-choice-estimation")
MAIN_CONFIGS = MAIN / "applications/combinatorial_auction/scripts/c_block/configs"
MAIN_RESULTS = MAIN / "applications/combinatorial_auction/scripts/c_block/results"

SPECS_DEFAULT = [
    "boot", "boot_3", "boot_5", "boot_pop_scaling_winners",
    "sar_rho00", "sar_rho02", "sar_rho04", "sar_rho06",
]


def _fresh(path):
    sys.path = [s for s in sys.path if s not in (str(WORKTREE), str(MAIN))]
    sys.path.insert(0, str(path))
    os.chdir(path)
    for m in list(sys.modules):
        if m.startswith("applications.combinatorial_auction"):
            sys.modules.pop(m, None)


def _run_main(stem):
    _fresh(MAIN)
    from applications.combinatorial_auction.scripts.c_block.second_stage import compute
    # Redirect to archive if the stem's config is there.
    cfg_dir = _find_config(stem)
    compute.CONFIGS_DIR = cfg_dir
    data = compute._load_data()
    rows, names = compute.run_spec(stem, *data)
    return rows, names


def _find_config(stem):
    for d in (MAIN_CONFIGS, MAIN_CONFIGS / "archive"):
        p = d / f"{stem}.yaml"
        if p.exists():
            return d
    raise FileNotFoundError(f"{stem}.yaml not in {MAIN_CONFIGS} or archive/")


def _run_new(stem):
    _fresh(WORKTREE)
    from applications.combinatorial_auction.pipeline.second_stage.compute import decompose
    rows, names = decompose(stem, configs_dir=_find_config(stem), results_dir=MAIN_RESULTS)
    return rows, names


def _diff(a, b, path, tol=1e-8):
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return [f"{path}: keys differ: {set(a) ^ set(b)}"]
        out = []
        for k in a:
            out += _diff(a[k], b[k], f"{path}.{k}")
        return out
    try:
        fa, fb = float(a), float(b)
    except (TypeError, ValueError):
        if a != b:
            return [f"{path}: {a!r} vs {b!r}"]
        return []
    if np.isnan(fa) and np.isnan(fb):
        return []
    if abs(fa - fb) > tol and abs(fa - fb) / max(abs(fa), abs(fb), 1e-12) > tol:
        return [f"{path}: {fa:.6g} vs {fb:.6g}  (Δ={fa-fb:.3e})"]
    return []


def compare(stem):
    if not (MAIN_RESULTS / stem / "bootstrap_result.json").exists():
        print(f"[{stem}] SKIP no bootstrap_result.json")
        return True
    new_rows, new_names = _run_new(stem)
    main_rows, main_names = _run_main(stem)
    if new_names != main_names:
        print(f"[{stem}] FAIL names {new_names} vs {main_names}")
        return False
    if len(new_rows) != len(main_rows):
        print(f"[{stem}] FAIL n_rows {len(new_rows)} vs {len(main_rows)}")
        return False

    errors = []
    for i, (a, b) in enumerate(zip(new_rows, main_rows)):
        errors += _diff(a, b, f"row[{i}]")
    if errors:
        print(f"[{stem}] FAIL ({len(errors)} diffs)")
        for e in errors[:15]:
            print(f"   {e}")
        return False
    print(f"[{stem}] OK  n_draws={len(new_rows)}  n_cov={len(new_names)}")
    return True


if __name__ == "__main__":
    names = sys.argv[1:] or SPECS_DEFAULT
    ok = all(compare(n) for n in names)
    sys.exit(0 if ok else 1)
