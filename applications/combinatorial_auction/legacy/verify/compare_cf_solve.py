"""End-to-end CF comparison: run main's CF and new CF on same inputs, diff."""
import sys, os, json, tempfile, subprocess
import numpy as np
from pathlib import Path

WORKTREE = Path("/Users/enzo-macbookpro/MyProjects/combinatorial-choice-estimation/.claude/worktrees/zealous-golick")
MAIN     = Path("/Users/enzo-macbookpro/MyProjects/combinatorial-choice-estimation")
MAIN_CONFIGS = MAIN / "applications/combinatorial_auction/scripts/c_block/configs"
MAIN_RESULTS = MAIN / "applications/combinatorial_auction/scripts/c_block/results"

SPEC = sys.argv[1] if len(sys.argv) > 1 else "boot"
VARIANT = sys.argv[2] if len(sys.argv) > 2 else "no_xi"

# Run new CF on main's config+result, write to a temp dir
out_dir = Path(tempfile.mkdtemp(prefix="cf_new_"))
print(f"new CF → {out_dir}")

# fresh-import new code
sys.path = [s for s in sys.path if s not in (str(WORKTREE), str(MAIN))]
sys.path.insert(0, str(WORKTREE))
os.chdir(WORKTREE)
for m in list(sys.modules):
    if m.startswith("applications.combinatorial_auction"):
        sys.modules.pop(m, None)

from applications.combinatorial_auction.pipeline.counterfactual import run as cfrun
cfrun.main(SPEC, configs_dir=MAIN_CONFIGS, results_dir=MAIN_RESULTS, out_dir=out_dir)

# compare
new = json.load(open(out_dir / f"cf_{VARIANT}.json"))
main = json.load(open(MAIN_RESULTS / SPEC / f"cf_{VARIANT}.json"))

errors = []
for k in sorted(set(new) | set(main)):
    if k not in new or k not in main:
        errors.append(f"{k}: only in one side")
        continue
    a, b = new[k], main[k]
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            errors.append(f"{k}: len {len(a)} vs {len(b)}")
            continue
        aa, bb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        d = np.abs(aa - bb)
        if d.max() > 1e-6:
            errors.append(f"{k}: max_abs_diff={d.max():.3e}  n_off={(d > 1e-6).sum()}")
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if abs(a - b) > 1e-6 * max(1, abs(a), abs(b)):
            errors.append(f"{k}: {a} vs {b}")
    else:
        if a != b:
            errors.append(f"{k}: {a!r} vs {b!r}")

if errors:
    print(f"[{SPEC}/{VARIANT}] FAIL ({len(errors)} diffs)")
    for e in errors[:15]:
        print(f"   {e}")
    sys.exit(1)
print(f"[{SPEC}/{VARIANT}] OK")
