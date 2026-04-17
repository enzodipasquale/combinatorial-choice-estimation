#!/usr/bin/env python3
"""Loop wrapper: run `estimate` on every spatial_correlated_rho*.yaml config.

The SAR model is handled entirely by pipeline.errors (via the spatial_rho key)
and pipeline.estimate — this module only iterates over the rho grid for
convenience. For a single rho, invoke estimate.py directly.

Usage:
    mpirun -n N python -m applications.combinatorial_auction.pipeline.spatial_correlated.run_all
    mpirun -n N python -m applications.combinatorial_auction.pipeline.spatial_correlated.run_all 0.0 0.4
"""
import sys
from pathlib import Path

from applications.combinatorial_auction.pipeline.estimate import main as estimate_main

APP_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_RHOS = ["00", "02", "04", "06"]


def main(rhos=None):
    rhos = rhos or DEFAULT_RHOS
    failures = []
    for rho in rhos:
        cfg = APP_ROOT / "configs" / f"spatial_correlated_rho{rho}.yaml"
        if not cfg.exists():
            print(f"[rho={rho}] config not found at {cfg}")
            failures.append(rho)
            continue
        print(f"\n═══ spatial_correlated_rho{rho} ═══")
        try:
            estimate_main(str(cfg))
        except Exception as e:
            print(f"[rho={rho}] FAILED: {e}")
            failures.append(rho)
    if failures:
        print(f"\nFailed rhos: {failures}")
        sys.exit(1)


if __name__ == "__main__":
    args = [a.lstrip("0.").rjust(2, "0") if "." in a else a.rjust(2, "0")
            for a in sys.argv[1:]]
    main(args or None)
