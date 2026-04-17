#!/usr/bin/env python3
"""CF MTA prices vs A·(observed BTA prices).

Usage: python -m applications.combinatorial_auction.pipeline.counterfactual.analyze SPEC [--tag with_xi|no_xi]
"""
import sys, json, argparse
from pathlib import Path
import numpy as np

APP_ROOT  = Path(__file__).resolve().parent.parent.parent
REPO_ROOT = APP_ROOT.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from applications.combinatorial_auction.data.loaders import load_raw, load_aggregation_matrix


def run(spec, tag="no_xi", *, results_dir=None):
    res_dir = Path(results_dir) if results_dir else APP_ROOT / "results"
    cf = json.load(open(res_dir / spec / f"cf_{tag}.json"))
    prices_mta = np.asarray(cf["prices"])
    mta_nums   = cf["continental_mta_nums"]

    raw = load_raw()
    btas = raw["bta_data"]["bta"].astype(int).values
    A, _ = load_aggregation_matrix(btas)
    price_bta = raw["bta_data"]["bid"].to_numpy(dtype=float) / 1e9
    price_bta_agg = A @ price_bta

    print(f"CF[{spec}/{tag}]: {cf['n_mtas']} MTAs")
    print(f"  Revenue   CF={prices_mta.sum():.2f}B   observed A·p={price_bta_agg.sum():.2f}B   "
          f"ratio={prices_mta.sum()/price_bta_agg.sum():.3f}")
    print(f"  corr(CF, A·p) = {np.corrcoef(prices_mta, price_bta_agg)[0, 1]:.4f}")
    print(f"  zero-price MTAs: {(prices_mta < 1e-6).sum()}")
    print()
    print(f"  {'MTA':>4}  {'CF ($M)':>10}  {'Obs ($M)':>10}  {'Ratio':>8}")
    print(f"  {'-'*42}")
    for i, m in enumerate(mta_nums):
        ratio = prices_mta[i] / price_bta_agg[i] if price_bta_agg[i] > 0 else float("nan")
        print(f"  {m:>4}  {prices_mta[i]*1e3:>10.1f}  {price_bta_agg[i]*1e3:>10.1f}  {ratio:>8.2%}")


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore", category=RuntimeWarning)
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--tag", default="no_xi", choices=["with_xi", "no_xi"])
    args = ap.parse_args()
    run(args.spec, args.tag)
