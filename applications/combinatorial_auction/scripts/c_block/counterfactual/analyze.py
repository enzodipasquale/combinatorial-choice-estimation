#!/usr/bin/env python3
"""Analyze counterfactual results: C-block bidders on MTAs."""
import json, sys
import numpy as np
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent.parent.parent
APP_DIR = SCRIPTS_DIR.parent
sys.path.insert(0, str(APP_DIR.parent.parent))

from applications.combinatorial_auction.data.loaders import load_bta_data, load_aggregation_matrix

CF_DIR = Path(__file__).parent


def run(result_file="result_counterfactual.json"):
    result = json.load(open(CF_DIR / result_file))
    prices_mta = np.array(result["prices"])
    n_mtas = result["n_mtas"]
    mta_nums = result["continental_mta_nums"]

    # observed BTA prices
    raw = load_bta_data()
    btas = raw["bta_data"]["bta"].values.astype(int)
    A = load_aggregation_matrix(btas)
    price_bta = raw["bta_data"]["bid"].to_numpy().astype(float) / 1e9
    price_bta_agg = A @ price_bta  # sum of BTA prices within each MTA

    print(f"{'='*60}")
    print(f"Counterfactual: C-block bidders on MTAs (N = {n_mtas})")
    print(f"{'='*60}")

    print(f"\n  {'Metric':<30} {'Counterfactual':>15} {'Observed (A@p)':>15}")
    print(f"  {'-'*60}")
    print(f"  {'Total revenue ($B)':<30} {prices_mta.sum():>15.4f} {price_bta_agg.sum():>15.4f}")
    print(f"  {'Mean price ($B)':<30} {prices_mta.mean():>15.6f} {price_bta_agg.mean():>15.6f}")
    print(f"  {'Median price ($B)':<30} {np.median(prices_mta):>15.6f} {np.median(price_bta_agg):>15.6f}")
    print(f"  {'Min price ($B)':<30} {prices_mta.min():>15.6f} {price_bta_agg.min():>15.6f}")
    print(f"  {'Max price ($B)':<30} {prices_mta.max():>15.6f} {price_bta_agg.max():>15.6f}")

    # per-MTA comparison
    print(f"\n  {'MTA':<8} {'CF price':>12} {'Obs A@p':>12} {'Ratio':>10}")
    print(f"  {'-'*42}")
    for i, m in enumerate(mta_nums):
        ratio = prices_mta[i] / price_bta_agg[i] if price_bta_agg[i] > 0 else np.nan
        print(f"  {m:<8} {prices_mta[i]:>12.6f} {price_bta_agg[i]:>12.6f} {ratio:>10.3f}")

    corr = np.corrcoef(prices_mta, price_bta_agg)[0, 1]
    print(f"\n  Correlation(CF, Obs): {corr:.4f}")
    print(f"  Revenue ratio (CF/Obs): {prices_mta.sum() / price_bta_agg.sum():.4f}")


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore", category=RuntimeWarning)
    run()
