#!/usr/bin/env python3
"""Eligibility evolution during the C-block auction."""
import numpy as np
import pandas as pd
from pathlib import Path

from applications.combinatorial_auction.data.analysis.helpers import RAW_DIR


def load():
    elig = pd.read_csv(RAW_DIR / "cblock-eligibility.csv")
    bidders = pd.read_csv(RAW_DIR / "biddercblk_03_28_2004_pln.csv")
    bidders = bidders[bidders["bidder_num"] != 9999]
    return elig, bidders


def bidder_summary(elig, bidders):
    """Per-bidder summary: initial/final eligibility, decay, activity."""
    records = []
    for bn in bidders["bidder_num_fox"]:
        sub = elig[elig["bidder_num_fox"] == bn].sort_values("round_num")
        active = sub[sub["max_elig"] > 0]
        if len(active) == 0:
            continue
        records.append({
            "bidder": bn,
            "co_name": bidders.loc[bidders["bidder_num_fox"] == bn, "co_name"].iloc[0],
            "pops_eligible": bidders.loc[bidders["bidder_num_fox"] == bn, "pops_eligible"].iloc[0],
            "initial_elig": active.iloc[0]["max_elig"],
            "final_elig": active.iloc[-1]["max_elig"],
            "first_round": active["round_num"].min(),
            "last_round": active["round_num"].max(),
            "initial_waivers": int(active.iloc[0]["rmng_waivr"]),
            "final_waivers": int(active.iloc[-1]["rmng_waivr"]),
            "n_rounds_active": len(active),
        })
    df = pd.DataFrame(records)
    df["decay"] = df["final_elig"] / df["initial_elig"]
    df["survived"] = df["last_round"] == elig["round_num"].max()
    return df


def aggregate_by_round(elig):
    """Per-round aggregates: total eligibility, active bidders."""
    g = elig[elig["max_elig"] > 0].groupby("round_num")
    return pd.DataFrame({
        "n_active": g["bidder_num_fox"].nunique(),
        "total_elig": g["max_elig"].sum(),
        "mean_elig": g["max_elig"].mean(),
        "median_elig": g["max_elig"].median(),
    })


def report():
    elig, bidders = load()
    n_rounds = elig["round_num"].max()
    summary = bidder_summary(elig, bidders)
    by_round = aggregate_by_round(elig)

    surv = summary[summary["survived"]]
    early = summary[~summary["survived"]]

    print(f"{'='*70}")
    print(f"  C-BLOCK ELIGIBILITY EVOLUTION ({n_rounds} rounds, {len(summary)} bidders)")
    print(f"{'='*70}")

    print(f"\n  1. AGGREGATE BY ROUND")
    print(f"  {'Round':<10} {'Active':>8} {'Total elig':>15} {'Mean elig':>15}")
    print(f"  {'-'*50}")
    for r in [1, 10, 25, 50, 75, 100, 125, 150, 175, n_rounds]:
        if r in by_round.index:
            row = by_round.loc[r]
            print(f"  {r:<10} {int(row['n_active']):>8} {row['total_elig']:>15.0f} {row['mean_elig']:>15.0f}")

    r1 = by_round.loc[1]
    rl = by_round.loc[n_rounds]
    print(f"\n  Total elig decay (round {n_rounds} / round 1): {rl['total_elig']/r1['total_elig']:.2%}")
    print(f"  Active bidders: {int(r1['n_active'])} -> {int(rl['n_active'])}")

    print(f"\n  2. BIDDER-LEVEL DECAY (final_elig / initial_elig)")
    print(f"  {'':30} {'All':>10} {'Survivors':>10} {'Exiters':>10}")
    print(f"  {'-'*62}")
    for label, fn in [("N", len), ("mean", lambda x: x["decay"].mean()),
                      ("median", lambda x: x["decay"].median()),
                      ("p10", lambda x: x["decay"].quantile(0.1)),
                      ("p25", lambda x: x["decay"].quantile(0.25)),
                      ("min", lambda x: x["decay"].min())]:
        a = fn(summary) if label == "N" else fn(summary)
        s = fn(surv) if label == "N" else fn(surv)
        e = fn(early) if label == "N" else fn(early)
        fmt = "d" if label == "N" else ".4f"
        print(f"  {label:<30} {a:>10{fmt}} {s:>10{fmt}} {e:>10{fmt}}")

    print(f"\n  3. EXIT TIMING (early exiters)")
    print(f"  {'':30} {'Value':>10}")
    print(f"  {'-'*42}")
    print(f"  {'N exiters':<30} {len(early):>10}")
    print(f"  {'Mean last round':<30} {early['last_round'].mean():>10.0f}")
    print(f"  {'Median last round':<30} {early['last_round'].median():>10.0f}")
    print(f"  {'Exit by round 25':<30} {(early['last_round'] <= 25).sum():>10}")
    print(f"  {'Exit by round 50':<30} {(early['last_round'] <= 50).sum():>10}")
    print(f"  {'Exit by round 100':<30} {(early['last_round'] <= 100).sum():>10}")

    print(f"\n  4. WAIVERS")
    print(f"  {'':30} {'Survivors':>10} {'Exiters':>10}")
    print(f"  {'-'*52}")
    print(f"  {'Initial waivers (mean)':<30} {surv['initial_waivers'].mean():>10.1f} {early['initial_waivers'].mean():>10.1f}")
    print(f"  {'Final waivers (mean)':<30} {surv['final_waivers'].mean():>10.1f} {early['final_waivers'].mean():>10.1f}")
    print(f"  {'Used all waivers':<30} {(surv['final_waivers']==0).sum():>10} {(early['final_waivers']==0).sum():>10}")

    print(f"\n  5. TOP 10 SURVIVORS BY INITIAL ELIGIBILITY")
    top = surv.nlargest(10, "initial_elig")
    print(f"  {'Bidder':<30} {'Init elig':>12} {'Final elig':>12} {'Decay':>8} {'Waivers':>8}")
    print(f"  {'-'*72}")
    for _, row in top.iterrows():
        name = row["co_name"][:28]
        print(f"  {name:<30} {row['initial_elig']:>12.0f} {row['final_elig']:>12.0f} {row['decay']:>8.2%} {row['final_waivers']:>8}")

    print(f"\n  6. LARGEST DECAY (survivors only)")
    worst = surv.nsmallest(10, "decay")
    print(f"  {'Bidder':<30} {'Init elig':>12} {'Final elig':>12} {'Decay':>8} {'Waivers':>8}")
    print(f"  {'-'*72}")
    for _, row in worst.iterrows():
        name = row["co_name"][:28]
        print(f"  {name:<30} {row['initial_elig']:>12.0f} {row['final_elig']:>12.0f} {row['decay']:>8.2%} {row['final_waivers']:>8}")

    print(f"\n  7. OVERSTATING ELIGIBILITY")
    print(f"  pops_eligible (form 175) vs initial max_elig (round 1):")
    ratio = summary["initial_elig"] / summary["pops_eligible"]
    print(f"  {'Ratio (initial_elig / pops_eligible)':<40} {'mean':>8} {'median':>8} {'min':>8} {'max':>8}")
    print(f"  {'':40} {ratio.mean():>8.1f} {ratio.median():>8.1f} {ratio.min():>8.1f} {ratio.max():>8.1f}")
    print(f"\n  initial_elig is in raw population units; pops_eligible is in")
    print(f"  form-175 units. The ratio reflects the unit conversion, not overstating.")
    print(f"  The key question is whether pops_eligible (used in estimation) reflects")
    print(f"  the effective constraint at the time of the winning bid.")
    print(f"\n  Effective elig at last active round / initial elig:")
    print(f"  {'All bidders':<30} {summary['decay'].mean():>8.2%}")
    print(f"  {'Winners (survived)':<30} {surv['decay'].mean():>8.2%}")
    print(f"  -> Winners used {1-surv['decay'].mean():.0%} of initial eligibility on average")


if __name__ == "__main__":
    report()
