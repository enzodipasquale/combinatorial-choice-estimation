"""Round-level eligibility evolution during the C-block auction.

Exposes three primitives:
    load()                → (elig_df, bidders_df)
    bidder_summary(...)   → per-bidder decay, last active round, waivers
    aggregate_by_round(.) → total/mean/median eligibility per round
"""
import pandas as pd

from applications.combinatorial_auction.data.loaders import RAW


def load():
    elig = pd.read_csv(RAW / "cblock-eligibility.csv")
    bidders = pd.read_csv(RAW / "biddercblk_03_28_2004_pln.csv")
    bidders = bidders[bidders["bidder_num"] != 9999]
    return elig, bidders


def bidder_summary(elig, bidders):
    rows = []
    last_round = elig["round_num"].max()
    for bn in bidders["bidder_num_fox"]:
        active = elig[(elig["bidder_num_fox"] == bn) &
                      (elig["max_elig"] > 0)].sort_values("round_num")
        if active.empty:
            continue
        first, last = active.iloc[0], active.iloc[-1]
        rows.append({
            "bidder": bn,
            "co_name": bidders.loc[bidders["bidder_num_fox"] == bn, "co_name"].iloc[0],
            "pops_eligible": bidders.loc[bidders["bidder_num_fox"] == bn, "pops_eligible"].iloc[0],
            "initial_elig": first["max_elig"],
            "final_elig": last["max_elig"],
            "first_round": first["round_num"],
            "last_round": last["round_num"],
            "initial_waivers": int(first["rmng_waivr"]),
            "final_waivers": int(last["rmng_waivr"]),
            "n_rounds_active": len(active),
        })
    df = pd.DataFrame(rows)
    df["decay"] = df["final_elig"] / df["initial_elig"]
    df["survived"] = df["last_round"] == last_round
    return df


def aggregate_by_round(elig):
    g = elig[elig["max_elig"] > 0].groupby("round_num")
    return pd.DataFrame({
        "n_active":    g["bidder_num_fox"].nunique(),
        "total_elig":  g["max_elig"].sum(),
        "mean_elig":   g["max_elig"].mean(),
        "median_elig": g["max_elig"].median(),
    })


if __name__ == "__main__":
    elig, bidders = load()
    s = bidder_summary(elig, bidders)
    r = aggregate_by_round(elig)
    surv, exit_ = s[s["survived"]], s[~s["survived"]]
    n_rounds = elig["round_num"].max()
    print(f"{n_rounds} rounds, {len(s)} bidders "
          f"({len(surv)} survivors, {len(exit_)} exiters)")
    print(f"total-elig decay r{n_rounds}/r1 = "
          f"{r.loc[n_rounds, 'total_elig'] / r.loc[1, 'total_elig']:.1%}")
    print(f"winner decay (final/initial elig): "
          f"mean={surv['decay'].mean():.2%}, median={surv['decay'].median():.2%}")
