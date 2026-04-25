"""Bidder resilience to outbid events — binscatter of retention vs importance.

Outbid event
------------
A triple (i, j, t) where bidder i held SHB on license j at round t-1 and not
at t.  Excluded:
  * withdrawals: i appears in the withdrawal file for j at t.
  * short SHB runs: i was SHB on j for fewer than 3 consecutive rounds
    immediately before t.
  * eligibility-forced dropouts: i's max-elig at round t is below pop_j
    (pop_j is what i would need to keep holding this license under the
    activity rule), or i exited the auction at t.

Scalars per event
-----------------
    active_{i,t} = {j' : i held SHB or submitted a bid on j' in [t-w, t]}

    importance(i,j,t) = pop_j *
                        Σ_{j' ∈ active_{i,t}, j' ≠ j} pop_{j'} / d(j,j')^p

    retention(i,j,t)  =
        Σ_{j' ≠ j} pop_{j'} · 1{i submits a bid on j' in [t+1, t+w]} / d(j,j')^p
      ─────────────────────────────────────────────────────────────────────────
        Σ_{j' ∈ active_{i,t}, j' ≠ j} pop_{j'} / d(j,j')^p

Exhibit
-------
Binscatter: 10 log10(importance) bins × mean retention. Horizontal line at 1.

Robustness variants: w ∈ {5, 10, 20}, p ∈ {1, 2, 4}, optional non-binding-
eligibility subsample (bidders whose max_elig never dropped while active).

Run
---
    python -m applications.combinatorial_auction.data.descriptive.figures.resilience
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from applications.combinatorial_auction.data.loaders import load_raw, RAW
from applications.combinatorial_auction.data.descriptive import OUT_FIG
from applications.combinatorial_auction.data.descriptive.style import (
    NAVY, SLATE, GOLD, DPI, style_ax,
)

SHB_FILE  = RAW / "cblock-high-bids-after-withdrawal.csv"
BID_FILE  = RAW / "cblock-submitted-bids.csv"
WD_FILE   = RAW / "cblock-bid-withdrawals-in-current-round.csv"
ELIG_FILE = RAW / "cblock-eligibility.csv"
DIST_FILE = RAW / "geographic-distance-population-weighted-centroid.csv"

_CONSECUTIVE_MIN = 3    # require SHB for >= this many rounds immediately before t


# ── helpers ──────────────────────────────────────────────────────────

def _bta(series) -> pd.Series:
    out = series.astype(str).str.extract(r"^B(\d+)$", expand=False)
    return pd.to_numeric(out, errors="coerce").astype("Int64")


def _load_shb():
    """Wide SHB panel: index=round, columns=bta, values=bidder_num.

    The raw file is the full history — every row gives "as of file_round,
    the current SHB on market is bidder_num, last set in round_num".  We
    take each file_round's snapshot: all rows with that file_round.
    """
    df = pd.read_csv(SHB_FILE)
    df["bta"] = _bta(df["market"])
    df = df.dropna(subset=["bta"])
    df["bta"] = df["bta"].astype(int)
    # Each row in the file is "SHB as of file_round on market". Rows are
    # replayed every round. We pivot file_round × bta → bidder_num.
    df = df.drop_duplicates(["file_round", "bta"], keep="last")
    return df.pivot(index="file_round", columns="bta", values="bidder_num")


def _load_bids():
    df = pd.read_csv(BID_FILE)
    df["bta"] = _bta(df["market"])
    df = df.dropna(subset=["bta"])
    df["bta"] = df["bta"].astype(int)
    return df[["round_num", "bta", "bidder_num"]]


def _load_withdrawals():
    df = pd.read_csv(WD_FILE)
    df["bta"] = _bta(df["market"])
    df = df.dropna(subset=["bta", "round_num", "bidder_num"])
    return set(zip(df["round_num"].astype(int),
                   df["bta"].astype(int),
                   df["bidder_num"].astype(int)))


def _load_elig():
    """(round, bidder_num) → max_elig."""
    df = pd.read_csv(ELIG_FILE)[["round_num", "bidder_num", "max_elig"]]
    return df.set_index(["round_num", "bidder_num"])["max_elig"]


def _load_distance_km(bta_ids):
    """Load the 493×493 distance matrix (meters), subset to bta_ids, → km.
    bta i sits at row/col i-1."""
    raw = pd.read_csv(DIST_FILE, header=None).to_numpy(dtype=float)
    idx = np.asarray(bta_ids, dtype=int) - 1
    return raw[np.ix_(idx, idx)] / 1000.0


# ── event detection ─────────────────────────────────────────────────

def find_outbid_events(shb_wide: pd.DataFrame, withdraw: set,
                      elig: pd.Series, pops: np.ndarray, bta_ids: np.ndarray,
                      consec_min: int = _CONSECUTIVE_MIN) -> pd.DataFrame:
    """Return a DataFrame with one row per surviving outbid event
    (bidder_num, bta, round_num=t).  Excludes withdrawals, short runs, and
    eligibility-forced losses."""
    rounds = sorted(shb_wide.index.tolist())
    bid_to_pop = dict(zip(bta_ids, pops))
    rows = []
    # Shift: SHB_{t-1} vs SHB_t across common markets.
    prev = None
    for t in rounds:
        snap = shb_wide.loc[t]
        if prev is not None:
            # Compare previous-round SHB with this round's.
            common = snap.index.intersection(prev.index)
            p = prev[common]
            c = snap[common]
            # Candidate outbid: previous holder ≠ current holder, and
            # previous holder was non-null.
            mask = (p.notna()) & (p != c)
            for bta in common[mask]:
                i_prev = int(p[bta])
                # 1. Skip if i_prev withdrew in round t.
                if (t, int(bta), i_prev) in withdraw:
                    continue
                # 2. Require >= consec_min consecutive SHB rounds before t.
                ok = True
                for lag in range(1, consec_min + 1):
                    r = t - lag
                    if r not in shb_wide.index:
                        ok = False; break
                    past = shb_wide.loc[r].get(bta, np.nan)
                    if pd.isna(past) or int(past) != i_prev:
                        ok = False; break
                if not ok:
                    continue
                # 3. Skip eligibility-forced: bidder's max_elig at t below
                # pop_bta (they couldn't maintain it anyway) OR bidder gone.
                e = elig.get((t, i_prev), np.nan)
                if pd.isna(e) or e < bid_to_pop.get(int(bta), 0):
                    continue
                rows.append((i_prev, int(bta), t))
        prev = snap
    return pd.DataFrame(rows, columns=["bidder_num", "bta", "round_num"])


# ── active set + scalars ────────────────────────────────────────────

def _active_set(i: int, t: int, w: int, shb_wide, bids_by_it) -> set:
    """Union of SHB markets and markets with a submitted bid in [t-w, t]."""
    out = set()
    for r in range(max(1, t - w), t + 1):
        # SHB on any market in round r held by i:
        if r in shb_wide.index:
            row = shb_wide.loc[r]
            out.update(int(b) for b in row[row == i].index.tolist())
        # submitted bids by i in round r:
        out.update(bids_by_it.get((i, r), ()))
    return out


def compute_event_scalars(events: pd.DataFrame, shb_wide, bids_by_it,
                          pops, bta_ids, d_km, *, window: int, power: int):
    """Add importance, retention, active_size to the events DataFrame."""
    idx_of = {int(b): k for k, b in enumerate(bta_ids)}
    pop_arr = np.asarray(pops, dtype=float)
    out_rows = []

    # build bidder → round → set(bids)
    for idx, ev in events.iterrows():
        i, j, t = int(ev["bidder_num"]), int(ev["bta"]), int(ev["round_num"])
        if j not in idx_of:
            continue
        act = _active_set(i, t, window, shb_wide, bids_by_it) - {j}
        # filter to continental BTAs we have in idx_of.
        act = {a for a in act if a in idx_of}
        if not act:
            continue
        aj = idx_of[j]
        a_idx = np.array([idx_of[a] for a in act])

        d = d_km[aj, a_idx]
        d = np.where((d > 0) & np.isfinite(d), d, np.nan)
        w_jj = 1.0 / (d ** power)
        w_jj = np.where(np.isfinite(w_jj), w_jj, 0.0)
        pops_other = pop_arr[a_idx]

        # denominator (shared by importance and retention)
        denom = float((pops_other * w_jj).sum())
        if denom <= 0:
            continue

        imp = pop_arr[aj] * denom

        # numerator for retention: bids on j' in (t, t+w]
        # Build indicator across a_idx
        future_bids = set()
        for r in range(t + 1, t + window + 1):
            future_bids.update(bids_by_it.get((i, r), ()))
        in_future = np.array([1.0 if a in future_bids else 0.0
                              for a in act])
        numer = float((pops_other * w_jj * in_future).sum())
        ret = numer / denom
        out_rows.append((i, j, t, len(act), imp, ret))

    return pd.DataFrame(out_rows,
                        columns=["bidder_num", "bta", "round_num",
                                 "n_active", "importance", "retention"])


# ── binscatter ──────────────────────────────────────────────────────

def binscatter(ev: pd.DataFrame, outfile, *,
               title_suffix: str = "", n_bins: int = 10):
    x = np.log10(ev["importance"].values)
    y = ev["retention"].values

    cuts = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    cuts[0] -= 1e-9
    bins = np.digitize(x, cuts[1:-1])
    df = pd.DataFrame({"x": x, "y": y, "bin": bins})
    by = df.groupby("bin").agg(x_mid=("x", "mean"),
                                y_mean=("y", "mean"),
                                y_se=("y", "sem"),
                                n=("y", "size"))

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.axhline(1.0, color=SLATE, lw=0.9, ls="--", alpha=0.85)
    ax.errorbar(by["x_mid"], by["y_mean"],
                yerr=1.96 * by["y_se"].fillna(0),
                fmt="o", color=NAVY, ms=5, lw=1.1, capsize=2)
    ax.set_xlabel(r"$\log_{10}$(importance)", fontsize=10, family="serif")
    ax.set_ylabel("mean retention ratio", fontsize=10, family="serif")
    ax.set_ylim(bottom=0)
    style_ax(ax)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_family("serif")
    if title_suffix:
        ax.set_title(title_suffix, fontsize=9, family="serif", color=SLATE)
    fig.tight_layout()
    fig.savefig(outfile, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return by


# ── main ────────────────────────────────────────────────────────────

def _build_context():
    raw = load_raw()
    bta = raw["bta_data"]
    bta_ids = bta["bta"].astype(int).to_numpy()
    pops = bta["pop90"].astype(float).to_numpy()
    d_km = _load_distance_km(bta_ids)

    shb_wide = _load_shb()
    bids_df  = _load_bids()
    withdraw = _load_withdrawals()
    elig     = _load_elig()

    # bids indexed by (bidder, round) → set of btas
    bids_by_it = (bids_df.groupby(["bidder_num", "round_num"])["bta"]
                        .apply(lambda s: set(s.astype(int)))
                        .to_dict())
    return dict(bta_ids=bta_ids, pops=pops, d_km=d_km,
                shb_wide=shb_wide, bids_df=bids_df, bids_by_it=bids_by_it,
                withdraw=withdraw, elig=elig)


def _non_binding_bidders(elig: pd.Series) -> set:
    """Bidders whose max_elig never dropped (strictly) round-over-round while
    active.  A crude proxy for 'never hit the activity constraint'."""
    df = elig.reset_index()
    df = df.sort_values(["bidder_num", "round_num"])
    df["diff"] = df.groupby("bidder_num")["max_elig"].diff()
    # "active" means max_elig > 0.
    active = df[df["max_elig"] > 0]
    ever_drop = (active.groupby("bidder_num")["diff"]
                        .apply(lambda s: (s.fillna(0) < 0).any()))
    return set(ever_drop[~ever_drop].index.tolist())


def run(*, window: int = 10, power: int = 2,
        restrict_non_binding: bool = False, tag: str = "main",
        ctx=None):
    ctx = ctx or _build_context()
    events_raw = find_outbid_events(ctx["shb_wide"], ctx["withdraw"],
                                    ctx["elig"], ctx["pops"], ctx["bta_ids"])
    n_raw = len(events_raw)

    if restrict_non_binding:
        keep_bidders = _non_binding_bidders(ctx["elig"])
        events_raw = events_raw[events_raw["bidder_num"].isin(keep_bidders)]

    ev = compute_event_scalars(events_raw,
                               ctx["shb_wide"], ctx["bids_by_it"],
                               ctx["pops"], ctx["bta_ids"], ctx["d_km"],
                               window=window, power=power)

    out_png = OUT_FIG / f"fig_resilience_{tag}.png"
    out_csv = OUT_FIG / f"fig_resilience_{tag}.csv"

    suffix = f"window = {window} rounds, distance power {power}"
    if restrict_non_binding:
        suffix += ", non-binding elig subsample"

    by = binscatter(ev, out_png, title_suffix=suffix)
    ev.to_csv(out_csv, index=False)

    # summary
    top_dec = ev.sort_values("importance").tail(max(1, len(ev) // 10))
    print(f"  fig_resilience_{tag}: {len(ev):,} events "
          f"(raw {n_raw:,}; dropped {n_raw - len(ev):,})")
    print(f"    mean retention (top importance decile) = "
          f"{top_dec['retention'].mean():.3f} "
          f"(overall {ev['retention'].mean():.3f})")
    return ev, by


def main():
    ctx = _build_context()
    print("== main spec: w=10, p=2 ==")
    run(ctx=ctx, window=10, power=2, tag="main")
    print("\n== robustness ==")
    for w in (5, 20):
        run(ctx=ctx, window=w, power=2, tag=f"w{w}")
    for p in (1, 4):
        run(ctx=ctx, window=10, power=p, tag=f"p{p}")
    run(ctx=ctx, window=10, power=2, restrict_non_binding=True,
        tag="nonbinding")


if __name__ == "__main__":
    main()
