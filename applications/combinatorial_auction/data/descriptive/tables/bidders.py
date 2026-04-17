"""Bidder descriptive statistics table (LaTeX)."""

import numpy as np
from pathlib import Path

from applications.combinatorial_auction.data.loaders import load_bta_data, build_context

OUT = Path(__file__).parent.parent / "output"


def _fmt(x, kind="num"):
    if np.isnan(x):
        return "--"
    if kind == "int":
        return f"{int(round(x)):,}"
    if kind == "pop_m":
        v = x / 1e6
        return f"{v:,.2f}" if v < 10 else f"{v:,.1f}"
    if kind == "usd_m":
        v = x / 1e6
        return f"{v:,.2f}" if v < 10 else f"{v:,.1f}"
    if kind == "pct":
        return f"{100 * x:.0f}\\%"
    return f"{x:,.2f}"


def _quantiles(x, qs=(0.25, 0.5, 0.75)):
    x = np.asarray(x)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return [np.nan] * len(qs)
    return [np.quantile(x, q) for q in qs]


def build_table():
    raw = load_bta_data()
    ctx = build_context(raw)
    bidder = raw["bidder_data"]
    bta = raw["bta_data"]
    c_obs = ctx["c_obs_bundles"]

    elig = bidder["pops_eligible"].values
    assets = bidder["assets"].values
    revenues = bidder["revenues"].values
    designated = bidder["designated"].values
    is_winner = c_obs.any(axis=1)

    pkg_size = c_obs.sum(axis=1)
    pkg_pop = c_obs @ bta["pop90"].values
    pkg_bid = c_obs @ bta["bid"].values

    def row(label, vals, kind):
        q25, q50, q75 = _quantiles(vals)
        mean = np.nanmean(vals) if len(vals) else np.nan
        lo = np.nanmin(vals) if len(vals) else np.nan
        hi = np.nanmax(vals) if len(vals) else np.nan
        return (f"{label} & "
                f"{_fmt(mean, kind)} & {_fmt(q50, kind)} & "
                f"{_fmt(q25, kind)} & {_fmt(q75, kind)} & "
                f"{_fmt(lo, kind)} & {_fmt(hi, kind)} \\\\")

    lines = [
        r"\begin{tabular}{l cccccc}",
        r"\toprule",
        r" & Mean & Median & p25 & p75 & Min & Max \\",
        r"\midrule",
        r"\multicolumn{7}{l}{\emph{Panel A. All bidders ($N = 252$)}} \\[2pt]",
        row("Eligibility (M pop)", elig, "pop_m"),
        row("Assets (\\$M)", assets, "usd_m"),
        row("Revenues (\\$M)", revenues, "usd_m"),
        f"Designated bidders & \\multicolumn{{6}}{{c}}{{{_fmt(designated.mean(), 'pct')}}} \\\\",
        r"\addlinespace",
        r"\multicolumn{7}{l}{\emph{Panel B. Winners ($N = 85$)}} \\[2pt]",
        row("Eligibility (M pop)", elig[is_winner], "pop_m"),
        row("Package size (\\# BTAs)", pkg_size[is_winner], "int"),
        row("Package pop (M)", pkg_pop[is_winner] / 1e6, "num"),
        row("Total winning bid (\\$M)", pkg_bid[is_winner], "usd_m"),
        r"\addlinespace",
        r"\multicolumn{7}{l}{\emph{Panel C. Non-winners ($N = 167$)}} \\[2pt]",
        row("Eligibility (M pop)", elig[~is_winner], "pop_m"),
        r"\bottomrule",
        r"\end{tabular}",
    ]

    tex = "\n".join(lines) + "\n"
    (OUT / "tab_bidders.tex").write_text(tex)
    print(f"  tab_bidders.tex written ({len(lines)} lines)")
    return tex


if __name__ == "__main__":
    print(build_table())
