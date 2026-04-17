"""Bidder descriptives table (LaTeX)."""
import numpy as np

from applications.combinatorial_auction.data.loaders import load_raw, build_context
from applications.combinatorial_auction.data.descriptive import OUT_TAB


def _fmt(x, kind):
    if np.isnan(x):
        return "--"
    if kind == "int":
        return f"{int(round(x)):,}"
    if kind == "m":          # USD millions or population millions
        v = x / 1e6
        return f"{v:,.2f}" if v < 10 else f"{v:,.1f}"
    if kind == "pct":
        return f"{100 * x:.0f}\\%"
    return f"{x:,.2f}"


def _stats_row(label, vals, kind):
    v = np.asarray(vals, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return f"{label} & " + " & ".join(["--"] * 6) + r" \\"
    mean, med = np.mean(v), np.median(v)
    q25, q75 = np.quantile(v, 0.25), np.quantile(v, 0.75)
    lo, hi = np.min(v), np.max(v)
    return (f"{label} & {_fmt(mean, kind)} & {_fmt(med, kind)} & "
            f"{_fmt(q25, kind)} & {_fmt(q75, kind)} & "
            f"{_fmt(lo, kind)} & {_fmt(hi, kind)} \\\\")


def build_table():
    raw = load_raw()
    ctx = build_context(raw)
    bidder, bta = raw["bidder_data"], raw["bta_data"]
    c_obs = ctx["c_obs_bundles"]
    elig   = bidder["pops_eligible"].values
    assets = bidder["assets"].values
    rev    = bidder["revenues"].values
    win    = c_obs.any(axis=1)

    pkg_size = c_obs.sum(axis=1)
    pkg_pop  = c_obs @ bta["pop90"].values
    pkg_bid  = c_obs @ bta["bid"].values

    n_all, n_win, n_los = len(bidder), int(win.sum()), int((~win).sum())

    rows = [
        r"\begin{tabular}{l cccccc}",
        r"\toprule",
        r" & Mean & Median & p25 & p75 & Min & Max \\",
        r"\midrule",
        rf"\multicolumn{{7}}{{l}}{{\emph{{Panel A. All bidders ($N = {n_all}$)}}}} \\[2pt]",
        _stats_row("Eligibility (M pop)", elig, "m"),
        _stats_row("Assets (\\$M)",       assets, "m"),
        _stats_row("Revenues (\\$M)",     rev, "m"),
        rf"Designated bidders & \multicolumn{{6}}{{c}}{{{_fmt(bidder['designated'].mean(), 'pct')}}} \\",
        r"\addlinespace",
        rf"\multicolumn{{7}}{{l}}{{\emph{{Panel B. Winners ($N = {n_win}$)}}}} \\[2pt]",
        _stats_row("Eligibility (M pop)",       elig[win], "m"),
        _stats_row("Package size (\\# BTAs)",   pkg_size[win], "int"),
        _stats_row("Package pop (M)",           pkg_pop[win] / 1e6, "num"),
        _stats_row("Total winning bid (\\$M)",  pkg_bid[win], "m"),
        r"\addlinespace",
        rf"\multicolumn{{7}}{{l}}{{\emph{{Panel C. Non-winners ($N = {n_los}$)}}}} \\[2pt]",
        _stats_row("Eligibility (M pop)", elig[~win], "m"),
        r"\bottomrule",
        r"\end{tabular}",
    ]
    tex = "\n".join(rows) + "\n"
    (OUT_TAB / "tab_bidders.tex").write_text(tex)
    print(f"  tab_bidders.tex ({n_all}/{n_win}/{n_los} all/win/lose)")
    return tex


if __name__ == "__main__":
    print(build_table())
