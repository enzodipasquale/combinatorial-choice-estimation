"""BTA (license) descriptives table (LaTeX)."""
import numpy as np

from applications.combinatorial_auction.data.loaders import load_raw
from applications.combinatorial_auction.data.descriptive import OUT_TAB


def _fmt(x, kind):
    if np.isnan(x):
        return "--"
    if kind == "int":
        return f"{int(round(x)):,}"
    if kind == "k":
        return f"{x / 1e3:,.0f}"
    if kind == "m":
        return f"{x / 1e6:,.1f}"
    if kind == "pct":
        return f"{x * 100:.1f}\\%"
    return f"{x:,.2f}"


def _stats_row(label, vals, kind):
    v = np.asarray(vals, dtype=float)
    v = v[~np.isnan(v)]
    mean, med = np.mean(v), np.median(v)
    q25, q75 = np.quantile(v, 0.25), np.quantile(v, 0.75)
    lo, hi = np.min(v), np.max(v)
    return (f"{label} & {_fmt(mean, kind)} & {_fmt(med, kind)} & "
            f"{_fmt(q25, kind)} & {_fmt(q75, kind)} & "
            f"{_fmt(lo, kind)} & {_fmt(hi, kind)} \\\\")


def build_table():
    bta = load_raw()["bta_data"]
    pop = bta["pop90"].values
    bid = bta["bid"].values

    rows = [
        r"\begin{tabular}{l cccccc}",
        r"\toprule",
        r" & Mean & Median & p25 & p75 & Min & Max \\",
        r"\midrule",
        _stats_row("Population 1990 (K)",          pop, "k"),
        _stats_row("Winning bid (\\$M)",           bid, "m"),
        _stats_row("Per-capita income (\\$)",      bta["percapin"].values, "int"),
        _stats_row("Density (pop / km$^2$)",       bta["density"].values, "int"),
        _stats_row("Share HH income $<$ \\$35K",   bta["hhinc35k"].values, "pct"),
        r"\midrule",
        rf"Number of BTAs (continental) & \multicolumn{{6}}{{c}}{{{len(bta):,}}} \\",
        rf"Total winning bids & \multicolumn{{6}}{{c}}{{\${bid.sum() / 1e9:,.2f}B}} \\",
        rf"Total population 1990 & \multicolumn{{6}}{{c}}{{{pop.sum() / 1e6:,.1f}M}} \\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    tex = "\n".join(rows) + "\n"
    (OUT_TAB / "tab_btas.tex").write_text(tex)
    print(f"  tab_btas.tex ({len(bta)} BTAs, ${bid.sum() / 1e9:.2f}B total)")
    return tex


if __name__ == "__main__":
    print(build_table())
