"""BTA (license) descriptives table (LaTeX)."""
import numpy as np

from applications.combinatorial_auction.data.loaders import load_raw
from applications.combinatorial_auction.data.descriptive import OUT_TAB


def _fmt(x, kind):
    if np.isnan(x):
        return "--"
    if kind == "m2":  return f"{x / 1e6:,.2f}"
    if kind == "m1":  return f"{x / 1e6:,.1f}"
    raise ValueError(kind)


def _row(label, vals, kind):
    v = np.asarray(vals, dtype=float)
    v = v[~np.isnan(v)]
    return f"{label} & {_fmt(v.mean(), kind)} & {_fmt(np.median(v), kind)} \\\\"


def build_table():
    bta = load_raw()["bta_data"]
    pop = bta["pop90"].values
    bid = bta["bid"].values

    rows = [
        r"\begin{tabular}{l cc}",
        r"\toprule",
        r" & Mean & Median \\",
        r"\midrule",
        _row("Population (millions)",         pop, "m2"),
        _row("Winning bid (millions \\$)",    bid, "m1"),
        r"\midrule",
        rf"Number of BTAs     & \multicolumn{{2}}{{c}}{{{len(bta):,}}} \\",
        rf"Total winning bids & \multicolumn{{2}}{{c}}{{\${bid.sum() / 1e9:,.2f}B}} \\",
        rf"Total population   & \multicolumn{{2}}{{c}}{{{pop.sum() / 1e6:,.1f}M}} \\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    tex = "\n".join(rows) + "\n"
    (OUT_TAB / "tab_btas.tex").write_text(tex)
    print(f"  tab_btas.tex ({len(bta)} BTAs, ${bid.sum() / 1e9:.2f}B total)")
    return tex


if __name__ == "__main__":
    print(build_table())
