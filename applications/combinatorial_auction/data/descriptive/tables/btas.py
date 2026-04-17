"""BTA (license) descriptive statistics table (LaTeX)."""

import numpy as np
from pathlib import Path

from applications.combinatorial_auction.data.loaders import load_bta_data

OUT = Path(__file__).parent.parent / "output"


def _fmt(x, kind="num"):
    if np.isnan(x):
        return "--"
    if kind == "int":
        return f"{int(round(x)):,}"
    if kind == "k":
        return f"{x / 1e3:,.0f}"
    if kind == "m":
        return f"{x / 1e6:,.1f}"
    if kind == "usd":
        return f"\\${x:,.0f}"
    return f"{x:,.2f}"


def _stats(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    return dict(
        mean=np.mean(x), median=np.median(x),
        q25=np.quantile(x, 0.25), q75=np.quantile(x, 0.75),
        lo=np.min(x), hi=np.max(x),
    )


def build_table():
    raw = load_bta_data()
    bta = raw["bta_data"]

    pop = bta["pop90"].values
    bid = bta["bid"].values
    percapin = bta["percapin"].values
    density = bta["density"].values
    hhinc35k = bta["hhinc35k"].values

    def row(label, vals, kind):
        s = _stats(vals)
        return (f"{label} & {_fmt(s['mean'], kind)} & {_fmt(s['median'], kind)} "
                f"& {_fmt(s['q25'], kind)} & {_fmt(s['q75'], kind)} "
                f"& {_fmt(s['lo'], kind)} & {_fmt(s['hi'], kind)} \\\\")

    lines = [
        r"\begin{tabular}{l cccccc}",
        r"\toprule",
        r" & Mean & Median & p25 & p75 & Min & Max \\",
        r"\midrule",
        row("Population 1990 (K)", pop, "k"),
        row("Winning bid (\\$M)", bid, "m"),
        row("Per-capita income (\\$)", percapin, "int"),
        row("Density (pop / km$^2$)", density, "int"),
        r"Share HH income $<$ \$35K & " + " & ".join(
            [f"{v * 100:.1f}\\%" for v in [
                _stats(hhinc35k)[k] for k in ("mean", "median", "q25", "q75", "lo", "hi")
            ]]
        ) + r" \\",
        r"\midrule",
        f"Number of BTAs (continental) & \\multicolumn{{6}}{{c}}{{{len(bta):,}}} \\\\",
        f"Total winning bids & \\multicolumn{{6}}{{c}}{{\\${bid.sum() / 1e9:,.2f}B}} \\\\",
        f"Total population 1990 & \\multicolumn{{6}}{{c}}{{{pop.sum() / 1e6:,.1f}M}} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]

    tex = "\n".join(lines) + "\n"
    (OUT / "tab_btas.tex").write_text(tex)
    print(f"  tab_btas.tex written ({len(lines)} lines)")
    return tex


if __name__ == "__main__":
    print(build_table())
