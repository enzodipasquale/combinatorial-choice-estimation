"""Bidder descriptives tables (LaTeX).

    tab_bidders.tex         — pooled over all bidders (main slide).
    tab_bidders_split.tex   — Winners vs Non-winners side-by-side (appendix).
"""
import numpy as np

from applications.combinatorial_auction.data.loaders import load_raw, build_context
from applications.combinatorial_auction.data.descriptive import OUT_TAB


def _fmt(x, kind):
    if np.isnan(x):
        return "--"
    if kind == "m2":   return f"{x / 1e6:,.2f}"
    if kind == "m1":   return f"{x / 1e6:,.1f}"
    if kind == "int":  return f"{int(round(x)):,}"
    raise ValueError(kind)


def _row(label, vals, kind):
    v = np.asarray(vals, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return f"{label} & -- & -- \\\\"
    return f"{label} & {_fmt(v.mean(), kind)} & {_fmt(np.median(v), kind)} \\\\"


def _split_cell(vals, kind):
    v = np.asarray(vals, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return "-- & --"
    return f"{_fmt(v.mean(), kind)} & {_fmt(np.median(v), kind)}"


def _build_inputs():
    raw = load_raw()
    ctx = build_context(raw)
    bidders = raw["bidder_data"]
    btas = raw["bta_data"]
    c_obs = ctx["c_obs_bundles"]

    win = c_obs.any(axis=1)
    pkg_size = c_obs.sum(axis=1).astype(float)
    pkg_pop  = c_obs.astype(float) @ btas["pop90"].values.astype(float)
    pkg_bid  = c_obs.astype(float) @ btas["bid"].values.astype(float)

    return {
        "bidders": bidders,
        "win": win,
        "elig":   bidders["pops_eligible"].values.astype(float),
        "assets": bidders["assets"].values.astype(float),
        "rev":    bidders["revenues"].values.astype(float),
        "pkg_size": pkg_size,
        "pkg_pop":  pkg_pop,
        "pkg_bid":  pkg_bid,
    }


def build_main_table():
    d = _build_inputs()
    n_all = len(d["bidders"])

    rows = [
        r"\begin{tabular}{l cc}",
        r"\toprule",
        r" & Mean & Median \\",
        r"\midrule",
        rf"\multicolumn{{3}}{{l}}{{\textit{{All bidders ($N = {n_all}$)}}}} \\[2pt]",
        _row("Eligibility (millions pop)",         d["elig"],     "m2"),
        _row("Assets (millions \\$)",              d["assets"],   "m1"),
        _row("Revenues (millions \\$)",            d["rev"],      "m2"),
        _row("Package size (\\# BTAs)",            d["pkg_size"], "int"),
        _row("Package pop (millions)",             d["pkg_pop"],  "m2"),
        _row("Total winning bid (millions \\$)",   d["pkg_bid"],  "m1"),
        r"\bottomrule",
        r"\end{tabular}",
    ]
    tex = "\n".join(rows) + "\n"
    (OUT_TAB / "tab_bidders.tex").write_text(tex)
    print(f"  tab_bidders.tex (N = {n_all})")
    return tex


def build_split_table():
    """Appendix: side-by-side Winners vs Non-winners (Mean + Median each)."""
    d = _build_inputs()
    w, nw = d["win"], ~d["win"]
    n_w, n_nw = int(w.sum()), int(nw.sum())

    def row(label, vals, kind):
        return (f"{label} & {_split_cell(vals[w], kind)} "
                f"& {_split_cell(vals[nw], kind)} \\\\")

    rows = [
        r"\begin{tabular}{l cc cc}",
        r"\toprule",
        rf" & \multicolumn{{2}}{{c}}{{Winners ($N = {n_w}$)}} "
        rf"& \multicolumn{{2}}{{c}}{{Non-winners ($N = {n_nw}$)}} \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}",
        r" & Mean & Median & Mean & Median \\",
        r"\midrule",
        row("Eligibility (millions pop)",         d["elig"],     "m2"),
        row("Assets (millions \\$)",              d["assets"],   "m1"),
        row("Revenues (millions \\$)",            d["rev"],      "m2"),
        row("Package size (\\# BTAs)",            d["pkg_size"], "int"),
        row("Package pop (millions)",             d["pkg_pop"],  "m2"),
        row("Total winning bid (millions \\$)",   d["pkg_bid"],  "m1"),
        r"\bottomrule",
        r"\end{tabular}",
    ]
    tex = "\n".join(rows) + "\n"
    (OUT_TAB / "tab_bidders_split.tex").write_text(tex)
    print(f"  tab_bidders_split.tex (winners={n_w}, non-winners={n_nw})")
    return tex


if __name__ == "__main__":
    print(build_main_table())
    print()
    print(build_split_table())
