"""Pretty-print post-estimation tables across specs."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

from ...data.loaders import load_raw, DATASETS
from .compute import decompose_all, RESULTS

W, CW = 30, 20


def _col(rows, key):
    return np.array([r[key] for r in rows], dtype=float)


def _ms(mean, se, prec=2):
    return f"{mean:.{prec}f} ({se:.{prec}f})"


def _row(label, cells):
    print(f"  {label:<{W}}" + "".join(f"  {c:>{CW}}" for c in cells))


def _sep(n_specs, ch="="):
    print(f"  {ch * (W + (CW + 2) * n_specs)}")


def _header(specs):
    _row("", specs)
    _sep(len(specs), "-")


def _all_names_in_order(results):
    return list(dict.fromkeys(n for _, names in results.values() for n in names))


def _row_div_by_a1(label, key, results, a1, prec=4):
    """Print a row of (mean, std) for `key` divided by α₁, one cell per spec."""
    cells = [_ms((v := _col(r, key) / a1[s]).mean(), v.std(), prec)
             for s, (r, _) in results.items()]
    _row(label, cells)


def table_first_stage(results):
    specs = list(results.keys())
    print("\n  TABLE 1: FIRST-STAGE ESTIMATES  (boot mean, boot SE)")
    _sep(len(specs))
    _header(specs)
    for k in _all_names_in_order(results):
        cells = []
        for s in specs:
            rows, names = results[s]
            if k in names:
                v = _col(rows, f"theta_{k}")
                cells.append(_ms(v.mean(), v.std()))
            else:
                cells.append("—")
        _row(k, cells)
    print()


def table_iv(results):
    specs = list(results.keys())
    print("  TABLE 2: SECOND-STAGE IV")
    _sep(len(specs))
    _header(specs)
    for label, vec in [
        ("alpha_0",              lambda r: _col(r, "a0")),
        ("alpha_1",              lambda r: _col(r, "a1")),
        ("alpha_0/alpha_1 ($M)", lambda r: _col(r, "a0") / _col(r, "a1") * 1e3),
    ]:
        _row(label, [_ms(v.mean(), v.std()) for v in (vec(results[s][0]) for s in specs)])
    _row("IV R2",   [f"{_col(results[s][0], 'r2').mean():.4f}" for s in specs])
    _row("N draws", [str(len(results[s][0])) for s in specs])
    print()


def table_surplus(results):
    specs = list(results.keys())
    a1 = {s: _col(r, "a1") for s, (r, _) in results.items()}
    show_controls = any(_col(results[s][0], "controls_part").any() for s in specs)

    print("  TABLE 3: SURPLUS DECOMPOSITION  ($B = utils / alpha_1)")
    _sep(len(specs))
    _header(specs)

    _row_div_by_a1("Total surplus", "surplus", results, a1)
    _sep(len(specs), "-")

    # Named covariate contributions (may not exist in every spec).
    for k in _all_names_in_order(results):
        cells = []
        for s in specs:
            rows, names = results[s]
            if k in names:
                v = _col(rows, f"contrib_{k}") / a1[s]
                cells.append(_ms(v.mean(), v.std(), 4))
            else:
                cells.append("—")
        _row(k, cells)

    # Delta and its decomposition.
    _row_div_by_a1("  delta",       "fe_total",      results, a1)
    _row_div_by_a1("    n*a0/a1",   "a0_part",       results, a1)
    # -revenue = -price_part (display the negated mean).
    _row("    -revenue",
         [_ms(-(v := _col(r, "price_part") / a1[s]).mean(), v.std(), 4)
          for s, (r, _) in results.items()])
    if show_controls:
        _row_div_by_a1("    controls/a1", "controls_part", results, a1)
    _row_div_by_a1("    sum xi/a1", "xi_part",       results, a1)
    _sep(len(specs), "-")
    _row_div_by_a1("Entropy of choice", "entropy",    results, a1)
    print()


def table_per_bidder_surplus(results, top=15):
    """Top-N bidders by model-implied surplus ($B = utils/α₁).

    For bootstrap rows: averages per-bidder surplus across converged draws.
    Merges with bidder_data.csv (co_name, pops_eligible, Applicant_Status) so
    the top-bidder list can be eyeballed against real-world names.
    """
    bd = pd.read_csv(DATASETS / "bidder_data.csv")[
        ["bidder_num_fox", "co_name", "pops_eligible", "Applicant_Status"]
    ]
    print("  TABLE 4: TOP BIDDERS BY MODEL-IMPLIED SURPLUS  ($B)")
    for spec, (rows, _) in results.items():
        # Stack per-bidder arrays across draws; shape (n_draws, n_obs)
        arr_B = np.array([r["bidder_surplus_B"] for r in rows])
        arr_utils = np.array([r["bidder_surplus_utils"] for r in rows])
        mean_B  = arr_B.mean(axis=0)
        std_B   = arr_B.std(axis=0)
        df = pd.DataFrame({
            "surp_B_mean": mean_B,
            "surp_B_std":  std_B,
            "surp_utils":  arr_utils.mean(axis=0),
        })
        # Load raw bidder data in estimation order (from Fox-Bajari loader).
        raw = load_raw()
        bidders = raw["bidder_data"][["bidder_num_fox", "pops_eligible"]].copy()
        bidders = bidders.merge(bd[["bidder_num_fox", "co_name", "Applicant_Status"]],
                                on="bidder_num_fox", how="left")
        df = pd.concat([bidders.reset_index(drop=True), df], axis=1)
        df = df.sort_values("surp_B_mean", ascending=False).head(top).reset_index(drop=True)
        print(f"\n  {spec}  (total bidders: {len(mean_B)}, total surplus: {mean_B.sum():.4f} $B)")
        # Compact display
        for _, r in df.iterrows():
            de_tag = ""
            if isinstance(r.get("Applicant_Status"), str):
                de_tag = " [" + ",".join(x.strip()[:3] for x in r["Applicant_Status"].split(",")) + "]"
            print(f"    bidder {int(r['bidder_num_fox']):>3d}  "
                  f"{(r.get('co_name') or '—')[:35]:<35} "
                  f"surp=${r['surp_B_mean']:+8.4f}B "
                  f"(±{r['surp_B_std']:.4f})  "
                  f"elig={r['pops_eligible']:.2e}{de_tag}")
    print()


def run(specs):
    results = decompose_all(specs)
    if not results:
        print("No results to display.")
        return
    table_first_stage(results)
    table_iv(results)
    table_surplus(results)
    table_per_bidder_surplus(results)
