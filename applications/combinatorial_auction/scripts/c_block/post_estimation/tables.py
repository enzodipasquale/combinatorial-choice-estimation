"""Print post-estimation tables for preferred specifications."""
import numpy as np
from .compute import run_all, PREFERRED


def _col(rows, key):
    return np.array([r[key] for r in rows])


def _fmt(mean, se):
    return f"{mean:.2f} ({se:.2f})"


def _fmt4(mean, se):
    return f"{mean:.4f} ({se:.4f})"


def table_first_stage(results):
    """Table 1: First-stage covariate estimates."""
    specs = list(results.keys())
    all_covs = set()
    for rows, names in results.values():
        all_covs.update(names)

    # order: preserve each spec's ordering, merge
    ordered = []
    for rows, names in results.values():
        for n in names:
            if n not in ordered:
                ordered.append(n)

    W = 30
    col_w = 20
    header = f"  {'Covariate':<{W}}" + "".join(f"  {s:>{col_w}}" for s in specs)
    print(f"\n  TABLE 1: FIRST-STAGE ESTIMATES (boot mean, boot SE)")
    print(f"  {'='*(W + (col_w+2)*len(specs))}")
    print(header)
    print(f"  {'-'*(W + (col_w+2)*len(specs))}")
    for k in ordered:
        row = f"  {k:<{W}}"
        for stem in specs:
            rows, names = results[stem]
            if k in names:
                v = _col(rows, f"theta_{k}")
                row += f"  {_fmt(v.mean(), v.std()):>{col_w}}"
            else:
                row += f"  {'—':>{col_w}}"
        print(row)
    print()


def table_iv(results):
    """Table 2: Second-stage IV."""
    specs = list(results.keys())
    W = 30
    col_w = 20

    print(f"  TABLE 2: SECOND-STAGE IV (pop+hhinc, d>500km)")
    print(f"  {'='*(W + (col_w+2)*len(specs))}")
    print(f"  {'':<{W}}" + "".join(f"  {s:>{col_w}}" for s in specs))
    print(f"  {'-'*(W + (col_w+2)*len(specs))}")

    for label, fn in [
        ("alpha_0", lambda r: _col(r, "a0")),
        ("alpha_1", lambda r: _col(r, "a1")),
        ("alpha_0/alpha_1 ($M)", lambda r: _col(r, "a0") / _col(r, "a1") * 1e3),
        ("IV R2", lambda r: _col(r, "r2")),
    ]:
        row = f"  {label:<{W}}"
        for stem in specs:
            rows, _ = results[stem]
            v = fn(rows)
            if label == "IV R2":
                row += f"  {f'{v.mean():.4f}':>{col_w}}"
            else:
                row += f"  {_fmt(v.mean(), v.std()):>{col_w}}"
        print(row)

    n_row = f"  {'N draws':<{W}}"
    for stem in specs:
        rows, _ = results[stem]
        n_row += f"  {str(len(rows)):>{col_w}}"
    print(n_row)
    print()


def table_surplus(results):
    """Table 3: Surplus decomposition ($B)."""
    specs = list(results.keys())
    W = 30
    col_w = 20

    # collect all covariate names in order
    ordered = []
    for rows, names in results.values():
        for n in names:
            if n not in ordered:
                ordered.append(n)

    print(f"  TABLE 3: SURPLUS DECOMPOSITION ($B = utils / alpha_1)")
    print(f"  {'='*(W + (col_w+2)*len(specs))}")
    print(f"  {'':<{W}}" + "".join(f"  {s:>{col_w}}" for s in specs))
    print(f"  {'-'*(W + (col_w+2)*len(specs))}")

    def _row(label, key_fn, fmt=_fmt4):
        row = f"  {label:<{W}}"
        for stem in specs:
            rows, _ = results[stem]
            v = key_fn(rows, stem)
            row += f"  {fmt(v.mean(), v.std()):>{col_w}}"
        print(row)

    a1_cache = {s: _col(r, "a1") for s, (r, _) in results.items()}

    _row("Total surplus", lambda r, s: _col(r, "surplus") / a1_cache[s])
    print(f"  {'-'*(W + (col_w+2)*len(specs))}")

    for k in ordered:
        def _cov_fn(r, s, k=k):
            rows, names = results[s]
            if k not in names:
                return np.full(len(rows), np.nan)
            return _col(rows, f"contrib_{k}") / a1_cache[s]

        label = f"  {k}"
        row = f"  {label:<{W}}"
        for stem in specs:
            v = _cov_fn(None, stem)
            if np.isnan(v).all():
                row += f"  {'—':>{col_w}}"
            else:
                row += f"  {_fmt4(v.mean(), v.std()):>{col_w}}"
        print(row)

    _row("  delta", lambda r, s: _col(r, "fe_total") / a1_cache[s])
    _row("    n*a0/a1", lambda r, s: _col(r, "a0_part") / a1_cache[s])
    _row("    -revenue", lambda r, s: -_col(r, "price_part") / a1_cache[s])
    _row("    sum xi/a1", lambda r, s: _col(r, "xi_part") / a1_cache[s])

    print(f"  {'-'*(W + (col_w+2)*len(specs))}")
    _row("Entropy of choice", lambda r, s: _col(r, "entropy") / a1_cache[s])
    print()


def table_counterfactual(results):
    """Table 4: Counterfactual welfare (placeholder)."""
    print(f"  TABLE 4: COUNTERFACTUAL BTA vs MTA (requires bootstrap_welfare.py runs)")
    print(f"  (not yet computed)\n")


def run(specs=None):
    results = run_all(specs)
    table_first_stage(results)
    table_iv(results)
    table_surplus(results)
    table_counterfactual(results)
