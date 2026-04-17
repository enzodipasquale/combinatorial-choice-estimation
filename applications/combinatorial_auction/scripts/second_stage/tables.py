"""Pretty-print post-estimation tables across specs."""
import numpy as np
from .compute import decompose_all

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
    ordered = []
    for _, names in results.values():
        for n in names:
            if n not in ordered:
                ordered.append(n)
    return ordered


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
    for label, fn, fmt in [
        ("alpha_0",              lambda r: _col(r, "a0"),                               _ms),
        ("alpha_1",              lambda r: _col(r, "a1"),                               _ms),
        ("alpha_0/alpha_1 ($M)", lambda r: _col(r, "a0") / _col(r, "a1") * 1e3,         _ms),
        ("IV R2",                lambda r: _col(r, "r2"),                               lambda m,s: f"{m:.4f}"),
    ]:
        cells = []
        for s in specs:
            v = fn(results[s][0])
            cells.append(fmt(v.mean(), v.std()) if fmt is _ms else fmt(v.mean(), v.std()))
        _row(label, cells)
    _row("N draws", [str(len(results[s][0])) for s in specs])
    print()


def table_surplus(results):
    specs = list(results.keys())
    a1 = {s: _col(r, "a1") for s, (r, _) in results.items()}
    show_controls = any(_col(results[s][0], "controls_part").any() for s in specs)

    def _div(key, s):
        return _col(results[s][0], key) / a1[s]

    print("  TABLE 3: SURPLUS DECOMPOSITION  ($B = utils / alpha_1)")
    _sep(len(specs))
    _header(specs)

    _row("Total surplus", [_ms(v.mean(), v.std(), 4) for v in (_div("surplus", s) for s in specs)])
    _sep(len(specs), "-")

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

    _row("  delta",         [_ms(v.mean(), v.std(), 4) for v in (_div("fe_total",    s) for s in specs)])
    _row("    n*a0/a1",     [_ms(v.mean(), v.std(), 4) for v in (_div("a0_part",     s) for s in specs)])
    _row("    -revenue",    [_ms((-v).mean(), v.std(), 4) for v in (_div("price_part", s) for s in specs)])
    if show_controls:
        _row("    controls/a1", [_ms(v.mean(), v.std(), 4) for v in (_div("controls_part", s) for s in specs)])
    _row("    sum xi/a1",   [_ms(v.mean(), v.std(), 4) for v in (_div("xi_part",     s) for s in specs)])
    _sep(len(specs), "-")
    _row("Entropy of choice", [_ms(v.mean(), v.std(), 4) for v in (_div("entropy", s) for s in specs)])
    print()


def run(specs):
    results = decompose_all(specs)
    if not results:
        print("No results to display.")
        return
    table_first_stage(results)
    table_iv(results)
    table_surplus(results)
