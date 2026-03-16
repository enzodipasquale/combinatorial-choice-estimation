"""
Plot destination and firm fixed effects from the IPF revenue decomposition.

Reproduces:
  - Figure 9:  log(delta_j) vs distance from Mexico
  - Figure 10: histogram of log(delta_f)
"""

import sys
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.special import logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_data import (
    load_raw_data, filter_dataframe, build_revenue,
    build_all_gravity_features, _build_tensor_except,
)

# ── settings ──
COUNTRY = "MEX"
KEEP_TOP = 50


def compute_exp_rev_with_fes(y, max_iters=100):
    """Run IPF and return the log-scale fixed effects dict."""
    shape = y.shape
    n_axis = len(shape)
    FEs = {}
    log_marg_y = {}
    other = {}
    for axis, length in enumerate(shape):
        FEs[axis] = np.zeros(length)
        other_ax = tuple(i for i in range(n_axis) if i != axis)
        other[axis] = other_ax
        log_marg_y[axis] = np.log(y.sum(axis=other_ax))
    for it in range(max_iters):
        max_delta = 0.0
        for axis in range(n_axis):
            tensor_except = _build_tensor_except(FEs, shape, axis)
            updated = log_marg_y[axis] - logsumexp(
                tensor_except, axis=other[axis]
            )
            max_delta = max(
                max_delta, float(np.max(np.abs(updated - FEs[axis])))
            )
            FEs[axis] = updated
        if max_delta < 1e-12:
            break
    print(f"Converged in {it + 1} iterations (max_delta={max_delta:.3e})")
    return FEs  # 0: firm, 1: year, 2: destination


def main():
    dataframe = filter_dataframe(KEEP_TOP, load_raw_data(COUNTRY))
    revenue = build_revenue(dataframe) / 1e3  # thousands

    firms = dataframe["f"].unique()
    years = sorted(dataframe["y"].unique())
    destinations_raw = sorted(dataframe["d"].unique())

    FEs = compute_exp_rev_with_fes(revenue)
    log_delta_f = FEs[0]  # firm FEs (log scale)
    log_delta_t = FEs[1]  # year FEs (log scale)
    log_delta_j = FEs[2]  # destination FEs (log scale)

    # ── get distances from Mexico ──
    pairwise, _, home_to_dest, destinations = build_all_gravity_features(
        dataframe, home=COUNTRY
    )
    dist_km = home_to_dest["dist"]  # in km

    print(f"\nFirm FEs:        {len(log_delta_f)} values")
    print(f"Year FEs:        {len(log_delta_t)} values")
    print(f"Destination FEs: {len(log_delta_j)} values")
    print(f"Destinations:    {len(destinations)} (validated)")

    # destinations after validation may differ from destinations_raw
    # FEs[2] is indexed by destinations_raw; dist_km by validated destinations
    # Build mapping: validated dest -> index in destinations_raw
    dest_raw_map = {d: i for i, d in enumerate(destinations_raw)}
    valid_idx = []
    valid_dist = []
    valid_names = []
    for i, d in enumerate(destinations):
        if d in dest_raw_map:
            valid_idx.append(dest_raw_map[d])
            valid_dist.append(dist_km[i])
            valid_names.append(d)

    log_dj = log_delta_j[valid_idx]
    dists = np.array(valid_dist)

    # ── Figure 9: log(delta_j) vs distance ──
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            "Destination FE vs Distance from Mexico",
            "Distribution of Firm FEs",
        ],
        horizontal_spacing=0.12,
    )

    fig.add_trace(
        go.Scatter(
            x=dists,
            y=log_dj,
            mode="markers+text",
            text=valid_names,
            textposition="top center",
            textfont=dict(size=8),
            marker=dict(size=7, color="steelblue"),
            name="δ_j",
        ),
        row=1, col=1,
    )
    fig.update_xaxes(title_text="Distance from Mexico (km)", row=1, col=1)
    fig.update_yaxes(title_text="log δ_j", row=1, col=1)

    # ── Figure 10: histogram of log(delta_f) ──
    fig.add_trace(
        go.Histogram(
            x=log_delta_f,
            nbinsx=80,
            marker_color="steelblue",
            opacity=0.75,
            name="δ_f",
        ),
        row=1, col=2,
    )
    fig.update_xaxes(title_text="log δ_f", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=2)

    fig.update_layout(
        title="IPF Fixed Effects (revenue in thousands)",
        showlegend=False,
        width=1200,
        height=500,
    )

    out = Path(__file__).resolve().parent / "fe_plots.html"
    fig.write_html(str(out))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
