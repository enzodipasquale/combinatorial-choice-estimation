"""pop90 vs winning bid across the 480 continental BTAs — levels and log-log."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from applications.combinatorial_auction.data.loaders import load_raw
from applications.combinatorial_auction.data.descriptive import OUT_FIG
from applications.combinatorial_auction.data.descriptive.style import (
    NAVY, SLATE, DPI, style_ax, pop_formatter,
)


def _ols(y, x):
    X = np.column_stack([np.ones_like(x), x])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    r2 = 1 - ((y - X @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return b, r2


def _dollar(x, _):
    a = abs(x)
    if a >= 1e9:  return f"${x/1e9:g}B"
    if a >= 1e6:  return f"${x/1e6:g}M"
    if a >= 1e3:  return f"${x/1e3:g}K"
    return f"${x:g}"


def plot_levels(raw):
    bta = raw["bta_data"]
    pop = bta["pop90"].values.astype(float)
    bid = bta["bid"].values.astype(float)
    b, r2 = _ols(pop, bid)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    xs = np.linspace(bid.min(), bid.max(), 100)
    ax.scatter(bid, pop, s=18, alpha=0.55, color=NAVY,
               edgecolor="white", linewidth=0.3)
    ax.plot(xs, b[0] + b[1] * xs, color=SLATE, lw=1.2, ls="--",
            label=fr"OLS: $\beta={b[1]:.4f}$, $R^2={r2:.3f}$")
    ax.set_xlabel("Winning bid", fontsize=9, family="serif")
    ax.set_ylabel("Population (1990)", fontsize=9, family="serif")
    ax.xaxis.set_major_formatter(FuncFormatter(_dollar))
    ax.yaxis.set_major_formatter(FuncFormatter(pop_formatter))
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    style_ax(ax)

    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig_pop_vs_bid_levels.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig_pop_vs_bid_levels: N={len(bta)}, beta={b[1]:.4f}, R^2={r2:.3f}")


def plot_loglog(raw):
    bta = raw["bta_data"]
    pop = bta["pop90"].values.astype(float)
    bid = bta["bid"].values.astype(float)
    lp, lb = np.log10(pop), np.log10(bid)
    b, r2 = _ols(lp, lb)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    xs = np.linspace(lb.min(), lb.max(), 100)
    ax.scatter(lb, lp, s=18, alpha=0.55, color=NAVY,
               edgecolor="white", linewidth=0.3)
    ax.plot(xs, b[0] + b[1] * xs, color=SLATE, lw=1.2, ls="--",
            label=fr"OLS: $\beta={b[1]:.3f}$, $R^2={r2:.3f}$")
    ax.set_xlabel(r"$\log_{10}$(winning bid)", fontsize=9, family="serif")
    ax.set_ylabel(r"$\log_{10}$(population)", fontsize=9, family="serif")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    style_ax(ax)

    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig_pop_vs_bid_loglog.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig_pop_vs_bid_loglog: N={len(bta)}, beta={b[1]:.3f}, R^2={r2:.3f}")


if __name__ == "__main__":
    raw = load_raw()
    plot_levels(raw)
    plot_loglog(raw)
