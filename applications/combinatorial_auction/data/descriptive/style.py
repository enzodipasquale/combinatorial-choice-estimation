import matplotlib
matplotlib.use("Agg")

NAVY, GOLD, SLATE = "#1B2A4A", "#B8860B", "#4A6274"
DPI = 130


def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def pop_formatter(x, _):
    if x >= 1e6:
        return f"{x / 1e6:.0f}M"
    if x >= 1e3:
        return f"{x / 1e3:.0f}K"
    return f"{x:.0f}"
