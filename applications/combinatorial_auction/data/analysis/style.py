"""Shared plot style for auction descriptives."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAVY = "#1B2A4A"
GOLD = "#B8860B"
SLATE = "#4A6274"
RED = "#C0392B"

DPI = 200


def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def pop_formatter(x, _):
    if x >= 1e6:
        return f"{x / 1e6:.0f}M"
    if x >= 1e3:
        return f"{x / 1e3:.0f}K"
    return f"{x:.0f}"
