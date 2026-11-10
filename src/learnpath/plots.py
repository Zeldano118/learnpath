"""Shared chart style."""

import matplotlib.pyplot as plt

from . import config

# fixed order so a series keeps the same color in every chart
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e6e5e1", "#fcfcfb"


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 10, "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "text.color": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "legend.frameon": False, "legend.fontsize": 9,
        "font.size": 10, "figure.dpi": 110,
    })


def save(fig, name):
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES / f"{name}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    return path


def horizontal_grid(ax):
    # horizontal bars are read along x, so grid lines go vertical
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)


def label_bars(ax, bars, fmt="{:.0f}", horizontal=False, pad=3):
    for b in bars:
        v = b.get_width() if horizontal else b.get_height()
        if horizontal:
            ax.annotate(fmt.format(v), (v, b.get_y() + b.get_height() / 2), xytext=(pad, 0),
                        textcoords="offset points", va="center", fontsize=8, color=INK_2)
        else:
            ax.annotate(fmt.format(v), (b.get_x() + b.get_width() / 2, v), xytext=(0, pad),
                        textcoords="offset points", ha="center", fontsize=8, color=INK_2)
