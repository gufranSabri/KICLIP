# Shared figure style for the DCGD narrative: white background, no gridlines, visible spines + tick
# marks, no in-plot text/annotations (explanations go in a companion "how to read" .md, not the PNG).

from __future__ import annotations

import matplotlib as mpl
import matplotlib.patheffects as pe

BLUE, ORANGE, AQUA, VIOLET, MAGENTA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#d55181", "#898781"
INK, SUB = "#1f1f1f", "#5f5e5a"
HALO = [pe.Stroke(linewidth=3.2, foreground="white"), pe.Normal()]
W1, H1 = 5.2, 3.3


def apply_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Liberation Sans", "DejaVu Sans"],
        "font.size": 7.5, "axes.labelsize": 8.5, "axes.titlesize": 8.5,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
        "axes.facecolor": "white", "figure.facecolor": "white",
        "axes.edgecolor": INK, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False,
        "xtick.major.size": 3.5, "ytick.major.size": 3.5, "xtick.minor.size": 0, "ytick.minor.size": 0,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.major.pad": 4, "ytick.major.pad": 4, "xtick.color": INK, "ytick.color": INK,
        "axes.labelcolor": INK, "axes.labelpad": 5, "text.color": INK,
        "legend.frameon": True, "legend.fancybox": False, "legend.framealpha": 1.0, "legend.facecolor": "white",
        "legend.edgecolor": INK, "legend.borderpad": 0.6, "legend.handlelength": 1.8,
        "lines.linewidth": 1.7, "lines.solid_capstyle": "round",
        "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.04, "savefig.facecolor": "white",
    })


def marker_kw(c: str) -> dict:
    return dict(marker="o", markersize=4.6, markerfacecolor=c, markeredgecolor="white", markeredgewidth=0.9)


def legend(ax, **kw) -> None:
    ax.legend(**kw).get_frame().set_linewidth(0.8)
