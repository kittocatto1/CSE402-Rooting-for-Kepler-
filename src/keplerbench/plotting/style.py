"""Shared figure style, so all five of us produce consistent plots.

Owner: Anisa. Fully implemented - just import and use.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[3]
FIGURE_DIR = REPO_ROOT / "figures"

#: One fixed colour per method, used in every figure in the report.
SOLVER_COLORS = {
    "newton": "#4C72B0",
    "danby": "#DD8452",
    "markley": "#55A868",
    "nwm9": "#C44E52",
    "nwm11": "#8172B3",
}

#: One fixed line style per guess, so guess and solver are distinguishable
#: in the same plot without needing colour.
GUESS_STYLES = {
    "simple": ":",
    "canonical": "--",
    "radvel": "-.",
    "napier": "-",
}


def use_report_style() -> None:
    """Apply the shared rcParams. Call once at the top of a plotting script."""
    mpl.rcParams.update({
        "figure.figsize": (6.5, 4.0),
        "figure.dpi": 130,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    })


def save_figure(fig: plt.Figure, name: str, subdir: str = "") -> Path:
    """Save to figures/<subdir>/<name>.{pdf,png} and return the pdf path.

    PDF for the report (vector), PNG for the slides.
    """
    out_dir = FIGURE_DIR / subdir if subdir else FIGURE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{name}.pdf"
    fig.savefig(pdf)
    fig.savefig(out_dir / f"{name}.png")
    return pdf
