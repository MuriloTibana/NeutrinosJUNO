#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator


# ============================================================
# Paths
# ============================================================

NPZ_PATH = Path("cnf1_fit.npz")

PNG_PATH = Path("cnf1_solar_scan_paper_style.png")
PDF_PATH = Path("cnf1_solar_scan_paper_style.pdf")


# ============================================================
# JUNO reference parameters
#
# This is a correlated-Gaussian approximation used only for
# plotting the dashed JUNO reference contours.
# ============================================================

JUNO_BEST_SIN2_THETA12 = 0.3092
JUNO_SIGMA_SIN2_THETA12 = 0.0087

JUNO_BEST_DM21 = 7.50e-5
JUNO_SIGMA_DM21 = 0.12e-5

JUNO_CORRELATION = -0.23


# ============================================================
# Plot settings
# ============================================================

CNF1_COLOR = "#ec149d"
JUNO_COLOR = "black"

# 1 sigma, 2 sigma, and 3 sigma for two fitted parameters
CONTOUR_LEVELS = [
    2.30,
    6.18,
    11.83,
]

DM21_PLOT_SCALE = 1.0e5

PROFILE_CHI2_MAX = 9.0


# ============================================================
# Check input file
# ============================================================

if not NPZ_PATH.exists():
    raise FileNotFoundError(
        f"Could not find the fit file:\n"
        f"  {NPZ_PATH.resolve()}"
    )


# ============================================================
# Load cnf 1 solar scan
# ============================================================

required_keys = [
    "cnf1_sin2_theta12_grid",
    "cnf1_dm21_grid",
    "cnf1_dchi2_grid",
    "cnf1_best_sin2",
    "cnf1_best_dm21",
]

with np.load(
    NPZ_PATH,
    allow_pickle=True,
) as data:

    missing_keys = [
        key
        for key in required_keys
        if key not in data.files
    ]

    if missing_keys:
        raise KeyError(
            "The NPZ file is missing these arrays:\n"
            + "\n".join(
                f"  {key}"
                for key in missing_keys
            )
        )

    sin2_grid = np.asarray(
        data["cnf1_sin2_theta12_grid"],
        dtype=float,
    )

    dm21_grid = np.asarray(
        data["cnf1_dm21_grid"],
        dtype=float,
    )

    dchi2_cnf1 = np.asarray(
        data["cnf1_dchi2_grid"],
        dtype=float,
    )

    best_sin2 = float(
        data["cnf1_best_sin2"]
    )

    best_dm21 = float(
        data["cnf1_best_dm21"]
    )


# ============================================================
# Validate array shapes
# ============================================================

expected_shape = (
    len(dm21_grid),
    len(sin2_grid),
)

if dchi2_cnf1.shape != expected_shape:
    raise ValueError(
        "Unexpected Delta-chi-square grid shape.\n"
        f"Expected: {expected_shape}\n"
        f"Found:    {dchi2_cnf1.shape}"
    )


# Make sure the minimum is exactly zero
dchi2_cnf1 = (
    dchi2_cnf1
    - np.nanmin(dchi2_cnf1)
)


# ============================================================
# Construct dashed JUNO reference
# ============================================================

sin2_juno_grid = np.linspace(
    sin2_grid.min(),
    sin2_grid.max(),
    401,
)

dm21_juno_grid = np.linspace(
    dm21_grid.min(),
    dm21_grid.max(),
    401,
)

sin2_juno_mesh, dm21_juno_mesh = np.meshgrid(
    sin2_juno_grid,
    dm21_juno_grid,
)

sin2_standardized = (
    sin2_juno_mesh
    - JUNO_BEST_SIN2_THETA12
) / JUNO_SIGMA_SIN2_THETA12

dm21_standardized = (
    dm21_juno_mesh
    - JUNO_BEST_DM21
) / JUNO_SIGMA_DM21

dchi2_juno = (
    sin2_standardized**2
    - 2.0
    * JUNO_CORRELATION
    * sin2_standardized
    * dm21_standardized
    + dm21_standardized**2
) / (
    1.0
    - JUNO_CORRELATION**2
)

dchi2_juno = (
    dchi2_juno
    - np.nanmin(dchi2_juno)
)


# ============================================================
# Profile over the other oscillation parameter
#
# For sin²(theta12):
#   minimize over dm21
#
# For dm21:
#   minimize over sin²(theta12)
# ============================================================

profile_sin2_cnf1 = np.nanmin(
    dchi2_cnf1,
    axis=0,
)

profile_dm21_cnf1 = np.nanmin(
    dchi2_cnf1,
    axis=1,
)

profile_sin2_juno = np.nanmin(
    dchi2_juno,
    axis=0,
)

profile_dm21_juno = np.nanmin(
    dchi2_juno,
    axis=1,
)


# Set each profile minimum to zero
profile_sin2_cnf1 -= np.nanmin(
    profile_sin2_cnf1
)

profile_dm21_cnf1 -= np.nanmin(
    profile_dm21_cnf1
)

profile_sin2_juno -= np.nanmin(
    profile_sin2_juno
)

profile_dm21_juno -= np.nanmin(
    profile_dm21_juno
)


# ============================================================
# Matplotlib appearance
# ============================================================

plt.rcParams.update({
    "font.size": 13,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.linewidth": 1.1,
})


def style_axis(ax) -> None:
    """
    Give each axis inward-facing major and minor ticks.
    """

    ax.tick_params(
        direction="in",
        which="both",
        top=True,
        right=True,
        length=5,
        width=1.0,
    )

    ax.tick_params(
        which="minor",
        length=2.8,
    )

    ax.xaxis.set_minor_locator(
        AutoMinorLocator()
    )

    ax.yaxis.set_minor_locator(
        AutoMinorLocator()
    )


# ============================================================
# Figure layout
# ============================================================

fig = plt.figure(
    figsize=(7.6, 7.2)
)

grid = fig.add_gridspec(
    nrows=2,
    ncols=2,
    height_ratios=[
        0.78,
        1.55,
    ],
    width_ratios=[
        1.65,
        0.82,
    ],
    left=0.13,
    right=0.98,
    bottom=0.11,
    top=0.96,
    hspace=0.05,
    wspace=0.04,
)

ax_top = fig.add_subplot(
    grid[0, 0]
)

ax_main = fig.add_subplot(
    grid[1, 0],
    sharex=ax_top,
)

ax_right = fig.add_subplot(
    grid[1, 1],
    sharey=ax_main,
)

ax_legend = fig.add_subplot(
    grid[0, 1]
)

ax_legend.axis("off")


# ============================================================
# Top panel
#
# Profiled Delta chi-square versus sin²(theta12)
# ============================================================

ax_top.plot(
    sin2_grid,
    profile_sin2_cnf1,
    color=CNF1_COLOR,
    linewidth=2.0,
)

ax_top.plot(
    sin2_juno_grid,
    profile_sin2_juno,
    color=JUNO_COLOR,
    linestyle="--",
    linewidth=1.8,
)

ax_top.set_ylabel(
    r"$\Delta\chi^2$"
)

ax_top.set_ylim(
    0.0,
    PROFILE_CHI2_MAX,
)

ax_top.set_yticks(
    [0, 2, 4, 6, 8]
)

plt.setp(
    ax_top.get_xticklabels(),
    visible=False,
)

style_axis(ax_top)


# ============================================================
# Main panel
#
# Solar-parameter confidence contours
# ============================================================

sin2_cnf1_mesh, dm21_cnf1_mesh = np.meshgrid(
    sin2_grid,
    dm21_grid * DM21_PLOT_SCALE,
)

sin2_juno_plot_mesh, dm21_juno_plot_mesh = np.meshgrid(
    sin2_juno_grid,
    dm21_juno_grid * DM21_PLOT_SCALE,
)


# cnf 1 contours
ax_main.contour(
    sin2_cnf1_mesh,
    dm21_cnf1_mesh,
    dchi2_cnf1,
    levels=CONTOUR_LEVELS,
    colors=CNF1_COLOR,
    linewidths=2.0,
)


# JUNO dashed contours
ax_main.contour(
    sin2_juno_plot_mesh,
    dm21_juno_plot_mesh,
    dchi2_juno,
    levels=CONTOUR_LEVELS,
    colors=JUNO_COLOR,
    linestyles="--",
    linewidths=1.8,
)


# Best-fit point
ax_main.scatter(
    best_sin2,
    best_dm21 * DM21_PLOT_SCALE,
    marker="*",
    s=80,
    color=CNF1_COLOR,
    edgecolor="black",
    linewidth=0.5,
    zorder=5,
)


ax_main.set_xlabel(
    r"$\sin^2\theta_{12}$"
)

ax_main.set_ylabel(
    r"$\Delta m^2_{21}"
    r"\,[10^{-5}\,\mathrm{eV}^2]$"
)

ax_main.set_xlim(
    sin2_grid.min(),
    sin2_grid.max(),
)

ax_main.set_ylim(
    dm21_grid.min() * DM21_PLOT_SCALE,
    dm21_grid.max() * DM21_PLOT_SCALE,
)

style_axis(ax_main)


# ============================================================
# Right panel
#
# Profiled Delta chi-square versus dm21
# ============================================================

ax_right.plot(
    profile_dm21_cnf1,
    dm21_grid * DM21_PLOT_SCALE,
    color=CNF1_COLOR,
    linewidth=2.0,
)

ax_right.plot(
    profile_dm21_juno,
    dm21_juno_grid * DM21_PLOT_SCALE,
    color=JUNO_COLOR,
    linestyle="--",
    linewidth=1.8,
)

ax_right.set_xlabel(
    r"$\Delta\chi^2$"
)

ax_right.set_xlim(
    0.0,
    PROFILE_CHI2_MAX,
)

ax_right.set_xticks(
    [0, 2, 4, 6, 8]
)

plt.setp(
    ax_right.get_yticklabels(),
    visible=False,
)

style_axis(ax_right)


# ============================================================
# Legend
# ============================================================

legend_handles = [
    Line2D(
        [0],
        [0],
        color=CNF1_COLOR,
        linewidth=2.0,
        label="cnf 1",
    ),
    Line2D(
        [0],
        [0],
        color=JUNO_COLOR,
        linestyle="--",
        linewidth=1.8,
        label="JUNO",
    ),
]

ax_legend.legend(
    handles=legend_handles,
    loc="center",
    frameon=True,
    fontsize=13,
)


# ============================================================
# Save figure
# ============================================================

fig.savefig(
    PNG_PATH,
    dpi=300,
    bbox_inches="tight",
)

# ============================================================
# Print numerical result
# ============================================================

print()
print("Best-fit solar oscillation parameters")
print("=" * 60)

print(
    f"sin²(theta12) = "
    f"{best_sin2:.8f}"
)

print(
    f"Delta m²21    = "
    f"{best_dm21:.8e} eV²"
)

print()
print("Saved figures")
print("=" * 60)
print(f"PNG: {PNG_PATH.resolve()}")
print(f"PDF: {PDF_PATH.resolve()}")


plt.show()