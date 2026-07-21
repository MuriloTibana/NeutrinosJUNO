
from pathlib import Path
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator


# ============================================================
# User settings
# ============================================================

DEFAULT_RESULTS_PATH = Path(
    "results/45_45_data_cnf2_all_systematics_fixed_NEW.npz"
)

DEFAULT_OUTPUT_PATH = Path(
    "img/figure2_cnf2_only.png"
)

SHOW_DATA_POINTS = False
SAVE_PDF = True

# Paper-like colors
CNF2_COLOR = "#E729A8"
CNF2_DARK = "#E729A8"
CONTOUR_COLOR = "#E729A8"
JUNO_COLOR = "black"

# Two-parameter confidence levels
CONTOUR_LEVELS = [2.30, 6.18, 11.83]


# ============================================================
# JUNO visual reference
# ============================================================

JUNO_BEST_SIN2_THETA12 = 0.3092
JUNO_SIGMA_SIN2_THETA12 = 0.0087

JUNO_BEST_DM21_1E5 = 7.50
JUNO_SIGMA_DM21_1E5 = 0.12

JUNO_CORRELATION = -0.23


def juno_reference_delta_chi2(theta, dm21_1e5):
    """
    Correlated-Gaussian JUNO reference used only for the
    black dashed comparison curves.
    """

    theta_standard = (
        theta - JUNO_BEST_SIN2_THETA12
    ) / JUNO_SIGMA_SIN2_THETA12

    dm_standard = (
        dm21_1e5 - JUNO_BEST_DM21_1E5
    ) / JUNO_SIGMA_DM21_1E5

    rho = JUNO_CORRELATION

    return (
        theta_standard**2
        - 2.0 * rho * theta_standard * dm_standard
        + dm_standard**2
    ) / (1.0 - rho**2)


# ============================================================
# Utility functions
# ============================================================

def get_array(results, *possible_names):
    for name in possible_names:
        if name in results:
            return results[name].copy()

    raise KeyError(
        "Could not find any of these arrays in the results file: "
        + ", ".join(possible_names)
    )


def get_scalar(results, *possible_names):
    array = get_array(results, *possible_names)
    return float(np.asarray(array).item())


def style_axis(axis):
    axis.tick_params(
        direction="in",
        top=True,
        right=True,
        which="both",
        length=4,
    )

    axis.tick_params(
        which="minor",
        length=2,
    )

    axis.xaxis.set_minor_locator(
        AutoMinorLocator()
    )

    axis.yaxis.set_minor_locator(
        AutoMinorLocator()
    )

    for spine in axis.spines.values():
        spine.set_linewidth(0.9)


def finite_profile_minimum(values, axis):
    finite_values = np.where(
        np.isfinite(values),
        values,
        np.inf,
    )

    profile = np.min(
        finite_values,
        axis=axis,
    )

    profile -= np.min(profile)

    return profile


def events_per_0p1_mev(counts_per_bin, bin_widths):
    """
    Convert counts in variable-width JUNO bins into the
    Figure-2 convention of events per 0.1 MeV.
    """

    counts_per_bin = np.asarray(
        counts_per_bin,
        dtype=float,
    )

    bin_widths = np.asarray(
        bin_widths,
        dtype=float,
    )

    return (
        counts_per_bin
        * 0.1
        / bin_widths
    )


# ============================================================
# Main plotting function
# ============================================================

def make_figure(results_path, output_path):
    if not results_path.exists():
        raise FileNotFoundError(
            f"Results file not found: {results_path}\n"
            "Run the cnf2 fit first or pass the correct .npz path."
        )

    with np.load(
        results_path,
        allow_pickle=False,
    ) as results:

        sin2_theta12_grid = get_array(
            results,
            "sin2_theta12_grid",
            "cnf2_sin2_theta12_grid",
        )

        dm21_grid = get_array(
            results,
            "dm21_grid",
            "cnf2_dm21_grid",
        )

        chi2_grid = get_array(
            results,
            "chi2_grid",
            "cnf2_chi2_grid",
        )

        delta_chi2_grid = get_array(
            results,
            "delta_chi2_grid",
            "dchi2_grid",
            "cnf2_dchi2_grid",
        )

        if "success_grid" in results:
            success_grid = results[
                "success_grid"
            ].copy()
        else:
            success_grid = np.isfinite(
                chi2_grid
            )

        best_sin2_theta12 = get_scalar(
            results,
            "best_sin2_theta12",
            "cnf2_best_sin2",
        )

        best_dm21 = get_scalar(
            results,
            "best_dm21",
            "cnf2_best_dm21",
        )

        best_chi2 = get_scalar(
            results,
            "best_chi2",
            "cnf2_chi2_min",
        )

        JUNO_energy = get_array(
            results,
            "JUNO_energy",
            "x_model",
        )

        JUNO_total = get_array(
            results,
            "JUNO_total",
            "N_obs_total",
        )

        JUNO_reactor = get_array(
            results,
            "JUNO_reactor",
            "N_obs_reactor",
        )

        best_total_prediction = get_array(
            results,
            "best_total_prediction",
            "cnf2_N_best_scan",
        )

        best_reactor_prediction = get_array(
            results,
            "best_reactor_prediction",
            "cnf2_N_reactor_best_scan",
        )

        if "JUNO_data" in results:
            JUNO_data = results[
                "JUNO_data"
            ].copy()
        elif "observed_spectrum" in results:
            JUNO_data = results[
                "observed_spectrum"
            ].copy()
        else:
            JUNO_data = None

        if "JUNO_bin_widths" in results:
            JUNO_bin_widths = results[
                "JUNO_bin_widths"
            ].copy()
        else:
            # Uniform 0.1 MeV fallback for older result files
            JUNO_bin_widths = np.full_like(
                JUNO_energy,
                0.1,
                dtype=float,
            )

    # Validate the saved shapes
    expected_shape = JUNO_energy.shape

    spectrum_arrays = {
        "JUNO total": JUNO_total,
        "JUNO reactor": JUNO_reactor,
        "cnf2 total": best_total_prediction,
        "cnf2 reactor": best_reactor_prediction,
    }

    for name, spectrum in spectrum_arrays.items():
        if spectrum.shape != expected_shape:
            raise ValueError(
                f"{name} has shape {spectrum.shape}, "
                f"but JUNO_energy has shape {expected_shape}."
            )

    if JUNO_bin_widths.shape != expected_shape:
        raise ValueError(
            "JUNO_bin_widths and JUNO_energy do not have "
            "the same shape."
        )

    valid_grid = (
        success_grid
        & np.isfinite(chi2_grid)
        & np.isfinite(delta_chi2_grid)
    )

    if not np.any(valid_grid):
        raise RuntimeError(
            "The results file contains no valid scan points."
        )

    masked_delta_chi2 = np.ma.masked_where(
        ~valid_grid,
        delta_chi2_grid,
    )

    # One-dimensional profiled curves
    chi2_for_profiles = np.where(
        valid_grid,
        chi2_grid,
        np.inf,
    )

    theta_profile = finite_profile_minimum(
        chi2_for_profiles,
        axis=0,
    )

    dm21_profile = finite_profile_minimum(
        chi2_for_profiles,
        axis=1,
    )

    # Fine Gaussian JUNO comparison
    theta_fine = np.linspace(
        sin2_theta12_grid.min(),
        sin2_theta12_grid.max(),
        700,
    )

    dm21_fine_1e5 = np.linspace(
        dm21_grid.min() * 1.0e5,
        dm21_grid.max() * 1.0e5,
        700,
    )

    juno_fine_surface = juno_reference_delta_chi2(
        theta_fine[:, None],
        dm21_fine_1e5[None, :],
    )

    juno_theta_profile = np.min(
        juno_fine_surface,
        axis=1,
    )

    juno_dm21_profile = np.min(
        juno_fine_surface,
        axis=0,
    )

    theta_mesh, dm21_mesh_1e5 = np.meshgrid(
        sin2_theta12_grid,
        dm21_grid * 1.0e5,
    )

    juno_contour_surface = juno_reference_delta_chi2(
        theta_mesh,
        dm21_mesh_1e5,
    )

    # Convert the variable-width spectra only for plotting
    JUNO_total_plot = events_per_0p1_mev(
        JUNO_total,
        JUNO_bin_widths,
    )

    JUNO_reactor_plot = events_per_0p1_mev(
        JUNO_reactor,
        JUNO_bin_widths,
    )

    cnf2_total_plot = events_per_0p1_mev(
        best_total_prediction,
        JUNO_bin_widths,
    )

    cnf2_reactor_plot = events_per_0p1_mev(
        best_reactor_prediction,
        JUNO_bin_widths,
    )

    JUNO_data_plot = None

    if (
        SHOW_DATA_POINTS
        and JUNO_data is not None
        and JUNO_data.shape == expected_shape
    ):
        JUNO_data_plot = events_per_0p1_mev(
            JUNO_data,
            JUNO_bin_widths,
        )

    # ========================================================
    # Figure layout
    # ========================================================

    plt.rcParams.update({
        "font.size": 10,
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.linewidth": 0.9,
    })

    figure = plt.figure(
        figsize=(12.4, 6.8),
    )

    grid = figure.add_gridspec(
        nrows=2,
        ncols=3,
        width_ratios=[1.22, 0.56, 1.18],
        height_ratios=[0.52, 1.0],
        left=0.07,
        right=0.985,
        bottom=0.105,
        top=0.965,
        wspace=0.07,
        hspace=0.05,
    )

    axis_theta = figure.add_subplot(
        grid[0, 0]
    )

    axis_contour = figure.add_subplot(
        grid[1, 0]
    )

    axis_dm21 = figure.add_subplot(
        grid[1, 1],
        sharey=axis_contour,
    )

    axis_spectrum = figure.add_subplot(
        grid[:, 2]
    )

    # ========================================================
    # Top theta profile
    # ========================================================

    axis_theta.plot(
        sin2_theta12_grid,
        theta_profile,
        color=CNF2_COLOR,
        linewidth=1.9,
        label="cnf 2",
    )

    axis_theta.plot(
        theta_fine,
        juno_theta_profile,
        color=JUNO_COLOR,
        linestyle="--",
        linewidth=1.6,
        label="JUNO",
    )

    axis_theta.set_xlim(
        sin2_theta12_grid.min(),
        sin2_theta12_grid.max(),
    )

    theta_ymax = max(
        8.0,
        np.nanmax(
            theta_profile[
                np.isfinite(theta_profile)
            ]
        ) * 1.08,
    )

    axis_theta.set_ylim(
        0.0,
        theta_ymax,
    )

    axis_theta.set_ylabel(
        r"$\Delta\chi^2$"
    )

    axis_theta.tick_params(
        labelbottom=False,
    )

    axis_theta.legend(
        loc="upper right",
        fontsize=10,
        frameon=False,
    )

    style_axis(
        axis_theta
    )

    # ========================================================
    # Central contour panel
    # ========================================================

    available_levels = [
        level
        for level in CONTOUR_LEVELS
        if level <= np.nanmax(
            delta_chi2_grid[
                valid_grid
            ]
        )
    ]

    if available_levels:
        axis_contour.contour(
            theta_mesh,
            dm21_mesh_1e5,
            masked_delta_chi2,
            levels=available_levels,
            colors=CONTOUR_COLOR,
            linewidths=1.9,
        )

    axis_contour.contour(
        theta_mesh,
        dm21_mesh_1e5,
        juno_contour_surface,
        levels=CONTOUR_LEVELS,
        colors=JUNO_COLOR,
        linestyles="--",
        linewidths=1.5,
    )

    axis_contour.plot(
        best_sin2_theta12,
        best_dm21 * 1.0e5,
        marker="*",
        color=CNF2_DARK,
        linestyle="none",
        markersize=10,
        zorder=5,
    )

    axis_contour.set_xlim(
        sin2_theta12_grid.min(),
        sin2_theta12_grid.max(),
    )

    axis_contour.set_ylim(
        dm21_grid.min() * 1.0e5,
        dm21_grid.max() * 1.0e5,
    )

    axis_contour.set_xlabel(
        r"$\sin^2\theta_{12}$"
    )

    axis_contour.set_ylabel(
        r"$\Delta m^2_{21}\,[10^{-5}\,\mathrm{eV}^2]$"
    )

    style_axis(
        axis_contour
    )

    # ========================================================
    # Right-hand dm21 profile
    # ========================================================

    axis_dm21.plot(
        dm21_profile,
        dm21_grid * 1.0e5,
        color=CNF2_COLOR,
        linewidth=1.9,
    )

    axis_dm21.plot(
        juno_dm21_profile,
        dm21_fine_1e5,
        color=JUNO_COLOR,
        linestyle="--",
        linewidth=1.6,
    )

    dm_profile_xmax = max(
        8.0,
        np.nanmax(
            dm21_profile[
                np.isfinite(dm21_profile)
            ]
        ) * 1.08,
    )

    axis_dm21.set_xlim(
        0.0,
        dm_profile_xmax,
    )

    axis_dm21.set_xlabel(
        r"$\Delta\chi^2$"
    )

    axis_dm21.tick_params(
        labelleft=False,
    )

    style_axis(
        axis_dm21
    )

    # ========================================================
    # Spectrum panel
    # ========================================================

    # Every JUNO curve is black dashed
    axis_spectrum.step(
        JUNO_energy,
        JUNO_total_plot,
        where="mid",
        color=JUNO_COLOR,
        linestyle="--",
        linewidth=1.7,
        label="JUNO",
    )

    axis_spectrum.step(
        JUNO_energy,
        JUNO_reactor_plot,
        where="mid",
        color=JUNO_COLOR,
        linestyle="--",
        linewidth=1.35,
    )

    # cnf2 curves follow the muted teal style of Figure 2
    axis_spectrum.step(
        JUNO_energy,
        cnf2_total_plot,
        where="mid",
        color=CNF2_COLOR,
        linewidth=1.9,
        label="cnf 2",
    )

    axis_spectrum.step(
        JUNO_energy,
        cnf2_reactor_plot,
        where="mid",
        color=CNF2_DARK,
        linewidth=1.45,
    )

    if JUNO_data_plot is not None:
        axis_spectrum.plot(
            JUNO_energy,
            JUNO_data_plot,
            marker="o",
            linestyle="none",
            markersize=2.8,
            color="0.35",
            label="data",
        )

    axis_spectrum.set_xlim(
        JUNO_energy.min(),
        JUNO_energy.max(),
    )

    axis_spectrum.set_xlabel(
        r"$E_{\mathrm{pr}}\ [\mathrm{MeV}]$"
    )

    axis_spectrum.set_ylabel(
        "events per 0.1 MeV"
    )

    axis_spectrum.legend(
        loc="upper right",
        fontsize=10,
        frameon=False,
    )

    style_axis(
        axis_spectrum
    )

    # ========================================================
    # Figure legend for contour styles
    # ========================================================

    contour_legend = [
        Line2D(
            [0],
            [0],
            color=CONTOUR_COLOR,
            linewidth=1.9,
            label="cnf 2",
        ),
        Line2D(
            [0],
            [0],
            color=JUNO_COLOR,
            linestyle="--",
            linewidth=1.5,
            label="JUNO",
        ),
    ]

    axis_contour.legend(
        handles=contour_legend,
        loc="upper right",
        fontsize=9,
        frameon=False,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=260,
        bbox_inches="tight",
    )

    if SAVE_PDF:
        pdf_path = output_path.with_suffix(
            ".pdf"
        )

        figure.savefig(
            pdf_path,
            bbox_inches="tight",
        )

        print(
            f"PDF saved to {pdf_path}"
        )

    plt.show()

    print("\nFigure-2-style cnf2 plot")
    print("=" * 72)
    print(f"Results file       = {results_path}")
    print(f"PNG saved to       = {output_path}")
    print(f"Best sin²(theta12) = {best_sin2_theta12:.8f}")
    print(f"Best Delta m²21    = {best_dm21:.8e} eV²")
    print(f"Minimum chi²       = {best_chi2:.6f}")


# ============================================================
# Command-line interface
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Create a Figure-2-style plot for the cnf2 "
            "JUNO all-systematics result."
        )
    )

    parser.add_argument(
        "results_path",
        type=Path,
        nargs="?",
        default=DEFAULT_RESULTS_PATH,
        help=(
            "Path to the cnf2 .npz result. "
            f"Default: {DEFAULT_RESULTS_PATH}"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=(
            "Output PNG path. "
            f"Default: {DEFAULT_OUTPUT_PATH}"
        ),
    )

    arguments = parser.parse_args()

    make_figure(
        arguments.results_path,
        arguments.output,
    )
