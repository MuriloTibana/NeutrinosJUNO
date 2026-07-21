import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.fit import calculateSpectrum
from src.background import normalizeToTable, interpolateToBins


# ============================================================
# Configuration
# ============================================================

N_THETA12 = 10
N_DM21 = 10

FIT_JUNO = "data"
CHI2_MODEL = "poisson"

RESULTS_PATH = f"results/{N_THETA12}_{N_DM21}_{FIT_JUNO}_{CHI2_MODEL}_TEST.npz"
JUNO_PATH = "data/spect-fit.txt"
BG_PATH = "data/digitized_backgrounds.csv"

LIVE_DAYS = 59.1
BIN_WIDTH = 0.1


# ============================================================
# Load fit results
# ============================================================

with np.load(RESULTS_PATH) as results:

    sin2_theta12_grid = results["sin2_theta12_grid"].copy()
    dm21_grid = results["dm21_grid"].copy()

    chi2_grid = results["chi2_grid"].copy()
    z_norm_grid = results["z_norm_grid"].copy()
    z_background_grid = results["z_background_grid"].copy()
    success_grid = results["success_grid"].copy()

    SIGMA_REACTOR_RATE = results["SIGMA_REACTOR_RATE"].item()
    Total_background_saved = results["Total_background"].copy()


delta_chi2_grid = chi2_grid - np.nanmin(chi2_grid)


# ============================================================
# Load JUNO spectra
# ============================================================

df_JUNO = pd.read_csv(JUNO_PATH, sep=r"\s+", header=None)

df_JUNO.columns = [
    "energy",
    "reactor_signal",
    "reactor_background",
    "data",
    "unoscillated_signal",
]

JUNO_energy = df_JUNO["energy"].to_numpy(dtype=float)
JUNO_data = df_JUNO["data"].to_numpy(dtype=float)

# Reactor-only published prediction
JUNO_reactor = df_JUNO["reactor_signal"].to_numpy(dtype=float)

# Published reactor-plus-background prediction
JUNO_total = df_JUNO["reactor_background"].to_numpy(dtype=float)

# Published background-only contribution
JUNO_background = JUNO_total - JUNO_reactor

JUNO_SPECTRA = {
    "data": JUNO_data,
    "total": JUNO_total,
    "reactor": JUNO_reactor,
}

JUNO_fit = JUNO_SPECTRA[FIT_JUNO]


# ============================================================
# Reconstruct nominal background components
# ============================================================

TABLE1_RATES_CPD = {
    "Li_He": 4.3,
    "geoneutrinos": 2.4,
    "world_reactors": 0.88,
    "bi_po": 0.18,
    "others": 0.04 + 0.02 + 0.05 + 0.08 + 4.9e-2,
}

TABLE1_TOTAL_EVENTS = {
    name: rate * LIVE_DAYS
    for name, rate in TABLE1_RATES_CPD.items()
}

df_raw = pd.read_csv(BG_PATH)
df_raw.columns = [str(column).strip() for column in df_raw.columns]

for column in df_raw.columns:
    df_raw[column] = pd.to_numeric(df_raw[column], errors="coerce")

df_raw = df_raw.dropna(subset=["E_prompt"])
E_raw = df_raw["E_prompt"].to_numpy(dtype=float)

E_edges = np.arange(0.0, 10.0 + BIN_WIDTH, BIN_WIDTH)
E_prompt_bins = 0.5 * (E_edges[:-1] + E_edges[1:])

Li_He_shape = interpolateToBins(
    E_raw,
    df_raw["Li_He"].to_numpy(dtype=float),
    E_prompt_bins,
)

geoneutrinos_shape = interpolateToBins(
    E_raw,
    df_raw["geoneutrinos"].to_numpy(dtype=float),
    E_prompt_bins,
)

world_reactors_shape = interpolateToBins(
    E_raw,
    df_raw["world_reactors_digitized"].to_numpy(dtype=float),
    E_prompt_bins,
)

bi_po_shape = interpolateToBins(
    E_raw,
    df_raw["bi_po"].to_numpy(dtype=float),
    E_prompt_bins,
)

others_shape = interpolateToBins(
    E_raw,
    df_raw["others"].to_numpy(dtype=float),
    E_prompt_bins,
)

Li_He = normalizeToTable(
    Li_He_shape,
    "Li_He",
    BIN_WIDTH,
    TABLE1_TOTAL_EVENTS,
)

geoneutrinos = normalizeToTable(
    geoneutrinos_shape,
    "geoneutrinos",
    BIN_WIDTH,
    TABLE1_TOTAL_EVENTS,
)

world_reactors = normalizeToTable(
    world_reactors_shape,
    "world_reactors",
    BIN_WIDTH,
    TABLE1_TOTAL_EVENTS,
)

bi_po = normalizeToTable(
    bi_po_shape,
    "bi_po",
    BIN_WIDTH,
    TABLE1_TOTAL_EVENTS,
)

others = normalizeToTable(
    others_shape,
    "others",
    BIN_WIDTH,
    TABLE1_TOTAL_EVENTS,
)

BACKGROUND_COMPONENTS = {
    "Li_He": Li_He,
    "geoneutrinos": geoneutrinos,
    "world_reactors": world_reactors,
    "bi_po": bi_po,
    "others": others,
}

SIGMA_BACKGROUND_RATE = {
    "Li_He": 0.33,
    "geoneutrinos": 0.56,
    "world_reactors": 0.10,
    "bi_po": 0.56,
    "others": 1.00,
}

BACKGROUND_NAMES = tuple(BACKGROUND_COMPONENTS)
PULL_NAMES = ("reactor",) + BACKGROUND_NAMES


# ============================================================
# Locate the best-fit grid point
# ============================================================

best_flat_index = np.nanargmin(chi2_grid)

best_dm_index, best_theta_index = np.unravel_index(
    best_flat_index,
    chi2_grid.shape,
)

best_sin2_theta12 = sin2_theta12_grid[best_theta_index]
best_dm21 = dm21_grid[best_dm_index]
best_chi2 = chi2_grid[best_dm_index, best_theta_index]

best_z_norm = z_norm_grid[best_dm_index, best_theta_index]

best_z_backgrounds = z_background_grid[
    best_dm_index,
    best_theta_index,
].copy()

best_pulls = np.concatenate((
    [best_z_norm],
    best_z_backgrounds,
))

best_reactor_factor = 1.0 + SIGMA_REACTOR_RATE * best_z_norm


# ============================================================
# Recompute the best-fit reactor spectrum
# ============================================================

Epr_best, spectrum_best = calculateSpectrum(
    best_sin2_theta12,
    best_dm21,
)

spectrum_best = np.asarray(spectrum_best, dtype=float)


# ============================================================
# Reconstruct backgrounds on the reactor-model grid
# ============================================================

background_nominal_components = {}
background_pulled_components = {}

best_background_nominal_grid = np.zeros_like(spectrum_best)
best_background_prediction_grid = np.zeros_like(spectrum_best)

for name, z_background in zip(BACKGROUND_NAMES, best_z_backgrounds):

    nominal_component = np.interp(
        Epr_best,
        E_prompt_bins,
        BACKGROUND_COMPONENTS[name],
        left=0.0,
        right=0.0,
    )

    background_factor = (
        1.0
        + SIGMA_BACKGROUND_RATE[name]
        * z_background
    )

    pulled_component = background_factor * nominal_component

    background_nominal_components[name] = nominal_component
    background_pulled_components[name] = pulled_component

    best_background_nominal_grid += nominal_component
    best_background_prediction_grid += pulled_component


# ============================================================
# Construct complete best-fit model
# ============================================================

best_reactor_nominal_grid = spectrum_best.copy()
best_reactor_prediction_grid = best_reactor_factor * spectrum_best

best_total_nominal_grid = (
    best_reactor_nominal_grid
    + best_background_nominal_grid
)

best_total_prediction_grid = (
    best_reactor_prediction_grid
    + best_background_prediction_grid
)


# Put all best-fit curves onto the JUNO energy grid
best_reactor_nominal = np.interp(
    JUNO_energy,
    Epr_best,
    best_reactor_nominal_grid,
    left=0.0,
    right=0.0,
)

best_reactor_prediction = np.interp(
    JUNO_energy,
    Epr_best,
    best_reactor_prediction_grid,
    left=0.0,
    right=0.0,
)

best_background_nominal = np.interp(
    JUNO_energy,
    Epr_best,
    best_background_nominal_grid,
    left=0.0,
    right=0.0,
)

best_background_prediction = np.interp(
    JUNO_energy,
    Epr_best,
    best_background_prediction_grid,
    left=0.0,
    right=0.0,
)

best_total_nominal = np.interp(
    JUNO_energy,
    Epr_best,
    best_total_nominal_grid,
    left=0.0,
    right=0.0,
)

best_total_prediction = np.interp(
    JUNO_energy,
    Epr_best,
    best_total_prediction_grid,
    left=0.0,
    right=0.0,
)


# ============================================================
# Print best-fit information
# ============================================================

print("\nBest-fit oscillation parameters")
print("=" * 70)
print(f"sin²(theta12) = {best_sin2_theta12:.8f}")
print(f"Delta m²21    = {best_dm21:.8e} eV²")
print(f"Minimum chi²  = {best_chi2:.6f}")

print("\nBest-fit nuisance pulls")
print("=" * 70)

for name, pull in zip(PULL_NAMES, best_pulls):
    print(f"z_{name:18s} = {pull:+.6f}")

print("\nApplied normalization shifts")
print("=" * 70)

reactor_shift = SIGMA_REACTOR_RATE * best_z_norm

print(
    f"reactor             = "
    f"{100.0 * reactor_shift:+.4f}%"
)

for name, pull in zip(BACKGROUND_NAMES, best_z_backgrounds):

    physical_shift = SIGMA_BACKGROUND_RATE[name] * pull

    print(
        f"{name:19s} = "
        f"{100.0 * physical_shift:+.4f}%"
    )

print("\nEvent totals on JUNO grid")
print("=" * 70)
print(f"Published JUNO reactor:      {JUNO_reactor.sum():.3f}")
print(f"Published JUNO background:   {JUNO_background.sum():.3f}")
print(f"Published JUNO total:        {JUNO_total.sum():.3f}")
print(f"Model reactor nominal:       {best_reactor_nominal.sum():.3f}")
print(f"Model reactor after pull:    {best_reactor_prediction.sum():.3f}")
print(f"Model background nominal:    {best_background_nominal.sum():.3f}")
print(f"Model background after pull: {best_background_prediction.sum():.3f}")
print(f"Model total after pulls:     {best_total_prediction.sum():.3f}")


# ============================================================
# Plot 1: Best-fit total spectrum
# ============================================================

fig, ax = plt.subplots(figsize=(9, 6))

ax.plot(
    JUNO_energy,
    JUNO_data,
    marker="o",
    linestyle="none",
    markersize=4,
    label="JUNO data",
)

ax.step(
    JUNO_energy,
    JUNO_total,
    where="mid",
    color="black",
    linestyle="--",
    linewidth=1.8,
    label="Published JUNO total",
)

ax.step(
    JUNO_energy,
    best_total_nominal,
    where="mid",
    linewidth=1.5,
    label="Model total before pulls",
)

ax.step(
    JUNO_energy,
    best_total_prediction,
    where="mid",
    linewidth=2.0,
    label="Model total after pulls",
)

ax.set_xlabel(
    r"Prompt energy $E_{\mathrm{prompt}}$ [MeV]"
)

ax.set_ylabel(
    "Events per 0.1 MeV"
)

ax.set_title(
    rf"$\sin^2\theta_{{12}}={best_sin2_theta12:.4f}$, "
    rf"$\Delta m^2_{{21}}={best_dm21:.3e}\ \mathrm{{eV}}^2$"
    "\n"
    rf"$\chi^2_{{\min}}={best_chi2:.3f}$, "
    rf"$z_R={best_z_norm:+.3f}$, "
    rf"$\delta_R={100.0 * reactor_shift:+.3f}\%$"
)

ax.set_xlim(
    JUNO_energy.min(),
    JUNO_energy.max(),
)

ax.grid(alpha=0.25)
ax.legend(fontsize=9)

fig.tight_layout()


# ============================================================
# Plot 2: Reactor and background separately
# ============================================================

fig, ax = plt.subplots(figsize=(9, 6))

ax.step(
    JUNO_energy,
    JUNO_reactor,
    where="mid",
    color="black",
    linestyle="--",
    linewidth=1.7,
    label="Published JUNO reactor",
)

ax.step(
    JUNO_energy,
    best_reactor_nominal,
    where="mid",
    linewidth=1.4,
    label="Model reactor nominal",
)

ax.step(
    JUNO_energy,
    best_reactor_prediction,
    where="mid",
    linewidth=2.0,
    label="Model reactor after pull",
)

ax.step(
    JUNO_energy,
    JUNO_background,
    where="mid",
    color="gray",
    linestyle="--",
    linewidth=1.7,
    label="Published JUNO background",
)

ax.step(
    JUNO_energy,
    best_background_prediction,
    where="mid",
    linewidth=2.0,
    label="Model background after pulls",
)

ax.set_xlabel(
    r"Prompt energy $E_{\mathrm{prompt}}$ [MeV]"
)

ax.set_ylabel(
    "Events per 0.1 MeV"
)

ax.set_title(
    "Reactor and background contributions"
)

ax.set_xlim(
    JUNO_energy.min(),
    JUNO_energy.max(),
)

ax.grid(alpha=0.25)
ax.legend(fontsize=9)

fig.tight_layout()


# ============================================================
# Plot 3: Best-fit nuisance pulls
# ============================================================

fig, ax = plt.subplots(figsize=(9, 5))

pull_positions = np.arange(len(PULL_NAMES))

ax.bar(
    pull_positions,
    best_pulls,
)

ax.axhline(
    0.0,
    color="black",
    linewidth=1.0,
)

ax.axhline(
    1.0,
    color="black",
    linestyle="--",
    linewidth=0.8,
)

ax.axhline(
    -1.0,
    color="black",
    linestyle="--",
    linewidth=0.8,
)

ax.set_xticks(pull_positions)

ax.set_xticklabels(
    PULL_NAMES,
    rotation=35,
    ha="right",
)

ax.set_ylabel(
    r"Best-fit pull $z_j$ [$\sigma$]"
)

ax.set_title(
    "Best-fit nuisance parameters"
)

ax.grid(
    axis="y",
    alpha=0.25,
)

fig.tight_layout()


# ============================================================
# Plot 4: Filled delta-chi-square map
# ============================================================

theta_mesh, dm21_mesh = np.meshgrid(
    sin2_theta12_grid,
    dm21_grid * 1.0e5,
)

invalid_points = (
    ~success_grid
    | ~np.isfinite(delta_chi2_grid)
)

delta_chi2_plot = np.ma.masked_where(
    invalid_points,
    delta_chi2_grid,
)

fig, ax = plt.subplots(figsize=(7.2, 6.8))

filled = ax.contourf(
    theta_mesh,
    dm21_mesh,
    delta_chi2_plot,
    levels=40,
)

fig.colorbar(
    filled,
    ax=ax,
    label=r"$\Delta\chi^2$",
)

ax.plot(
    best_sin2_theta12,
    best_dm21 * 1.0e5,
    marker="*",
    color="black",
    linestyle="none",
    markersize=11,
    label="Best fit",
)

ax.set_xlabel(
    r"$\sin^2\theta_{12}$"
)

ax.set_ylabel(
    r"$\Delta m^2_{21}\ [10^{-5}\,\mathrm{eV}^2]$"
)

ax.set_title(
    r"Profiled $\Delta\chi^2$"
)

ax.legend()
fig.tight_layout()


# ============================================================
# Plot 5: Confidence contours for two fitted parameters
# ============================================================

requested_levels = [
    2.30,
    6.18,
    11.83,
]

maximum_delta_chi2 = np.nanmax(
    delta_chi2_grid
)

contour_levels = [
    level
    for level in requested_levels
    if level <= maximum_delta_chi2
]

fig, ax = plt.subplots(figsize=(7.2, 6.8))

if contour_levels:

    cs = ax.contour(
        theta_mesh,
        dm21_mesh,
        delta_chi2_plot,
        levels=contour_levels,
        colors="deeppink",
        linewidths=1.8,
    )

    contour_labels = {
        2.30: r"$1\sigma$",
        6.18: r"$2\sigma$",
        11.83: r"$3\sigma$",
    }

    ax.clabel(
        cs,
        fmt=contour_labels,
        fontsize=10,
    )

else:

    print(
        "\nThe requested confidence levels are outside "
        f"the available range. Maximum Delta chi2 = "
        f"{maximum_delta_chi2:.6f}"
    )

ax.plot(
    best_sin2_theta12,
    best_dm21 * 1.0e5,
    marker="*",
    color="black",
    linestyle="none",
    markersize=11,
    label="Best fit",
)

ax.set_xlabel(
    r"$\sin^2\theta_{12}$"
)

ax.set_ylabel(
    r"$\Delta m^2_{21}\ [10^{-5}\,\mathrm{eV}^2]$"
)

ax.set_title(
    "Confidence regions, two parameters"
)

ax.legend()
fig.tight_layout()


# ============================================================
# Profile-ridge direction
# ============================================================

valid_dm_rows = np.any(
    np.isfinite(chi2_grid),
    axis=1,
)

dm21_ridge = dm21_grid[
    valid_dm_rows
]

chi2_ridge_rows = chi2_grid[
    valid_dm_rows
]

theta_min_at_each_dm = sin2_theta12_grid[
    np.nanargmin(
        chi2_ridge_rows,
        axis=1,
    )
]

print("\nProfile minimum at each Delta m²21")
print("=" * 70)

for dm21_value, theta_value in zip(
    dm21_ridge,
    theta_min_at_each_dm,
):

    print(
        f"{dm21_value * 1.0e5:.4f}  "
        f"{theta_value:.6f}"
    )

ridge_slope = np.polyfit(
    dm21_ridge * 1.0e5,
    theta_min_at_each_dm,
    1,
)[0]

print(
    f"\nRidge slope = {ridge_slope:+.8f}"
)


# ============================================================
# Plot 6: Ridge points
# ============================================================

fig, ax = plt.subplots(figsize=(7.2, 6.8))

ax.plot(
    theta_min_at_each_dm,
    dm21_ridge * 1.0e5,
    marker="o",
    linestyle="-",
    label="Profile minimum",
)

ax.plot(
    best_sin2_theta12,
    best_dm21 * 1.0e5,
    marker="*",
    color="black",
    linestyle="none",
    markersize=11,
    label="Global best fit",
)

ax.set_xlabel(
    r"$\sin^2\theta_{12}$"
)

ax.set_ylabel(
    r"$\Delta m^2_{21}\ [10^{-5}\,\mathrm{eV}^2]$"
)

ax.set_title(
    rf"Profile ridge, slope $={ridge_slope:+.5f}$"
)

ax.grid(alpha=0.25)
ax.legend()

fig.tight_layout()

plt.show()