from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.fit import calculateRawSpectrum, chi2_CNP, chi2_poisson
from src.background import normalizeToTable, interpolateToBins


# ============================================================
# Configuration
# ============================================================

N_THETA12 = 10
N_DM21 = 10

FIT_JUNO = "data"          # "data", "total", or "reactor"
CHI2_MODEL = "poisson"     # "poisson" or "cnp"

EPSILON = 1.0e-12
LARGE_CHI2 = 1.0e30

SIN2_THETA12_REFERENCE = 0.309
DM21_REFERENCE = 7.53e-5

LIVE_DAYS = 59.1
BIN_WIDTH = 0.1

JUNO_PATH = "data/spect-fit.txt"
BG_PATH = "data/digitized_backgrounds.csv"

RESULTS_PATH = Path(
    f"results/{N_THETA12}_{N_DM21}_{FIT_JUNO}_{CHI2_MODEL}_TEST.npz"
)

RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

if FIT_JUNO not in {"data", "total", "reactor"}:
    raise ValueError("FIT_JUNO must be 'data', 'total', or 'reactor'.")

if CHI2_MODEL not in {"poisson", "cnp"}:
    raise ValueError("CHI2_MODEL must be 'poisson' or 'cnp'.")


# ============================================================
# JUNO spectra
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
JUNO_total = df_JUNO["reactor_background"].to_numpy(dtype=float)
JUNO_reactor = df_JUNO["reactor_signal"].to_numpy(dtype=float)
JUNO_background = JUNO_total - JUNO_reactor

JUNO_SPECTRA = {
    "data": JUNO_data,
    "total": JUNO_total,
    "reactor": JUNO_reactor,
}

JUNO_fit = JUNO_SPECTRA[FIT_JUNO].copy()


# ============================================================
# Statistical model
# ============================================================

CHI2_MODELS = {
    "cnp": chi2_CNP,
    "poisson": chi2_poisson,
}

chi2_function = CHI2_MODELS[CHI2_MODEL]


# ============================================================
# Nominal background spectra
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

BACKGROUND_NAMES = tuple(BACKGROUND_COMPONENTS)
N_BACKGROUND_PULLS = len(BACKGROUND_NAMES)

Total_background = np.sum(
    np.stack(list(BACKGROUND_COMPONENTS.values())),
    axis=0,
)


# ============================================================
# Put backgrounds directly on the JUNO fit grid
# ============================================================

BACKGROUND_COMPONENTS_FIT = {
    name: np.interp(
        JUNO_energy,
        E_prompt_bins,
        component,
        left=0.0,
        right=0.0,
    )
    for name, component in BACKGROUND_COMPONENTS.items()
}

TOTAL_BACKGROUND_FIT = np.sum(
    np.stack(list(BACKGROUND_COMPONENTS_FIT.values())),
    axis=0,
)


# ============================================================
# Reactor event-rate uncertainty
# ============================================================

REACTOR_RATE_UNCERTAINTIES = {
    "target_protons": 0.010,
    "reference_spectrum": 0.012,
    "thermal_power": 0.005,
    "fission_fraction": 0.006,
    "spent_nuclear_fuel": 0.003,
    "non_equilibrium": 0.002,
    "different_fission_fraction": 0.001,
}

SIGMA_REACTOR_RATE = np.sqrt(
    np.sum(
        np.asarray(
            list(REACTOR_RATE_UNCERTAINTIES.values()),
            dtype=float,
        ) ** 2
    )
)

print(
    "Combined reactor event-rate uncertainty: "
    f"{100.0 * SIGMA_REACTOR_RATE:.3f}%"
)


# ============================================================
# Background normalization uncertainties
# ============================================================

SIGMA_BACKGROUND_RATE = {
    "Li_He": 0.33,
    "geoneutrinos": 0.56,
    "world_reactors": 0.10,
    "bi_po": 0.56,
    "others": 1.00,
}

PULL_NAMES = np.asarray(
    ("reactor",) + BACKGROUND_NAMES
)

N_PULLS = len(PULL_NAMES)


# ============================================================
# Fixed reactor-spectrum normalization
# ============================================================

Epr_reference, raw_reference = calculateRawSpectrum(
    SIN2_THETA12_REFERENCE,
    DM21_REFERENCE,
)

raw_reference = np.asarray(raw_reference, dtype=float)

raw_reference_fit = np.interp(
    JUNO_energy,
    Epr_reference,
    raw_reference,
    left=0.0,
    right=0.0,
)

raw_reference_total = np.sum(raw_reference_fit)
JUNO_reactor_total = np.sum(JUNO_reactor)

if not np.isfinite(raw_reference_total) or raw_reference_total <= 0.0:
    raise ValueError(
        "The raw reference reactor spectrum has a "
        "nonpositive or nonfinite total."
    )

REFERENCE_NORMALIZATION = (
    JUNO_reactor_total
    / raw_reference_total
)

normalized_reference_fit = (
    REFERENCE_NORMALIZATION
    * raw_reference_fit
)

print("\nFixed reactor-spectrum normalization")
print("=" * 70)
print(f"Reference sin²(theta12)   = {SIN2_THETA12_REFERENCE:.8f}")
print(f"Reference Delta m²21      = {DM21_REFERENCE:.8e} eV²")
print(f"Raw reference total       = {raw_reference_total:.8e}")
print(f"JUNO reactor total        = {JUNO_reactor_total:.6f}")
print(f"Reference normalization   = {REFERENCE_NORMALIZATION:.8e}")
print(f"Normalized model total    = {normalized_reference_fit.sum():.6f}")


# ============================================================
# Pull bounds
# ============================================================

reactor_lower_bound = max(
    -5.0,
    -0.999 / SIGMA_REACTOR_RATE,
)

PULL_BOUNDS = [
    (reactor_lower_bound, 5.0)
]

for name in BACKGROUND_NAMES:
    sigma = SIGMA_BACKGROUND_RATE[name]

    lower_bound = max(
        -5.0,
        -0.999 / sigma,
    )

    PULL_BOUNDS.append(
        (lower_bound, 5.0)
    )


# ============================================================
# Oscillation scan
# ============================================================

sin2_theta12_grid = np.linspace(
    0.27,
    0.35,
    N_THETA12,
)

dm21_grid = np.linspace(
    7.0e-5,
    8.0e-5,
    N_DM21,
)

chi2_grid = np.full(
    (N_DM21, N_THETA12),
    np.nan,
    dtype=float,
)

pull_grid = np.full(
    (N_DM21, N_THETA12, N_PULLS),
    np.nan,
    dtype=float,
)

success_grid = np.zeros(
    (N_DM21, N_THETA12),
    dtype=bool,
)

total_points = N_THETA12 * N_DM21
completed_points = 0


# ============================================================
# Profile nuisance parameters
# ============================================================

for i_dm, dm21_test in enumerate(dm21_grid):

    for i_theta, sin2_theta12_test in enumerate(
        sin2_theta12_grid
    ):

        Epr_centers, raw_spectrum = calculateRawSpectrum(
            sin2_theta12_test,
            dm21_test,
        )

        raw_spectrum = np.asarray(
            raw_spectrum,
            dtype=float,
        )

        reactor_spectrum = (
            REFERENCE_NORMALIZATION
            * raw_spectrum
        )

        reactor_nominal_fit = np.interp(
            JUNO_energy,
            Epr_centers,
            reactor_spectrum,
            left=0.0,
            right=0.0,
        )

        def chi2_for_this_point(pulls):

            pulls = np.asarray(
                pulls,
                dtype=float,
            )

            z_reactor = pulls[0]
            z_backgrounds = pulls[1:]

            reactor_factor = (
                1.0
                + SIGMA_REACTOR_RATE
                * z_reactor
            )

            if reactor_factor <= 0.0:
                return LARGE_CHI2

            reactor_prediction = (
                reactor_factor
                * reactor_nominal_fit
            )

            background_prediction = np.zeros_like(
                JUNO_energy,
                dtype=float,
            )

            for name, z_background in zip(
                BACKGROUND_NAMES,
                z_backgrounds,
            ):

                background_factor = (
                    1.0
                    + SIGMA_BACKGROUND_RATE[name]
                    * z_background
                )

                if background_factor <= 0.0:
                    return LARGE_CHI2

                background_prediction += (
                    background_factor
                    * BACKGROUND_COMPONENTS_FIT[name]
                )

            if FIT_JUNO == "reactor":
                prediction_fit = reactor_prediction
            else:
                prediction_fit = (
                    reactor_prediction
                    + background_prediction
                )

            prediction_fit = np.clip(
                prediction_fit,
                EPSILON,
                None,
            )

            chi2_data = chi2_function(
                JUNO_fit,
                prediction_fit,
            )

            chi2_pull = np.sum(
                pulls ** 2
            )

            return chi2_data + chi2_pull

        # Warm start from the nearest successful point
        if (
            i_theta > 0
            and success_grid[i_dm, i_theta - 1]
        ):

            initial_pulls = pull_grid[
                i_dm,
                i_theta - 1,
            ].copy()

        elif (
            i_dm > 0
            and success_grid[i_dm - 1, i_theta]
        ):

            initial_pulls = pull_grid[
                i_dm - 1,
                i_theta,
            ].copy()

        else:

            initial_pulls = np.zeros(
                N_PULLS,
                dtype=float,
            )

        fit_result = minimize(
            chi2_for_this_point,
            x0=initial_pulls,
            method="L-BFGS-B",
            bounds=PULL_BOUNDS,
            options={
                "ftol": 1.0e-10,
                "gtol": 1.0e-7,
                "maxiter": 1000,
            },
        )

        # Retry failed or nonfinite fits from zero using Powell
        if (
            not fit_result.success
            or not np.isfinite(fit_result.fun)
        ):

            retry_result = minimize(
                chi2_for_this_point,
                x0=np.zeros(N_PULLS, dtype=float),
                method="Powell",
                bounds=PULL_BOUNDS,
                options={
                    "ftol": 1.0e-9,
                    "xtol": 1.0e-7,
                    "maxiter": 2000,
                },
            )

            if (
                np.isfinite(retry_result.fun)
                and (
                    not np.isfinite(fit_result.fun)
                    or retry_result.fun < fit_result.fun
                )
            ):

                fit_result = retry_result

        point_success = (
            fit_result.success
            and np.isfinite(fit_result.fun)
        )

        chi2_grid[i_dm, i_theta] = fit_result.fun
        pull_grid[i_dm, i_theta] = fit_result.x
        success_grid[i_dm, i_theta] = point_success

        completed_points += 1

        pull_text = " | ".join(
            f"z_{name}={value:+.6f}"
            for name, value in zip(
                PULL_NAMES,
                fit_result.x,
            )
        )

        print(
            f"Completed {completed_points}/{total_points} | "
            f"chi2={fit_result.fun:.6f} | "
            f"{pull_text} | "
            f"success={point_success}"
        )


# ============================================================
# Separate saved pull grids
# ============================================================

z_norm_grid = pull_grid[:, :, 0].copy()
z_background_grid = pull_grid[:, :, 1:].copy()


# ============================================================
# Best-fit grid point
# ============================================================

valid_grid = (
    np.isfinite(chi2_grid)
    & success_grid
)

if not np.any(valid_grid):
    raise RuntimeError(
        "No oscillation-grid point converged successfully."
    )

chi2_for_minimum = np.where(
    valid_grid,
    chi2_grid,
    np.inf,
)

best_flat_index = np.argmin(
    chi2_for_minimum
)

best_dm_index, best_theta_index = np.unravel_index(
    best_flat_index,
    chi2_grid.shape,
)

best_sin2_theta12 = sin2_theta12_grid[
    best_theta_index
]

best_dm21 = dm21_grid[
    best_dm_index
]

best_chi2 = chi2_grid[
    best_dm_index,
    best_theta_index,
]

best_pulls = pull_grid[
    best_dm_index,
    best_theta_index,
].copy()

best_z_reactor = best_pulls[0]
best_z_backgrounds = best_pulls[1:].copy()

best_reactor_factor = (
    1.0
    + SIGMA_REACTOR_RATE
    * best_z_reactor
)

delta_chi2_grid = (
    chi2_grid
    - best_chi2
)


# ============================================================
# Recompute best-fit spectra
# ============================================================

Epr_best, raw_spectrum_best = calculateRawSpectrum(
    best_sin2_theta12,
    best_dm21,
)

raw_spectrum_best = np.asarray(
    raw_spectrum_best,
    dtype=float,
)

spectrum_best = (
    REFERENCE_NORMALIZATION
    * raw_spectrum_best
)

best_reactor_nominal = np.interp(
    JUNO_energy,
    Epr_best,
    spectrum_best,
    left=0.0,
    right=0.0,
)

best_reactor_prediction = (
    best_reactor_factor
    * best_reactor_nominal
)

best_background_components = {}
best_background_prediction = np.zeros_like(
    JUNO_energy,
    dtype=float,
)

for name, z_background in zip(
    BACKGROUND_NAMES,
    best_z_backgrounds,
):

    background_factor = (
        1.0
        + SIGMA_BACKGROUND_RATE[name]
        * z_background
    )

    pulled_component = (
        background_factor
        * BACKGROUND_COMPONENTS_FIT[name]
    )

    best_background_components[name] = (
        pulled_component
    )

    best_background_prediction += (
        pulled_component
    )

best_background_nominal = (
    TOTAL_BACKGROUND_FIT.copy()
)

best_total_nominal = (
    best_reactor_nominal
    + best_background_nominal
)

best_total_prediction = (
    best_reactor_prediction
    + best_background_prediction
)

if FIT_JUNO == "reactor":
    best_fit_prediction = (
        best_reactor_prediction
    )
else:
    best_fit_prediction = (
        best_total_prediction
    )

best_chi2_data = chi2_function(
    JUNO_fit,
    np.clip(
        best_fit_prediction,
        EPSILON,
        None,
    ),
)

best_chi2_pull = np.sum(
    best_pulls ** 2
)


# ============================================================
# Print best-fit diagnostics
# ============================================================

print("\nBest-fit results")
print("=" * 75)
print(f"sin²(theta12)              = {best_sin2_theta12:.8f}")
print(f"Delta m²21                 = {best_dm21:.8e} eV²")
print(f"Minimum chi²               = {best_chi2:.6f}")
print(f"Data contribution          = {best_chi2_data:.6f}")
print(f"Pull contribution          = {best_chi2_pull:.6f}")
print(f"Successful fits            = {success_grid.sum()}/{success_grid.size}")

print("\nBest-fit nuisance pulls")
print("=" * 75)

for name, value in zip(
    PULL_NAMES,
    best_pulls,
):
    print(f"z_{name:20s} = {value:+.6f}")

print("\nApplied physical shifts")
print("=" * 75)

reactor_shift = (
    SIGMA_REACTOR_RATE
    * best_z_reactor
)

print(
    f"reactor                  = "
    f"{100.0 * reactor_shift:+.4f}%"
)

for name, value in zip(
    BACKGROUND_NAMES,
    best_z_backgrounds,
):

    shift = (
        SIGMA_BACKGROUND_RATE[name]
        * value
    )

    print(
        f"{name:24s} = "
        f"{100.0 * shift:+.4f}%"
    )

print("\nEvent totals on the JUNO grid")
print("=" * 75)
print(f"Observed spectrum          = {JUNO_fit.sum():.6f}")
print(f"JUNO reactor               = {JUNO_reactor.sum():.6f}")
print(f"JUNO background            = {JUNO_background.sum():.6f}")
print(f"Model reactor nominal      = {best_reactor_nominal.sum():.6f}")
print(f"Model reactor after pull   = {best_reactor_prediction.sum():.6f}")
print(f"Model background nominal   = {best_background_nominal.sum():.6f}")
print(f"Model background after pull= {best_background_prediction.sum():.6f}")
print(f"Model total after pulls    = {best_total_prediction.sum():.6f}")


# ============================================================
# Save results
# ============================================================

background_sigma_array = np.asarray(
    [
        SIGMA_BACKGROUND_RATE[name]
        for name in BACKGROUND_NAMES
    ],
    dtype=float,
)

background_nominal_components = np.stack(
    [
        BACKGROUND_COMPONENTS_FIT[name]
        for name in BACKGROUND_NAMES
    ]
)

background_best_components = np.stack(
    [
        best_background_components[name]
        for name in BACKGROUND_NAMES
    ]
)

np.savez_compressed(
    RESULTS_PATH,

    # Configuration
    fit_juno=np.asarray(FIT_JUNO),
    chi2_model=np.asarray(CHI2_MODEL),

    # Scan axes
    sin2_theta12_grid=sin2_theta12_grid,
    dm21_grid=dm21_grid,

    # Chi-square grids
    chi2_grid=chi2_grid,
    delta_chi2_grid=delta_chi2_grid,
    success_grid=success_grid,

    # Pull grids
    pull_names=PULL_NAMES,
    pull_grid=pull_grid,
    z_norm_grid=z_norm_grid,
    z_background_grid=z_background_grid,

    # Uncertainties
    sigma_reactor_rate=SIGMA_REACTOR_RATE,
    sigma_background_rate=background_sigma_array,
    background_names=np.asarray(BACKGROUND_NAMES),

    # Fixed normalization
    sin2_theta12_reference=SIN2_THETA12_REFERENCE,
    dm21_reference=DM21_REFERENCE,
    reference_normalization=REFERENCE_NORMALIZATION,

    # Best-fit parameters
    best_sin2_theta12=best_sin2_theta12,
    best_dm21=best_dm21,
    best_chi2=best_chi2,
    best_chi2_data=best_chi2_data,
    best_chi2_pull=best_chi2_pull,
    best_pulls=best_pulls,
    best_z_reactor=best_z_reactor,
    best_z_backgrounds=best_z_backgrounds,
    best_reactor_factor=best_reactor_factor,

    # Data grids
    JUNO_energy=JUNO_energy,
    observed_spectrum=JUNO_fit,
    JUNO_data=JUNO_data,
    JUNO_total=JUNO_total,
    JUNO_reactor=JUNO_reactor,
    JUNO_background=JUNO_background,
    E_prompt_bins=E_prompt_bins,

    # Best-fit reactor spectra
    Epr_best=Epr_best,
    raw_spectrum_best=raw_spectrum_best,
    spectrum_best=spectrum_best,
    best_reactor_nominal=best_reactor_nominal,
    best_reactor_prediction=best_reactor_prediction,

    # Best-fit backgrounds
    Total_background=Total_background,
    best_background_nominal=best_background_nominal,
    best_background_prediction=best_background_prediction,
    background_nominal_components=background_nominal_components,
    background_best_components=background_best_components,

    # Complete predictions
    best_total_nominal=best_total_nominal,
    best_total_prediction=best_total_prediction,
    best_fit_prediction=best_fit_prediction,
)

print(f"\nResults saved to {RESULTS_PATH}")