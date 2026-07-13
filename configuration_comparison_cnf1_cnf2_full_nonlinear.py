import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize
from scipy.special import erf

from src.readDayaBay import (
    read_total_flux,
    read_covariance_matrix,
    recast_covariance_matrix,
)
from src.deltaSplines import create_delta_basis
from src.continuous_flux import build_continuous_flux_model
from src.phiHuber import phi_huber_weighted
from src.crossSectionIBD import sigma_ibd
from src.oscillation import neutrino_oscillation
from src.background import normalizeToTable, interpolateToBins


# ============================================================
# User options
# ============================================================

# Scan region
SIN2_THETA12_MIN = 0.27
SIN2_THETA12_MAX = 0.35

DM21_MIN = 7.0e-5
DM21_MAX = 8.0e-5

# A full nonlinear nuisance fit is much slower than the former
# linearized-template fit. Start with 25 x 25 while testing.
# Increase to 41 x 41 or 51 x 51 for the final smooth contours.
N_THETA12 = 25
N_DM21 = 25

# Prompt-energy fitting range
FIT_ENERGY_MIN = 0.8
FIT_ENERGY_MAX = 8.0

# Reactor absolute normalization method:
#
#   "area" -> match total JUNO reactor events
#   "peak" -> match the maximum JUNO reactor bin
#
# This normalization is calculated once at the reference
# oscillation point and then held fixed during the scan.
REACTOR_NORMALIZATION_MODE = "area"

# Refine the best grid point using a continuous minimizer.
REFINE_BEST_FIT = True

# Numerical profiling options for the full nonlinear nuisance fit.
# Every scan point minimizes the nuisance parameters with L-BFGS-B.
PROFILE_MAXITER = 100
PROFILE_FTOL = 1.0e-8
PROFILE_GTOL = 1.0e-5
PULL_BOUND_ABS = 5.0

# The response matrix is recalculated at the current scale and
# resolution pulls. These small steps are used only to calculate
# their local numerical derivatives for L-BFGS-B.
RESPONSE_DERIVATIVE_STEP = 1.0e-3

# Output paths
FIG_PATH = Path("oscillation_fit_figure2_style_full_nonlinear.png")
RESULTS_PATH = Path("oscillation_fit_cnf1_cnf2_full_nonlinear.npz")

# Match the public JUNO unoscillated spectrum bin by bin.
# This is the tuning used in the independent reanalysis to absorb
# small differences in reactor composition, response, and power.
USE_BIN_BY_BIN_UNOSCILLATED_CORRECTION = True

# Official JUNO solar-parameter result.  The black dashed reference
# is drawn as a correlated Gaussian approximation to the published
# best fit and one-dimensional errors.
JUNO_BEST_SIN2_THETA12 = 0.3092
JUNO_SIGMA_SIN2_THETA12 = 0.0087
JUNO_BEST_DM21 = 7.50e-5
JUNO_SIGMA_DM21 = 0.12e-5
JUNO_CORRELATION = -0.23


# ============================================================
# Analysis configurations
# ============================================================

# Configurations corresponding to the table shown in the paper.
#
# cnf1:
#   r_BG = 1
#   r_nl = 1
#   r_res = 1
#   sigma_res = 5%
#   CNP chi-square
#
# cnf2:
#   r_BG = 1.15
#   r_nl = 1
#   r_res = 1
#   sigma_res = 5%
#   Poisson chi-square

CONFIGURATIONS = {
    "cnf1": {
        "r_bg": 1.00,
        "r_nl": 1.00,
        "r_res": 1.00,
        "sigma_resolution": 0.05,
        "chi2_type": "cnp",
        "plot_color": "deeppink",
    },
    "cnf2": {
        "r_bg": 1.15,
        "r_nl": 1.00,
        "r_res": 1.00,
        "sigma_resolution": 0.05,
        "chi2_type": "poisson",
        "plot_color": "teal",
    },
}


# ============================================================
# Input paths
# ============================================================

DYB_PATH = Path(
    "data/DYB_unfolded_spectra_tot_U235_Pu239.txt"
)

JUNO_PATH = Path(
    "data/spect-fit.txt"
)

NONLINEARITY_PATH = Path(
    "data/positron_nonlinearity.csv"
)

BG_PATH = Path(
    "data/digitized_backgrounds.csv"
)


# ============================================================
# Physical constants
# ============================================================

kg_to_MeV = 5.61e29

m_p = 1.6726219e-27 * kg_to_MeV
m_n = 1.6749275e-27 * kg_to_MeV
m_e = 9.1093837e-31 * kg_to_MeV

Delta = m_n - m_p


# ============================================================
# Fixed oscillation parameters
# ============================================================

sin2_theta13 = 0.02215
dm31 = 2.513e-3


# Reference point used only to establish C_norm
REFERENCE_SIN2_THETA12 = 0.308
REFERENCE_DM21 = 7.49e-5


# ============================================================
# Prompt and neutrino energy grids
# ============================================================

BIN_WIDTH = 0.1

Epr_edges = np.arange(
    0.0,
    10.0 + BIN_WIDTH,
    BIN_WIDTH,
)

Epr_centers = 0.5 * (
    Epr_edges[:-1]
    + Epr_edges[1:]
)

E_prompt_bins = Epr_centers.copy()

# Increase to 2000 after the code is working if desired.
E_nu = np.linspace(
    1.81,
    10.0,
    1400,
)


# ============================================================
# Nominal detector response
# ============================================================

res_a = 0.033
res_b = 0.01
res_c = 0.0

prompt_alpha = 1.0
prompt_beta = 0.0


# ============================================================
# Systematic uncertainties
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
        np.array(
            list(REACTOR_RATE_UNCERTAINTIES.values()),
            dtype=float,
        ) ** 2
    )
)


BACKGROUND_NORM_SIGMAS = {
    "Li_He": 0.33,
    "geoneutrinos": 0.56,
    "world_reactors": 0.10,
    "bi_po": 0.56,
    "others": 1.00,
}


# Li/He shape uncertainty:
#
# 20% at 1 MeV and linearly proportional to energy.
SIGMA_LIHE_SHAPE_AT_1MEV = 0.20

# Reconstructed-energy scale uncertainty
SIGMA_ENERGY_SCALE = 0.005


# ============================================================
# Reactor information
# ============================================================

reactors = [
    {
        "name": "Taishan-1",
        "P_GWth": 4.6,
        "L_km": 52.77,
    },
    {
        "name": "Taishan-2",
        "P_GWth": 4.6,
        "L_km": 52.64,
    },
    {
        "name": "Yangjiang-1",
        "P_GWth": 2.9,
        "L_km": 52.74,
    },
    {
        "name": "Yangjiang-2",
        "P_GWth": 2.9,
        "L_km": 52.82,
    },
    {
        "name": "Yangjiang-3",
        "P_GWth": 2.9,
        "L_km": 52.41,
    },
    {
        "name": "Yangjiang-4",
        "P_GWth": 2.9,
        "L_km": 52.49,
    },
    {
        "name": "Yangjiang-5",
        "P_GWth": 2.9,
        "L_km": 52.11,
    },
    {
        "name": "Yangjiang-6",
        "P_GWth": 2.9,
        "L_km": 52.19,
    },
    {
        "name": "DayaBay-effective",
        "P_GWth": 17.4,
        "L_km": 215.0,
    },
]

reactor_data = pd.DataFrame(
    reactors
)

reactor_data["w"] = (
    reactor_data["P_GWth"]
    / (
        4.0
        * np.pi
        * reactor_data["L_km"] ** 2
    )
)


# ============================================================
# Huber coefficients and fission fractions
# ============================================================

alpha = {
    "U235": np.array([
        4.367,
        -4.577,
        2.100,
        -5.294e-1,
        6.186e-2,
        -2.777e-3,
    ]),
    "Pu239": np.array([
        4.757,
        -5.392,
        2.563,
        -6.596e-1,
        7.820e-2,
        -3.536e-3,
    ]),
    "Pu241": np.array([
        2.990,
        -2.882,
        1.278,
        -3.343e-1,
        3.905e-2,
        -1.754e-3,
    ]),
}

frac = {
    "U235": 0.564,
    "U238": 0.076,
    "Pu239": 0.304,
    "Pu241": 0.056,
}


# ============================================================
# Utility: trapezoidal integration weights
# ============================================================

def trapezoid_weights(x):
    """
    Return weights such that

        integral f(x) dx approximately equals sum(w * f).
    """

    x = np.asarray(
        x,
        dtype=float,
    )

    weights = np.zeros_like(
        x
    )

    weights[1:-1] = 0.5 * (
        x[2:]
        - x[:-2]
    )

    weights[0] = 0.5 * (
        x[1]
        - x[0]
    )

    weights[-1] = 0.5 * (
        x[-1]
        - x[-2]
    )

    return weights


trap_weights = trapezoid_weights(
    E_nu
)


# ============================================================
# Load Daya Bay continuous flux model
# ============================================================

df_total = read_total_flux(
    DYB_PATH,
    "Total",
)

C_ij = read_covariance_matrix(
    DYB_PATH
)

Psi_ik = recast_covariance_matrix(
    C_ij
)

Phi0 = df_total[
    "Flux"
].to_numpy(dtype=float)

E_high = df_total[
    "E_high"
].to_numpy(dtype=float)

E_low = df_total[
    "E_low"
].to_numpy(dtype=float)

E_center = df_total[
    "E_center"
].to_numpy(dtype=float)

delta, splines, I = create_delta_basis(
    E_center
)

phi_cont, extras = build_continuous_flux_model(
    E_center,
    E_low,
    E_high,
    Phi0,
    Psi_ik,
    delta,
    phi_huber_weighted,
    sigma_ibd,
    frac,
    alpha,
    Delta,
    m_e,
    500,
)

nbin = int(
    extras["nbin"]
)

print(
    f"Number of Daya Bay flux modes: {nbin}"
)


# ============================================================
# Precompute continuous flux basis on E_nu
# ============================================================

zero_flux_pulls = np.zeros(
    nbin,
    dtype=float,
)

phi0_E = np.asarray(
    phi_cont(
        E_nu,
        zero_flux_pulls,
    ),
    dtype=float,
).ravel()

phi0_E = np.clip(
    phi0_E,
    0.0,
    None,
)


phi_basis_E = np.zeros(
    (
        len(E_nu),
        nbin,
    ),
    dtype=float,
)

for mode_index in range(nbin):

    unit_pull = np.zeros(
        nbin,
        dtype=float,
    )

    unit_pull[
        mode_index
    ] = 1.0

    phi_mode = np.asarray(
        phi_cont(
            E_nu,
            unit_pull,
        ),
        dtype=float,
    ).ravel()

    phi_basis_E[
        :,
        mode_index
    ] = (
        phi_mode
        - phi0_E
    )


# ============================================================
# IBD cross section
# ============================================================

sigma_IBD = sigma_ibd(
    E_nu,
    Delta,
    m_e,
)


# ============================================================
# Load detector nonlinearity
# ============================================================

def load_nonlinearity_curve(path):
    """
    Load the first two usable numerical columns as
    energy and nonlinearity factor.
    """

    df = pd.read_csv(
        path
    )

    numeric = df.apply(
        pd.to_numeric,
        errors="coerce",
    )

    usable_columns = [
        column
        for column in numeric.columns
        if numeric[column].notna().sum() > 1
    ]

    if len(usable_columns) < 2:

        df = pd.read_csv(
            path,
            header=None,
        )

        numeric = df.apply(
            pd.to_numeric,
            errors="coerce",
        )

        usable_columns = [
            column
            for column in numeric.columns
            if numeric[column].notna().sum() > 1
        ]

    if len(usable_columns) < 2:

        raise ValueError(
            "Could not identify two numerical columns "
            "in the nonlinearity file."
        )

    energy = numeric[
        usable_columns[0]
    ].to_numpy(dtype=float)

    factor = numeric[
        usable_columns[1]
    ].to_numpy(dtype=float)

    valid = (
        np.isfinite(energy)
        & np.isfinite(factor)
    )

    energy = energy[
        valid
    ]

    factor = factor[
        valid
    ]

    order = np.argsort(
        energy
    )

    energy = energy[
        order
    ]

    factor = factor[
        order
    ]

    energy, unique_indices = np.unique(
        energy,
        return_index=True,
    )

    factor = factor[
        unique_indices
    ]

    if len(energy) < 2:

        raise ValueError(
            "The nonlinearity file needs at least "
            "two valid data points."
        )

    return energy, factor


E_nl_points, F_nl_points = (
    load_nonlinearity_curve(
        NONLINEARITY_PATH
    )
)

F_nl_interpolator = PchipInterpolator(
    E_nl_points,
    F_nl_points,
    extrapolate=False,
)


def F_nl(E_prompt):
    """
    Evaluate the nominal nonlinearity correction.
    """

    E_prompt = np.asarray(
        E_prompt,
        dtype=float,
    )

    E_clipped = np.clip(
        E_prompt,
        E_nl_points[0],
        E_nl_points[-1],
    )

    return np.asarray(
        F_nl_interpolator(
            E_clipped
        ),
        dtype=float,
    )


# ============================================================
# Detector response
# ============================================================

def sigma_prompt(E_prompt):
    """
    Nominal prompt-energy resolution.
    """

    E_prompt = np.asarray(
        E_prompt,
        dtype=float,
    )

    E_safe = np.clip(
        E_prompt,
        1.0e-10,
        None,
    )

    return np.sqrt(
        res_a**2 * E_safe
        + res_b**2 * E_safe**2
        + res_c**2
    )


def compute_response_matrix(
    r_nl,
    r_res,
    sigma_resolution,
    xi_scale=0.0,
    xi_resolution=0.0,
):
    """
    Construct the prompt-energy response matrix.

    The energy-scale pull is

        1 + sigma_scale * xi_scale.

    The resolution pull is

        1 + sigma_resolution * xi_resolution.
    """

    E_visible = (
        E_nu
        - Delta
        + m_e
    )

    E_prompt_true = (
        prompt_alpha
        * E_visible
        + prompt_beta
    )

    nominal_nonlinearity = F_nl(
        E_prompt_true
    )

    # r_nl = 1 corresponds to the nominal nonlinearity.
    effective_nonlinearity = (
        1.0
        + r_nl
        * (
            nominal_nonlinearity
            - 1.0
        )
    )

    scale_factor = (
        1.0
        + SIGMA_ENERGY_SCALE
        * xi_scale
    )

    mu = (
        E_prompt_true
        * scale_factor
        * effective_nonlinearity
    )

    resolution_factor = (
        r_res
        * (
            1.0
            + sigma_resolution
            * xi_resolution
        )
    )

    if resolution_factor <= 0.0:

        raise ValueError(
            "Energy-resolution factor became nonpositive."
        )

    sigma_E = (
        resolution_factor
        * sigma_prompt(mu)
    )

    sigma_E = np.clip(
        sigma_E,
        1.0e-12,
        None,
    )

    lower_edges = (
        Epr_edges[:-1, None]
    )

    upper_edges = (
        Epr_edges[1:, None]
    )

    mu_matrix = mu[
        None,
        :
    ]

    sigma_matrix = sigma_E[
        None,
        :
    ]

    z_upper = (
        upper_edges
        - mu_matrix
    ) / (
        np.sqrt(2.0)
        * sigma_matrix
    )

    z_lower = (
        lower_edges
        - mu_matrix
    ) / (
        np.sqrt(2.0)
        * sigma_matrix
    )

    response = 0.5 * (
        erf(z_upper)
        - erf(z_lower)
    )

    response[
        :,
        E_visible <= 0.0
    ] = 0.0

    return response


# ============================================================
# Weighted survival probability
# ============================================================

def compute_weighted_survival(
    sin2_theta12,
    dm21,
):
    """
    Calculate the reactor-power and baseline weighted
    electron-antineutrino survival probability.
    """

    Pee_weighted = np.zeros_like(
        E_nu
    )

    for _, reactor in reactor_data.iterrows():

        L_km = float(
            reactor["L_km"]
        )

        reactor_weight = float(
            reactor["w"]
        )

        Pee_reactor = neutrino_oscillation(
            E_nu,
            L_km,
            sin2_theta12,
            sin2_theta13,
            dm21,
            dm31,
        )

        Pee_weighted += (
            reactor_weight
            * Pee_reactor
        )

    return Pee_weighted


# ============================================================
# Load JUNO reference spectrum and data
# ============================================================

df_JUNO = pd.read_csv(
    JUNO_PATH,
    sep=r"\s+",
    header=None,
)

df_JUNO.columns = [
    "energy",
    "reactor_signal",
    "reactor_background",
    "data",
    "unoscillated_signal",
]

juno_energy = df_JUNO[
    "energy"
].to_numpy(dtype=float)

juno_reactor_signal = df_JUNO[
    "reactor_signal"
].to_numpy(dtype=float)

# Official best-fit reactor + background spectrum.
juno_total_best_fit = df_JUNO[
    "reactor_background"
].to_numpy(dtype=float)

# Measured candidates.  This remains the observation vector used
# inside the statistical fit.
juno_data = df_JUNO[
    "data"
].to_numpy(dtype=float)

juno_unoscillated_signal = df_JUNO[
    "unoscillated_signal"
].to_numpy(dtype=float)

juno_order = np.argsort(juno_energy)

juno_energy = juno_energy[juno_order]
juno_reactor_signal = juno_reactor_signal[juno_order]
juno_total_best_fit = juno_total_best_fit[juno_order]
juno_data = juno_data[juno_order]
juno_unoscillated_signal = juno_unoscillated_signal[juno_order]

juno_reactor_on_grid = np.interp(
    E_prompt_bins,
    juno_energy,
    juno_reactor_signal,
    left=np.nan,
    right=np.nan,
)

juno_total_on_grid = np.interp(
    E_prompt_bins,
    juno_energy,
    juno_total_best_fit,
    left=np.nan,
    right=np.nan,
)

juno_unoscillated_on_grid = np.interp(
    E_prompt_bins,
    juno_energy,
    juno_unoscillated_signal,
    left=np.nan,
    right=np.nan,
)

observed_on_grid = np.interp(
    E_prompt_bins,
    juno_energy,
    juno_data,
    left=np.nan,
    right=np.nan,
)

fit_mask = (
    np.isfinite(observed_on_grid)
    & np.isfinite(juno_reactor_on_grid)
    & np.isfinite(juno_total_on_grid)
    & np.isfinite(juno_unoscillated_on_grid)
    & (E_prompt_bins >= FIT_ENERGY_MIN)
    & (E_prompt_bins <= FIT_ENERGY_MAX)
    & (observed_on_grid > 0.0)
)

if not np.any(fit_mask):
    raise ValueError(
        "No valid JUNO bins were found in the fitting range."
    )

# ============================================================
# Load and normalize background spectra
# ============================================================

LIVE_DAYS = 59.1

TABLE1_RATES_CPD = {
    "Li_He": 4.3,
    "geoneutrinos": 1.2,
    "world_reactors": 0.88,
    "bi_po": 0.18,
    "others": (
        0.04
        + 0.02
        + 0.05
        + 0.08
        + 4.9
    ),
}

TABLE1_TOTAL_EVENTS = {
    name: rate * LIVE_DAYS
    for name, rate
    in TABLE1_RATES_CPD.items()
}


df_raw = pd.read_csv(
    BG_PATH
)

df_raw.columns = [
    str(column).strip()
    for column in df_raw.columns
]

for column in df_raw.columns:

    df_raw[column] = pd.to_numeric(
        df_raw[column],
        errors="coerce",
    )

df_raw = df_raw.dropna(
    subset=["E_prompt"]
)

E_raw = df_raw[
    "E_prompt"
].to_numpy(dtype=float)


background_shapes = {
    "Li_He": interpolateToBins(
        E_raw,
        df_raw[
            "Li_He"
        ].to_numpy(dtype=float),
        E_prompt_bins,
    ),
    "geoneutrinos": interpolateToBins(
        E_raw,
        df_raw[
            "geoneutrinos"
        ].to_numpy(dtype=float),
        E_prompt_bins,
    ),
    # The independent reanalysis uses the unoscillated reactor
    # spectrum as the world-reactor shape because the long-baseline
    # oscillations are averaged out.
    "world_reactors": np.clip(
        np.nan_to_num(
            juno_unoscillated_on_grid,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ),
        0.0,
        None,
    ),
    "bi_po": interpolateToBins(
        E_raw,
        df_raw[
            "bi_po"
        ].to_numpy(dtype=float),
        E_prompt_bins,
    ),
    "others": interpolateToBins(
        E_raw,
        df_raw[
            "others"
        ].to_numpy(dtype=float),
        E_prompt_bins,
    ),
}


background_base = {}

for name, shape in background_shapes.items():

    background_base[name] = np.asarray(
        normalizeToTable(
            shape,
            name,
            BIN_WIDTH,
            TABLE1_TOTAL_EVENTS,
        ),
        dtype=float,
    )


# ============================================================
# Nuisance-parameter definitions
# ============================================================

NUISANCE_NAMES = [
    "reactor_normalization",
    "energy_scale",
    "energy_resolution",
    "Li_He_normalization",
    "Li_He_shape",
    "geoneutrino_normalization",
    "world_reactor_normalization",
    "bi_po_normalization",
    "other_background_normalization",
]

NUISANCE_NAMES += [
    f"flux_mode_{index + 1:02d}"
    for index in range(nbin)
]

N_NUISANCE = len(
    NUISANCE_NAMES
)

IDX_REACTOR_NORM = 0
IDX_ENERGY_SCALE = 1
IDX_ENERGY_RESOLUTION = 2
IDX_LIHE_NORM = 3
IDX_LIHE_SHAPE = 4
IDX_GEO_NORM = 5
IDX_WORLD_NORM = 6
IDX_BIPO_NORM = 7
IDX_OTHER_NORM = 8

FLUX_START = 9
FLUX_STOP = FLUX_START + nbin


# ============================================================
# Prepare each configuration
# ============================================================

# Only the nominal response matrix is cached for the reactor
# calibration. During the nuisance fit, the response matrix is
# rebuilt at the current energy-scale and resolution pulls.
configuration_cache = {}

for config_name, config in CONFIGURATIONS.items():

    R0 = compute_response_matrix(
        r_nl=config["r_nl"],
        r_res=config["r_res"],
        sigma_resolution=config["sigma_resolution"],
        xi_scale=0.0,
        xi_resolution=0.0,
    )

    backgrounds = {}

    for name, spectrum in background_base.items():

        # The fixed r_BG factor is applied to every background
        # except geoneutrinos, following the configuration table.
        if name == "geoneutrinos":
            backgrounds[name] = spectrum.copy()
        else:
            backgrounds[name] = config["r_bg"] * spectrum

    total_background = np.zeros_like(E_prompt_bins, dtype=float)

    for spectrum in backgrounds.values():
        total_background += spectrum

    configuration_cache[config_name] = {
        "R0": R0,
        "backgrounds": backgrounds,
        "total_background": total_background,
    }


# ============================================================
# Calibrate the reactor prediction to the JUNO unoscillated spectrum
# ============================================================

# With no oscillations, every reactor contributes only its geometric
# and thermal-power weight.
noosc_weighted_probability = float(
    np.sum(reactor_data["w"].to_numpy(dtype=float))
)

reference_unoscillated_kernel = (
    phi0_E
    * sigma_IBD
    * noosc_weighted_probability
    * trap_weights
)

reference_raw_unoscillated = (
    configuration_cache["cnf1"]["R0"]
    @ reference_unoscillated_kernel
)

normalization_mask = (
    fit_mask
    & np.isfinite(reference_raw_unoscillated)
    & (reference_raw_unoscillated > 0.0)
    & (juno_unoscillated_on_grid > 0.0)
)

if not np.any(normalization_mask):
    raise ValueError(
        "No common positive bins are available for reactor calibration."
    )

if REACTOR_NORMALIZATION_MODE == "area":
    C_norm = (
        np.sum(juno_unoscillated_on_grid[normalization_mask])
        / np.sum(reference_raw_unoscillated[normalization_mask])
    )

elif REACTOR_NORMALIZATION_MODE == "peak":
    C_norm = (
        np.max(juno_unoscillated_on_grid[normalization_mask])
        / np.max(reference_raw_unoscillated[normalization_mask])
    )

else:
    raise ValueError(
        "REACTOR_NORMALIZATION_MODE must be 'area' or 'peak'."
    )

scaled_reference_unoscillated = (
    C_norm * reference_raw_unoscillated
)

REACTOR_BIN_CORRECTION = np.ones_like(
    E_prompt_bins,
    dtype=float,
)

if USE_BIN_BY_BIN_UNOSCILLATED_CORRECTION:
    valid_correction = (
        np.isfinite(juno_unoscillated_on_grid)
        & np.isfinite(scaled_reference_unoscillated)
        & (juno_unoscillated_on_grid > 0.0)
        & (scaled_reference_unoscillated > 0.0)
    )

    REACTOR_BIN_CORRECTION[valid_correction] = (
        juno_unoscillated_on_grid[valid_correction]
        / scaled_reference_unoscillated[valid_correction]
    )

    # Do not let empty edge bins generate pathological correction
    # factors.  The correction inside the fit range should remain
    # close to one.
    REACTOR_BIN_CORRECTION = np.clip(
        REACTOR_BIN_CORRECTION,
        0.50,
        1.50,
    )

print(f"Fixed reactor normalization C_norm = {C_norm:.8e}")
print(
    "Bin-by-bin unoscillated correction in fit range: "
    f"{np.min(REACTOR_BIN_CORRECTION[fit_mask]):.4f} to "
    f"{np.max(REACTOR_BIN_CORRECTION[fit_mask]):.4f}"
)

# ============================================================
# Full nonlinear nuisance model
# ============================================================

# The Li/He shape uncertainty is 20% at 1 MeV and grows linearly
# with prompt energy. It is applied multiplicatively to Li/He.
LIHE_SHAPE_FRACTION = (
    SIGMA_LIHE_SHAPE_AT_1MEV
    * E_prompt_bins
    / 1.0
)


def positive_pull_lower_bound(sigma):
    """Lower pull bound that keeps 1 + sigma * xi positive."""

    if sigma <= 0.0:
        return -PULL_BOUND_ABS

    return max(
        -PULL_BOUND_ABS,
        (-1.0 + 1.0e-6) / sigma,
    )


# Build parameter-specific bounds. This keeps normalization and
# Li/He shape factors physical without clipping inside the fit.
PULL_BOUNDS = [
    (-PULL_BOUND_ABS, PULL_BOUND_ABS)
    for _ in range(N_NUISANCE)
]

PULL_BOUNDS[IDX_REACTOR_NORM] = (
    positive_pull_lower_bound(SIGMA_REACTOR_RATE),
    PULL_BOUND_ABS,
)

PULL_BOUNDS[IDX_LIHE_NORM] = (
    positive_pull_lower_bound(BACKGROUND_NORM_SIGMAS["Li_He"]),
    PULL_BOUND_ABS,
)

PULL_BOUNDS[IDX_GEO_NORM] = (
    positive_pull_lower_bound(BACKGROUND_NORM_SIGMAS["geoneutrinos"]),
    PULL_BOUND_ABS,
)

PULL_BOUNDS[IDX_WORLD_NORM] = (
    positive_pull_lower_bound(BACKGROUND_NORM_SIGMAS["world_reactors"]),
    PULL_BOUND_ABS,
)

PULL_BOUNDS[IDX_BIPO_NORM] = (
    positive_pull_lower_bound(BACKGROUND_NORM_SIGMAS["bi_po"]),
    PULL_BOUND_ABS,
)

PULL_BOUNDS[IDX_OTHER_NORM] = (
    positive_pull_lower_bound(BACKGROUND_NORM_SIGMAS["others"]),
    PULL_BOUND_ABS,
)

# Restrict the negative Li/He shape pull only over bins in which
# the nominal Li/He component is present.
lihe_reference = background_base["Li_He"]
lihe_active = (
    lihe_reference
    > 1.0e-12 * max(float(np.max(lihe_reference)), 1.0)
)

if np.any(lihe_active):
    maximum_lihe_shape_fraction = float(
        np.max(LIHE_SHAPE_FRACTION[lihe_active])
    )
else:
    maximum_lihe_shape_fraction = float(
        np.max(LIHE_SHAPE_FRACTION)
    )

if maximum_lihe_shape_fraction > 0.0:
    lihe_shape_lower = max(
        -PULL_BOUND_ABS,
        (-1.0 + 1.0e-6) / maximum_lihe_shape_fraction,
    )
else:
    lihe_shape_lower = -PULL_BOUND_ABS

PULL_BOUNDS[IDX_LIHE_SHAPE] = (
    lihe_shape_lower,
    PULL_BOUND_ABS,
)


def clip_to_pull_bounds(pulls):
    """Clip a warm start into the allowed nuisance region."""

    pulls = np.asarray(pulls, dtype=float).copy()

    for index, (lower, upper) in enumerate(PULL_BOUNDS):
        pulls[index] = np.clip(pulls[index], lower, upper)

    return pulls


def chi2_data_and_derivative(observed, predicted, chi2_type):
    """
    Return the statistical chi-square and its derivative with
    respect to the predicted number of events in every bin.
    """

    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    if np.any(predicted <= 0.0):
        return np.inf, np.full_like(predicted, np.nan)

    if chi2_type == "cnp":

        data_term = (
            predicted**2 / (3.0 * observed)
            - observed
            + 2.0 * observed**2 / (3.0 * predicted)
        )

        derivative = (
            2.0 * predicted / (3.0 * observed)
            - 2.0 * observed**2 / (3.0 * predicted**2)
        )

        return float(np.sum(data_term)), derivative

    if chi2_type == "poisson":

        data_term = (
            predicted
            - observed
            + observed * np.log(observed / predicted)
        )

        derivative = 2.0 * (1.0 - observed / predicted)

        return float(2.0 * np.sum(data_term)), derivative

    raise ValueError(
        "chi2_type must be 'cnp' or 'poisson'."
    )


def evaluate_nonlinear_prediction(
    sin2_theta12,
    dm21,
    config_name,
    pulls,
    need_jacobian=False,
):
    """
    Recalculate the complete prediction at the current nuisance
    parameters.

    Unlike the former linearized fit, this function rebuilds the
    detector response matrix at the current energy-scale and
    energy-resolution pulls and reevaluates the pulled reactor flux.
    """

    pulls = np.asarray(pulls, dtype=float)

    if pulls.size != N_NUISANCE:
        raise ValueError(
            f"Expected {N_NUISANCE} nuisance pulls, "
            f"received {pulls.size}."
        )

    config = CONFIGURATIONS[config_name]
    cache = configuration_cache[config_name]
    backgrounds = cache["backgrounds"]

    xi_reactor = pulls[IDX_REACTOR_NORM]
    xi_scale = pulls[IDX_ENERGY_SCALE]
    xi_resolution = pulls[IDX_ENERGY_RESOLUTION]
    xi_lihe_norm = pulls[IDX_LIHE_NORM]
    xi_lihe_shape = pulls[IDX_LIHE_SHAPE]
    xi_geo = pulls[IDX_GEO_NORM]
    xi_world = pulls[IDX_WORLD_NORM]
    xi_bipo = pulls[IDX_BIPO_NORM]
    xi_other = pulls[IDX_OTHER_NORM]
    xi_flux = pulls[FLUX_START:FLUX_STOP]

    survival = compute_weighted_survival(
        sin2_theta12,
        dm21,
    )

    common_kernel = (
        sigma_IBD
        * survival
        * trap_weights
    )

    # The Daya Bay flux construction is linear in its covariance
    # modes. Evaluating it at the current pull vector is exact.
    flux_raw = (
        phi0_E
        + phi_basis_E @ xi_flux
    )

    # Preserve the physical nonnegative flux. The Jacobian below
    # correctly sets the flux-mode derivative to zero in clipped bins.
    flux_positive = flux_raw > 0.0
    phi_E = np.clip(flux_raw, 0.0, None)

    response = compute_response_matrix(
        r_nl=config["r_nl"],
        r_res=config["r_res"],
        sigma_resolution=config["sigma_resolution"],
        xi_scale=xi_scale,
        xi_resolution=xi_resolution,
    )

    reactor_kernel = phi_E * common_kernel

    reactor_raw = response @ reactor_kernel

    reactor_before_rate_pull = (
        C_norm
        * REACTOR_BIN_CORRECTION
        * reactor_raw
    )

    reactor_rate_factor = (
        1.0
        + SIGMA_REACTOR_RATE * xi_reactor
    )

    reactor = (
        reactor_rate_factor
        * reactor_before_rate_pull
    )

    lihe_norm_factor = (
        1.0
        + BACKGROUND_NORM_SIGMAS["Li_He"]
        * xi_lihe_norm
    )

    lihe_shape_factor = (
        1.0
        + LIHE_SHAPE_FRACTION
        * xi_lihe_shape
    )

    geo_factor = (
        1.0
        + BACKGROUND_NORM_SIGMAS["geoneutrinos"]
        * xi_geo
    )

    world_factor = (
        1.0
        + BACKGROUND_NORM_SIGMAS["world_reactors"]
        * xi_world
    )

    bipo_factor = (
        1.0
        + BACKGROUND_NORM_SIGMAS["bi_po"]
        * xi_bipo
    )

    other_factor = (
        1.0
        + BACKGROUND_NORM_SIGMAS["others"]
        * xi_other
    )

    lihe = (
        backgrounds["Li_He"]
        * lihe_norm_factor
        * lihe_shape_factor
    )

    geoneutrinos = (
        backgrounds["geoneutrinos"]
        * geo_factor
    )

    world_reactors = (
        backgrounds["world_reactors"]
        * world_factor
    )

    bi_po = (
        backgrounds["bi_po"]
        * bipo_factor
    )

    others = (
        backgrounds["others"]
        * other_factor
    )

    total_background = (
        lihe
        + geoneutrinos
        + world_reactors
        + bi_po
        + others
    )

    prediction = reactor + total_background

    components = {
        "reactor": reactor,
        "reactor_before_rate_pull": reactor_before_rate_pull,
        "Li_He": lihe,
        "geoneutrinos": geoneutrinos,
        "world_reactors": world_reactors,
        "bi_po": bi_po,
        "others": others,
        "total_background": total_background,
    }

    if not need_jacobian:
        return prediction, components, None

    jacobian = np.zeros(
        (len(E_prompt_bins), N_NUISANCE),
        dtype=float,
    )

    # Reactor normalization derivative.
    jacobian[:, IDX_REACTOR_NORM] = (
        SIGMA_REACTOR_RATE
        * reactor_before_rate_pull
    )

    # Flux-mode derivatives are exact at the current response.
    effective_flux_basis = (
        phi_basis_E
        * flux_positive[:, None]
    )

    flux_kernel_matrix = (
        effective_flux_basis
        * common_kernel[:, None]
    )

    reactor_flux_derivatives = (
        C_norm
        * REACTOR_BIN_CORRECTION[:, None]
        * (response @ flux_kernel_matrix)
    )

    jacobian[:, FLUX_START:FLUX_STOP] = (
        reactor_rate_factor
        * reactor_flux_derivatives
    )

    # The full response is recalculated around the current scale
    # and resolution pulls. These are local derivatives, not fixed
    # templates around the nominal point.
    step = RESPONSE_DERIVATIVE_STEP

    response_scale_plus = compute_response_matrix(
        r_nl=config["r_nl"],
        r_res=config["r_res"],
        sigma_resolution=config["sigma_resolution"],
        xi_scale=xi_scale + step,
        xi_resolution=xi_resolution,
    )

    response_scale_minus = compute_response_matrix(
        r_nl=config["r_nl"],
        r_res=config["r_res"],
        sigma_resolution=config["sigma_resolution"],
        xi_scale=xi_scale - step,
        xi_resolution=xi_resolution,
    )

    derivative_response_scale = (
        response_scale_plus
        - response_scale_minus
    ) / (2.0 * step)

    jacobian[:, IDX_ENERGY_SCALE] = (
        reactor_rate_factor
        * C_norm
        * REACTOR_BIN_CORRECTION
        * (derivative_response_scale @ reactor_kernel)
    )

    response_resolution_plus = compute_response_matrix(
        r_nl=config["r_nl"],
        r_res=config["r_res"],
        sigma_resolution=config["sigma_resolution"],
        xi_scale=xi_scale,
        xi_resolution=xi_resolution + step,
    )

    response_resolution_minus = compute_response_matrix(
        r_nl=config["r_nl"],
        r_res=config["r_res"],
        sigma_resolution=config["sigma_resolution"],
        xi_scale=xi_scale,
        xi_resolution=xi_resolution - step,
    )

    derivative_response_resolution = (
        response_resolution_plus
        - response_resolution_minus
    ) / (2.0 * step)

    jacobian[:, IDX_ENERGY_RESOLUTION] = (
        reactor_rate_factor
        * C_norm
        * REACTOR_BIN_CORRECTION
        * (derivative_response_resolution @ reactor_kernel)
    )

    # Background normalization and Li/He shape derivatives.
    jacobian[:, IDX_LIHE_NORM] = (
        BACKGROUND_NORM_SIGMAS["Li_He"]
        * backgrounds["Li_He"]
        * lihe_shape_factor
    )

    jacobian[:, IDX_LIHE_SHAPE] = (
        LIHE_SHAPE_FRACTION
        * backgrounds["Li_He"]
        * lihe_norm_factor
    )

    jacobian[:, IDX_GEO_NORM] = (
        BACKGROUND_NORM_SIGMAS["geoneutrinos"]
        * backgrounds["geoneutrinos"]
    )

    jacobian[:, IDX_WORLD_NORM] = (
        BACKGROUND_NORM_SIGMAS["world_reactors"]
        * backgrounds["world_reactors"]
    )

    jacobian[:, IDX_BIPO_NORM] = (
        BACKGROUND_NORM_SIGMAS["bi_po"]
        * backgrounds["bi_po"]
    )

    jacobian[:, IDX_OTHER_NORM] = (
        BACKGROUND_NORM_SIGMAS["others"]
        * backgrounds["others"]
    )

    return prediction, components, jacobian


def nonlinear_chi2_and_gradient(
    pulls,
    sin2_theta12,
    dm21,
    config_name,
):
    """Full nonlinear profiled objective for L-BFGS-B."""

    prediction, _, jacobian = evaluate_nonlinear_prediction(
        sin2_theta12=sin2_theta12,
        dm21=dm21,
        config_name=config_name,
        pulls=pulls,
        need_jacobian=True,
    )

    prediction_fit = prediction[fit_mask]
    jacobian_fit = jacobian[fit_mask, :]
    observed_fit = observed_on_grid[fit_mask]

    chi2_data, derivative_data = chi2_data_and_derivative(
        observed=observed_fit,
        predicted=prediction_fit,
        chi2_type=CONFIGURATIONS[config_name]["chi2_type"],
    )

    if not np.isfinite(chi2_data):
        return 1.0e100, np.zeros(N_NUISANCE, dtype=float)

    pulls = np.asarray(pulls, dtype=float)

    chi2_pull = float(np.sum(pulls**2))

    gradient = (
        jacobian_fit.T @ derivative_data
        + 2.0 * pulls
    )

    return chi2_data + chi2_pull, gradient


def profile_nonlinear_pulls(
    sin2_theta12,
    dm21,
    config_name,
    initial_pulls=None,
):
    """
    Numerically profile all nuisance parameters at one fixed
    oscillation point using the full nonlinear prediction.
    """

    if initial_pulls is None:
        initial_pulls = np.zeros(N_NUISANCE, dtype=float)

    initial_pulls = clip_to_pull_bounds(initial_pulls)

    result = minimize(
        nonlinear_chi2_and_gradient,
        x0=initial_pulls,
        args=(
            sin2_theta12,
            dm21,
            config_name,
        ),
        method="L-BFGS-B",
        jac=True,
        bounds=PULL_BOUNDS,
        options={
            "maxiter": PROFILE_MAXITER,
            "ftol": PROFILE_FTOL,
            "gtol": PROFILE_GTOL,
            "maxls": 30,
        },
    )

    best_pulls = clip_to_pull_bounds(result.x)

    best_prediction, best_components, _ = (
        evaluate_nonlinear_prediction(
            sin2_theta12=sin2_theta12,
            dm21=dm21,
            config_name=config_name,
            pulls=best_pulls,
            need_jacobian=False,
        )
    )

    best_chi2, _ = nonlinear_chi2_and_gradient(
        best_pulls,
        sin2_theta12,
        dm21,
        config_name,
    )

    return (
        float(best_chi2),
        best_pulls,
        best_prediction,
        best_components,
        result,
    )


# ============================================================
# Oscillation scan grids
# ============================================================

sin2_theta12_grid = np.linspace(
    SIN2_THETA12_MIN,
    SIN2_THETA12_MAX,
    N_THETA12,
)

dm21_grid = np.linspace(
    DM21_MIN,
    DM21_MAX,
    N_DM21,
)


# ============================================================
# Scan one analysis configuration
# ============================================================

def scan_configuration(config_name):
    """
    Scan sin^2(theta12) and dm21 while numerically profiling the
    complete nonlinear nuisance model at every grid point.
    """

    config = CONFIGURATIONS[config_name]

    chi2_grid = np.full(
        (N_DM21, N_THETA12),
        np.nan,
        dtype=float,
    )

    pulls_grid = np.zeros(
        (N_DM21, N_THETA12, N_NUISANCE),
        dtype=float,
    )

    success_grid = np.zeros(
        (N_DM21, N_THETA12),
        dtype=bool,
    )

    print()
    print("=" * 76)
    print(f"Scanning configuration: {config_name}")
    print(f"Statistical definition: {config['chi2_type']}")
    print("Nuisance treatment: full nonlinear numerical profiling")
    print("=" * 76)

    best_warm_start = np.zeros(N_NUISANCE, dtype=float)

    for i_dm, dm21_test in enumerate(dm21_grid):

        # Snake through the grid, allowing neighboring points to
        # provide warm starts without changing the physical result.
        if i_dm % 2 == 0:
            theta_indices = list(range(N_THETA12))
        else:
            theta_indices = list(range(N_THETA12 - 1, -1, -1))

        previous_pulls = None

        for i_theta in theta_indices:

            sin2_test = float(sin2_theta12_grid[i_theta])

            if previous_pulls is not None:
                initial_pulls = previous_pulls
            elif i_dm > 0:
                initial_pulls = pulls_grid[i_dm - 1, i_theta, :]
            else:
                initial_pulls = best_warm_start

            (
                chi2,
                best_pulls,
                _,
                _,
                optimizer_result,
            ) = profile_nonlinear_pulls(
                sin2_theta12=sin2_test,
                dm21=float(dm21_test),
                config_name=config_name,
                initial_pulls=initial_pulls,
            )

            # If a warm-started fit fails badly, retry once from
            # zero pulls and retain whichever solution is better.
            if (
                not optimizer_result.success
                or not np.isfinite(chi2)
            ):
                retry = profile_nonlinear_pulls(
                    sin2_theta12=sin2_test,
                    dm21=float(dm21_test),
                    config_name=config_name,
                    initial_pulls=np.zeros(N_NUISANCE),
                )

                if retry[0] < chi2 or not np.isfinite(chi2):
                    (
                        chi2,
                        best_pulls,
                        _,
                        _,
                        optimizer_result,
                    ) = retry

            chi2_grid[i_dm, i_theta] = chi2
            pulls_grid[i_dm, i_theta, :] = best_pulls
            success_grid[i_dm, i_theta] = optimizer_result.success

            previous_pulls = best_pulls.copy()

            if np.isfinite(chi2):
                current_minimum = np.nanmin(chi2_grid)
                if chi2 <= current_minimum:
                    best_warm_start = best_pulls.copy()

        row_success = int(np.sum(success_grid[i_dm, :]))

        print(
            f"Completed dm21 row {i_dm + 1:3d}/{N_DM21}: "
            f"{dm21_test:.6e} eV^2 | "
            f"successful profiles {row_success}/{N_THETA12}"
        )

    if not np.any(np.isfinite(chi2_grid)):
        raise RuntimeError(
            f"No finite profile fit was obtained for {config_name}."
        )

    best_grid_index = np.unravel_index(
        np.nanargmin(chi2_grid),
        chi2_grid.shape,
    )

    best_grid_dm_index = best_grid_index[0]
    best_grid_theta_index = best_grid_index[1]

    best_theta = float(
        sin2_theta12_grid[best_grid_theta_index]
    )

    best_dm21 = float(
        dm21_grid[best_grid_dm_index]
    )

    best_pulls = pulls_grid[
        best_grid_dm_index,
        best_grid_theta_index,
        :,
    ].copy()

    # --------------------------------------------------------
    # Optional continuous refinement in the two oscillation
    # parameters. Nuisance pulls remain numerically profiled.
    # --------------------------------------------------------

    if REFINE_BEST_FIT:

        dm_scale = 1.0e5
        fixed_refinement_start = best_pulls.copy()

        def profiled_physics_chi2(physics_parameters):

            theta_test = float(physics_parameters[0])
            dm_test = float(physics_parameters[1] / dm_scale)

            chi2, _, _, _, _ = profile_nonlinear_pulls(
                sin2_theta12=theta_test,
                dm21=dm_test,
                config_name=config_name,
                initial_pulls=fixed_refinement_start,
            )

            return chi2

        refinement = minimize(
            profiled_physics_chi2,
            x0=np.array([
                best_theta,
                best_dm21 * dm_scale,
            ]),
            method="L-BFGS-B",
            bounds=[
                (SIN2_THETA12_MIN, SIN2_THETA12_MAX),
                (DM21_MIN * dm_scale, DM21_MAX * dm_scale),
            ],
            options={
                "maxiter": 60,
                "ftol": 1.0e-9,
                "maxls": 20,
            },
        )

        best_theta = float(refinement.x[0])
        best_dm21 = float(refinement.x[1] / dm_scale)

        if not refinement.success:
            print(
                "Continuous oscillation refinement did not fully "
                "converge; its final finite point is retained."
            )

    # --------------------------------------------------------
    # Final nonlinear profile at the refined best-fit point
    # --------------------------------------------------------

    (
        best_chi2,
        best_pulls,
        best_prediction,
        best_components,
        best_optimizer_result,
    ) = profile_nonlinear_pulls(
        sin2_theta12=best_theta,
        dm21=best_dm21,
        config_name=config_name,
        initial_pulls=best_pulls,
    )

    # Lower histogram in the paper: reactor-only spectrum at the
    # best-fit oscillation point with every nuisance pull set to zero.
    zero_prediction, zero_components, _ = (
        evaluate_nonlinear_prediction(
            sin2_theta12=best_theta,
            dm21=best_dm21,
            config_name=config_name,
            pulls=np.zeros(N_NUISANCE),
            need_jacobian=False,
        )
    )

    best_reactor_no_pulls = zero_components["reactor"]

    delta_chi2_grid = chi2_grid - best_chi2
    delta_chi2_grid = np.maximum(delta_chi2_grid, 0.0)

    profile_theta12 = np.nanmin(
        delta_chi2_grid,
        axis=0,
    )

    profile_dm21 = np.nanmin(
        delta_chi2_grid,
        axis=1,
    )

    print()
    print(f"Best fit for {config_name}")
    print("-" * 76)
    print(f"sin^2(theta12) = {best_theta:.8f}")
    print(f"dm21            = {best_dm21:.8e} eV^2")
    print(f"chi2_min        = {best_chi2:.6f}")
    print(
        "nuisance optimizer success = "
        f"{best_optimizer_result.success}"
    )
    print(
        "nuisance optimizer message = "
        f"{best_optimizer_result.message}"
    )

    print("Largest fitted nuisance pulls:")

    pull_order = np.argsort(np.abs(best_pulls))[::-1]

    for pull_index in pull_order[:12]:
        print(
            f"  {NUISANCE_NAMES[pull_index]:32s} "
            f"{best_pulls[pull_index]:+9.4f}"
        )

    return {
        "config_name": config_name,
        "chi2_grid": chi2_grid,
        "delta_chi2_grid": delta_chi2_grid,
        "pulls_grid": pulls_grid,
        "success_grid": success_grid,
        "best_theta12": best_theta,
        "best_dm21": best_dm21,
        "best_chi2": best_chi2,
        "best_pulls": best_pulls,
        "best_prediction": best_prediction,
        "best_reactor_no_pulls": best_reactor_no_pulls,
        "best_components": best_components,
        "profile_theta12": profile_theta12,
        "profile_dm21": profile_dm21,
    }


# ============================================================
# Run cnf1 and cnf2 scans
# ============================================================

results = {}

for configuration_name in [
    "cnf1",
    "cnf2",
]:

    results[
        configuration_name
    ] = scan_configuration(
        configuration_name
    )


# ============================================================
# Save numerical results
# ============================================================

RESULTS_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

np.savez(
    RESULTS_PATH,
    sin2_theta12_grid=sin2_theta12_grid,
    dm21_grid=dm21_grid,

    chi2_cnf1=results[
        "cnf1"
    ]["chi2_grid"],

    delta_chi2_cnf1=results[
        "cnf1"
    ]["delta_chi2_grid"],

    best_theta12_cnf1=results[
        "cnf1"
    ]["best_theta12"],

    best_dm21_cnf1=results[
        "cnf1"
    ]["best_dm21"],

    best_chi2_cnf1=results[
        "cnf1"
    ]["best_chi2"],

    best_pulls_cnf1=results[
        "cnf1"
    ]["best_pulls"],

    best_prediction_cnf1=results[
        "cnf1"
    ]["best_prediction"],

    best_reactor_no_pulls_cnf1=results[
        "cnf1"
    ]["best_reactor_no_pulls"],

    success_grid_cnf1=results[
        "cnf1"
    ]["success_grid"],

    chi2_cnf2=results[
        "cnf2"
    ]["chi2_grid"],

    delta_chi2_cnf2=results[
        "cnf2"
    ]["delta_chi2_grid"],

    best_theta12_cnf2=results[
        "cnf2"
    ]["best_theta12"],

    best_dm21_cnf2=results[
        "cnf2"
    ]["best_dm21"],

    best_chi2_cnf2=results[
        "cnf2"
    ]["best_chi2"],

    best_pulls_cnf2=results[
        "cnf2"
    ]["best_pulls"],

    best_prediction_cnf2=results[
        "cnf2"
    ]["best_prediction"],

    best_reactor_no_pulls_cnf2=results[
        "cnf2"
    ]["best_reactor_no_pulls"],

    success_grid_cnf2=results[
        "cnf2"
    ]["success_grid"],

    fit_mode=np.array(
        "full_nonlinear_numerical_profiling",
        dtype=str,
    ),

    nuisance_names=np.array(
        NUISANCE_NAMES,
        dtype=str,
    ),

    energy=E_prompt_bins,
    observed=observed_on_grid,
)

print(
    f"\nSaved numerical results to: {RESULTS_PATH}"
)


# ============================================================
# Figure 2-style plot
# ============================================================

fig = plt.figure(
    figsize=(13.0, 6.8)
)

grid = fig.add_gridspec(
    nrows=2,
    ncols=4,
    width_ratios=[
        3.5,
        1.35,
        0.20,
        2.8,
    ],
    height_ratios=[
        1.25,
        3.2,
    ],
    hspace=0.08,
    wspace=0.15,
)


# Top profile in sin^2(theta12)
ax_theta_profile = fig.add_subplot(
    grid[0, 0]
)

# Main two-dimensional contour plot
ax_contour = fig.add_subplot(
    grid[1, 0]
)

# Right profile in dm21
ax_dm_profile = fig.add_subplot(
    grid[1, 1],
    sharey=ax_contour,
)

# Best-fit spectra
ax_spectrum_cnf1 = fig.add_subplot(
    grid[0, 3]
)

ax_spectrum_cnf2 = fig.add_subplot(
    grid[1, 3],
    sharex=ax_spectrum_cnf1,
)


# ============================================================
# Plot confidence contours and profiles
# ============================================================

# Two-parameter confidence levels:
#
# 1 sigma:  2.30
# 2 sigma:  6.18
# 3 sigma: 11.83
CONTOUR_LEVELS = [
    2.30,
    6.18,
    11.83,
]

LINE_STYLES = [
    "-",
    "--",
    ":",
]

# Correlated-Gaussian representation of the official JUNO result.
# This uses the published best fit and 1D errors.  It is intended as
# a faithful visual reference, not as a replacement for a released
# official two-dimensional likelihood grid.
THETA_MESH, DM21_MESH = np.meshgrid(
    sin2_theta12_grid,
    dm21_grid,
)

juno_x = (
    (THETA_MESH - JUNO_BEST_SIN2_THETA12)
    / JUNO_SIGMA_SIN2_THETA12
)

juno_y = (
    (DM21_MESH - JUNO_BEST_DM21)
    / JUNO_SIGMA_DM21
)

JUNO_DELTA_CHI2_GRID = (
    juno_x**2
    - 2.0 * JUNO_CORRELATION * juno_x * juno_y
    + juno_y**2
) / (1.0 - JUNO_CORRELATION**2)

JUNO_PROFILE_THETA12 = (
    (sin2_theta12_grid - JUNO_BEST_SIN2_THETA12)
    / JUNO_SIGMA_SIN2_THETA12
) ** 2

JUNO_PROFILE_DM21 = (
    (dm21_grid - JUNO_BEST_DM21)
    / JUNO_SIGMA_DM21
) ** 2


for config_name in [
    "cnf1",
    "cnf2",
]:

    config = CONFIGURATIONS[
        config_name
    ]

    result = results[
        config_name
    ]

    color = config[
        "plot_color"
    ]


    # --------------------------------------------------------
    # Top one-dimensional theta12 profile
    # --------------------------------------------------------

    ax_theta_profile.plot(
        sin2_theta12_grid,
        result[
            "profile_theta12"
        ],
        color=color,
        lw=2.0,
        label=config_name,
    )


    # --------------------------------------------------------
    # Two-dimensional contours
    # --------------------------------------------------------

    contour_set = ax_contour.contour(
        sin2_theta12_grid,
        dm21_grid * 1.0e5,
        result[
            "delta_chi2_grid"
        ],
        levels=CONTOUR_LEVELS,
        colors=[
            color,
        ],
        linestyles=LINE_STYLES,
        linewidths=1.7,
    )

    ax_contour.clabel(
        contour_set,
        inline=True,
        fontsize=8,
        fmt={
            2.30: r"$1\sigma$",
            6.18: r"$2\sigma$",
            11.83: r"$3\sigma$",
        },
    )

    ax_contour.scatter(
        result[
            "best_theta12"
        ],
        result[
            "best_dm21"
        ] * 1.0e5,
        marker="x",
        s=65,
        linewidths=2.0,
        color=color,
    )


    # --------------------------------------------------------
    # Right one-dimensional dm21 profile
    # --------------------------------------------------------

    ax_dm_profile.plot(
        result[
            "profile_dm21"
        ],
        dm21_grid * 1.0e5,
        color=color,
        lw=2.0,
    )


# ------------------------------------------------------------
# Official JUNO reference, shown as black dashed curves
# ------------------------------------------------------------

ax_theta_profile.plot(
    sin2_theta12_grid,
    JUNO_PROFILE_THETA12,
    color="black",
    linestyle="--",
    lw=1.8,
    label="JUNO",
)

ax_contour.contour(
    sin2_theta12_grid,
    dm21_grid * 1.0e5,
    JUNO_DELTA_CHI2_GRID,
    levels=CONTOUR_LEVELS,
    colors="black",
    linestyles="--",
    linewidths=1.6,
)

ax_dm_profile.plot(
    JUNO_PROFILE_DM21,
    dm21_grid * 1.0e5,
    color="black",
    linestyle="--",
    lw=1.8,
)

# ============================================================
# Format one-dimensional profile plots
# ============================================================

for level in [
    1.0,
    4.0,
    9.0,
]:

    ax_theta_profile.axhline(
        level,
        color="0.75",
        lw=0.8,
        zorder=0,
    )

    ax_dm_profile.axvline(
        level,
        color="0.75",
        lw=0.8,
        zorder=0,
    )


ax_theta_profile.set_ylabel(
    r"$\Delta\chi^2$"
)

ax_theta_profile.set_xlim(
    SIN2_THETA12_MIN,
    SIN2_THETA12_MAX,
)

ax_theta_profile.set_ylim(
    0.0,
    10.0,
)

ax_theta_profile.tick_params(
    labelbottom=False
)

ax_theta_profile.grid(
    alpha=0.25
)

ax_theta_profile.legend(
    frameon=False,
    loc="upper left",
)


ax_contour.set_xlabel(
    r"$\sin^2\theta_{12}$"
)

ax_contour.set_ylabel(
    r"$\Delta m_{21}^2\ [10^{-5}\ {\rm eV}^2]$"
)

ax_contour.set_xlim(
    SIN2_THETA12_MIN,
    SIN2_THETA12_MAX,
)

ax_contour.set_ylim(
    DM21_MIN * 1.0e5,
    DM21_MAX * 1.0e5,
)

ax_contour.grid(
    alpha=0.20
)


# Dummy curves for the contour legend
for config_name in [
    "cnf1",
    "cnf2",
]:

    ax_contour.plot(
        [],
        [],
        color=CONFIGURATIONS[
            config_name
        ]["plot_color"],
        lw=2.0,
        label=config_name,
    )

ax_contour.plot(
    [],
    [],
    color="black",
    linestyle="--",
    lw=1.8,
    label="JUNO",
)

ax_contour.legend(
    frameon=False,
    loc="upper right",
)


ax_dm_profile.set_xlabel(
    r"$\Delta\chi^2$"
)

ax_dm_profile.set_xlim(
    0.0,
    10.0,
)

ax_dm_profile.tick_params(
    labelleft=False
)

ax_dm_profile.grid(
    alpha=0.25
)


# ============================================================
# Plot best-fit spectra
# ============================================================

spectrum_axes = {
    "cnf1": ax_spectrum_cnf1,
    "cnf2": ax_spectrum_cnf2,
}

for config_name, axis in spectrum_axes.items():
    result = results[config_name]
    color = CONFIGURATIONS[config_name]["plot_color"]

    # Lower histograms: reactor-only spectra without pull shifts.
    axis.step(
        E_prompt_bins[fit_mask],
        result["best_reactor_no_pulls"][fit_mask],
        where="mid",
        color=color,
        lw=1.8,
        label=config_name,
    )

    axis.step(
        E_prompt_bins[fit_mask],
        juno_reactor_on_grid[fit_mask],
        where="mid",
        color="black",
        linestyle="--",
        lw=1.3,
        label="JUNO",
    )

    # Upper histograms: reactor + backgrounds after profiling pulls.
    axis.step(
        E_prompt_bins[fit_mask],
        result["best_prediction"][fit_mask],
        where="mid",
        color=color,
        lw=1.8,
    )

    axis.step(
        E_prompt_bins[fit_mask],
        juno_total_on_grid[fit_mask],
        where="mid",
        color="black",
        linestyle="--",
        lw=1.3,
    )

    axis.set_xlim(FIT_ENERGY_MIN, FIT_ENERGY_MAX)
    axis.set_ylabel("Events per 0.1 MeV")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, fontsize=9)

    axis.text(
        0.97,
        0.94,
        (
            rf"$\sin^2\theta_{{12}}"
            rf"={result['best_theta12']:.4f}$"
            "\n"
            rf"$\Delta m_{{21}}^2"
            rf"={result['best_dm21'] * 1.0e5:.3f}"
            rf"\times 10^{{-5}}\ \mathrm{{eV}}^2$"
        ),
        transform=axis.transAxes,
        horizontalalignment="right",
        verticalalignment="top",
        fontsize=8.5,
    )

ax_spectrum_cnf1.tick_params(labelbottom=False)
ax_spectrum_cnf2.set_xlabel(r"$E_{\rm pr}\ [{\rm MeV}]$")

# ============================================================
# Save and show figure
# ============================================================

FIG_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

plt.savefig(
    FIG_PATH,
    dpi=300,
    bbox_inches="tight",
)

print(
    f"Saved figure to: {FIG_PATH}"
)

plt.show()


# ============================================================
# Final summary
# ============================================================

print()
print(
    "Official JUNO reference: "
    f"sin^2(theta12) = {JUNO_BEST_SIN2_THETA12:.4f} +/- "
    f"{JUNO_SIGMA_SIN2_THETA12:.4f}, "
    f"dm21 = ({JUNO_BEST_DM21 * 1.0e5:.2f} +/- "
    f"{JUNO_SIGMA_DM21 * 1.0e5:.2f}) x 10^-5 eV^2"
)
print("=" * 78)
print("FINAL OSCILLATION FIT RESULTS")
print("=" * 78)

for config_name in [
    "cnf1",
    "cnf2",
]:

    result = results[
        config_name
    ]

    print(
        f"\n{config_name}"
    )

    print(
        f"sin^2(theta12) = "
        f"{result['best_theta12']:.8f}"
    )

    print(
        f"dm21            = "
        f"{result['best_dm21']:.8e} eV^2"
    )

    print(
        f"chi2_min        = "
        f"{result['best_chi2']:.6f}"
    )

print()
print("=" * 78)