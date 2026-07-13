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

# Increase these for smoother contours.
# 41 x 41 is a reasonable starting point.
N_THETA12 = 40
N_DM21 = 40

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

# Refine the best grid point using a continuous minimizer
REFINE_BEST_FIT = True

# Maximum Newton iterations used to profile nuisance pulls
MAX_PULL_ITERATIONS = 35

# Numerical convergence tolerance
PULL_TOLERANCE = 1.0e-7

# Output paths
FIG_PATH = Path("oscillation_fit_figure2_style.png")
RESULTS_PATH = Path("oscillation_fit_cnf1_cnf2.npz")


# ============================================================
# JUNO reference curves
# ============================================================

# These values define a correlated-Gaussian approximation to the
# JUNO solar-parameter result. The resulting black dashed curves
# are a visual reference, not an exact released JUNO likelihood grid.
JUNO_BEST_SIN2_THETA12 = 0.3092
JUNO_SIGMA_SIN2_THETA12 = 0.0087

JUNO_BEST_DM21 = 7.50e-5
JUNO_SIGMA_DM21 = 0.12e-5

# Correlation controls the tilt of the 2D dashed contours.
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

juno_data = df_JUNO[
    "data"
].to_numpy(dtype=float)

juno_order = np.argsort(
    juno_energy
)

juno_energy = juno_energy[
    juno_order
]

juno_reactor_signal = juno_reactor_signal[
    juno_order
]

juno_data = juno_data[
    juno_order
]


juno_reactor_on_grid = np.interp(
    E_prompt_bins,
    juno_energy,
    juno_reactor_signal,
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
    np.isfinite(
        observed_on_grid
    )
    & np.isfinite(
        juno_reactor_on_grid
    )
    & (
        E_prompt_bins
        >= FIT_ENERGY_MIN
    )
    & (
        E_prompt_bins
        <= FIT_ENERGY_MAX
    )
    & (
        observed_on_grid
        > 0.0
    )
)

if not np.any(
    fit_mask
):

    raise ValueError(
        "No valid JUNO bins were found in the fitting range."
    )


# ============================================================
# Load and normalize background spectra
# ============================================================

LIVE_DAYS = 59.1

TABLE1_RATES_CPD = {
    "Li_He": 4.3,
    "geoneutrinos": 2.4,
    "world_reactors": 0.88,
    "bi_po": 0.18,
    "others": (
        0.04
        + 0.02
        + 0.05
        + 0.08
        + 4.9e-2
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
    "world_reactors": interpolateToBins(
        E_raw,
        df_raw[
            "world_reactors_digitized"
        ].to_numpy(dtype=float),
        E_prompt_bins,
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

configuration_cache = {}

for config_name, config in CONFIGURATIONS.items():

    R0 = compute_response_matrix(
        r_nl=config["r_nl"],
        r_res=config["r_res"],
        sigma_resolution=config[
            "sigma_resolution"
        ],
        xi_scale=0.0,
        xi_resolution=0.0,
    )

    R_scale_minus = compute_response_matrix(
        r_nl=config["r_nl"],
        r_res=config["r_res"],
        sigma_resolution=config[
            "sigma_resolution"
        ],
        xi_scale=-1.0,
        xi_resolution=0.0,
    )

    R_scale_plus = compute_response_matrix(
        r_nl=config["r_nl"],
        r_res=config["r_res"],
        sigma_resolution=config[
            "sigma_resolution"
        ],
        xi_scale=+1.0,
        xi_resolution=0.0,
    )

    R_resolution_minus = compute_response_matrix(
        r_nl=config["r_nl"],
        r_res=config["r_res"],
        sigma_resolution=config[
            "sigma_resolution"
        ],
        xi_scale=0.0,
        xi_resolution=-1.0,
    )

    R_resolution_plus = compute_response_matrix(
        r_nl=config["r_nl"],
        r_res=config["r_res"],
        sigma_resolution=config[
            "sigma_resolution"
        ],
        xi_scale=0.0,
        xi_resolution=+1.0,
    )

    backgrounds = {}

    for name, spectrum in background_base.items():

        if name == "geoneutrinos":

            backgrounds[name] = (
                spectrum.copy()
            )

        else:

            backgrounds[name] = (
                config["r_bg"]
                * spectrum
            )

    total_background = np.zeros_like(
        E_prompt_bins
    )

    for spectrum in backgrounds.values():

        total_background += (
            spectrum
        )

    configuration_cache[
        config_name
    ] = {
        "R0": R0,
        "R_scale_minus": R_scale_minus,
        "R_scale_plus": R_scale_plus,
        "R_resolution_minus": R_resolution_minus,
        "R_resolution_plus": R_resolution_plus,
        "backgrounds": backgrounds,
        "total_background": total_background,
    }


# ============================================================
# Calculate fixed reactor normalization C_norm
# ============================================================

reference_survival = compute_weighted_survival(
    REFERENCE_SIN2_THETA12,
    REFERENCE_DM21,
)

reference_kernel = (
    phi0_E
    * sigma_IBD
    * reference_survival
    * trap_weights
)

reference_raw_reactor = (
    configuration_cache[
        "cnf1"
    ]["R0"]
    @ reference_kernel
)


if REACTOR_NORMALIZATION_MODE == "area":

    model_total = np.sum(
        reference_raw_reactor[
            fit_mask
        ]
    )

    juno_total = np.sum(
        juno_reactor_on_grid[
            fit_mask
        ]
    )

    if model_total <= 0.0:

        raise ValueError(
            "The reference model reactor total is nonpositive."
        )

    C_norm = (
        juno_total
        / model_total
    )


elif REACTOR_NORMALIZATION_MODE == "peak":

    model_peak = np.max(
        reference_raw_reactor[
            fit_mask
        ]
    )

    juno_peak = np.max(
        juno_reactor_on_grid[
            fit_mask
        ]
    )

    if model_peak <= 0.0:

        raise ValueError(
            "The reference model reactor peak is nonpositive."
        )

    C_norm = (
        juno_peak
        / model_peak
    )


else:

    raise ValueError(
        "REACTOR_NORMALIZATION_MODE must be "
        "'area' or 'peak'."
    )


print(
    f"Fixed reactor normalization C_norm = {C_norm:.8e}"
)


# ============================================================
# Build linearized prediction and pull templates
# ============================================================

def build_linearized_model(
    sin2_theta12,
    dm21,
    config_name,
):
    """
    Build

        prediction = nominal + A @ xi

    where xi contains all standardized nuisance pulls.

    Background and flux pulls are linear exactly.

    Energy scale and resolution are represented by symmetric
    finite-difference derivatives around their nominal values.
    """

    cache = configuration_cache[
        config_name
    ]

    survival = compute_weighted_survival(
        sin2_theta12,
        dm21,
    )

    common_energy_kernel = (
        sigma_IBD
        * survival
        * trap_weights
    )

    central_kernel = (
        phi0_E
        * common_energy_kernel
    )

    flux_kernel_matrix = (
        phi_basis_E
        * common_energy_kernel[
            :,
            None
        ]
    )


    # --------------------------------------------------------
    # Nominal reactor prediction
    # --------------------------------------------------------

    reactor_raw = (
        cache["R0"]
        @ central_kernel
    )

    reactor_nominal = (
        C_norm
        * reactor_raw
    )


    # --------------------------------------------------------
    # Flux-mode templates
    # --------------------------------------------------------

    flux_templates = (
        C_norm
        * (
            cache["R0"]
            @ flux_kernel_matrix
        )
    )


    # --------------------------------------------------------
    # Energy-scale template
    # --------------------------------------------------------

    reactor_scale_minus = (
        C_norm
        * (
            cache["R_scale_minus"]
            @ central_kernel
        )
    )

    reactor_scale_plus = (
        C_norm
        * (
            cache["R_scale_plus"]
            @ central_kernel
        )
    )

    scale_template = 0.5 * (
        reactor_scale_plus
        - reactor_scale_minus
    )


    # --------------------------------------------------------
    # Resolution template
    # --------------------------------------------------------

    reactor_resolution_minus = (
        C_norm
        * (
            cache["R_resolution_minus"]
            @ central_kernel
        )
    )

    reactor_resolution_plus = (
        C_norm
        * (
            cache["R_resolution_plus"]
            @ central_kernel
        )
    )

    resolution_template = 0.5 * (
        reactor_resolution_plus
        - reactor_resolution_minus
    )


    # --------------------------------------------------------
    # Nominal backgrounds
    # --------------------------------------------------------

    backgrounds = cache[
        "backgrounds"
    ]

    total_background = cache[
        "total_background"
    ]


    # --------------------------------------------------------
    # Complete nominal prediction
    # --------------------------------------------------------

    nominal = (
        reactor_nominal
        + total_background
    )


    # --------------------------------------------------------
    # Pull-template matrix
    # --------------------------------------------------------

    A = np.zeros(
        (
            len(E_prompt_bins),
            N_NUISANCE,
        ),
        dtype=float,
    )

    # Reactor-rate normalization
    A[
        :,
        IDX_REACTOR_NORM
    ] = (
        SIGMA_REACTOR_RATE
        * reactor_nominal
    )

    # Energy response
    A[
        :,
        IDX_ENERGY_SCALE
    ] = scale_template

    A[
        :,
        IDX_ENERGY_RESOLUTION
    ] = resolution_template

    # Background normalizations
    A[
        :,
        IDX_LIHE_NORM
    ] = (
        BACKGROUND_NORM_SIGMAS["Li_He"]
        * backgrounds["Li_He"]
    )

    # Li/He shape uncertainty
    lihe_shape_fraction = (
        SIGMA_LIHE_SHAPE_AT_1MEV
        * E_prompt_bins
        / 1.0
    )

    A[
        :,
        IDX_LIHE_SHAPE
    ] = (
        lihe_shape_fraction
        * backgrounds["Li_He"]
    )

    A[
        :,
        IDX_GEO_NORM
    ] = (
        BACKGROUND_NORM_SIGMAS[
            "geoneutrinos"
        ]
        * backgrounds[
            "geoneutrinos"
        ]
    )

    A[
        :,
        IDX_WORLD_NORM
    ] = (
        BACKGROUND_NORM_SIGMAS[
            "world_reactors"
        ]
        * backgrounds[
            "world_reactors"
        ]
    )

    A[
        :,
        IDX_BIPO_NORM
    ] = (
        BACKGROUND_NORM_SIGMAS[
            "bi_po"
        ]
        * backgrounds[
            "bi_po"
        ]
    )

    A[
        :,
        IDX_OTHER_NORM
    ] = (
        BACKGROUND_NORM_SIGMAS[
            "others"
        ]
        * backgrounds[
            "others"
        ]
    )

    # Daya Bay flux eigenmode pulls
    A[
        :,
        FLUX_START:FLUX_STOP
    ] = flux_templates

    return nominal, A


# ============================================================
# CNP and Poisson chi-square functions
# ============================================================

def chi2_value(
    observed,
    predicted,
    pulls,
    chi2_type,
):
    """
    Evaluate the data chi-square plus Gaussian pull penalties.
    """

    observed = np.asarray(
        observed,
        dtype=float,
    )

    predicted = np.asarray(
        predicted,
        dtype=float,
    )

    pulls = np.asarray(
        pulls,
        dtype=float,
    )

    if np.any(
        predicted <= 0.0
    ):

        return np.inf

    if chi2_type == "cnp":

        # Equivalent to
        #
        # (pred - obs)^2 /
        # [3 / (1/obs + 2/pred)]
        #
        data_term = (
            predicted**2
            / (
                3.0
                * observed
            )
            - observed
            + 2.0
            * observed**2
            / (
                3.0
                * predicted
            )
        )

        chi2_data = np.sum(
            data_term
        )


    elif chi2_type == "poisson":

        data_term = (
            predicted
            - observed
            + observed
            * np.log(
                observed
                / predicted
            )
        )

        chi2_data = (
            2.0
            * np.sum(
                data_term
            )
        )


    else:

        raise ValueError(
            "chi2_type must be 'cnp' or 'poisson'."
        )

    chi2_pulls = np.sum(
        pulls**2
    )

    return (
        chi2_data
        + chi2_pulls
    )


# ============================================================
# Profile Gaussian pull parameters with Newton iterations
# ============================================================

def profile_pulls(
    nominal_full,
    A_full,
    chi2_type,
    initial_pulls=None,
):
    """
    Minimize the chi-square over all nuisance pulls.

    Since the prediction is linearized as

        N = N0 + A xi,

    the gradient and Hessian can be calculated directly.
    """

    nominal = np.asarray(
        nominal_full[
            fit_mask
        ],
        dtype=float,
    )

    A = np.asarray(
        A_full[
            fit_mask,
            :
        ],
        dtype=float,
    )

    observed = np.asarray(
        observed_on_grid[
            fit_mask
        ],
        dtype=float,
    )


    if initial_pulls is None:

        pulls = np.zeros(
            N_NUISANCE,
            dtype=float,
        )

    else:

        pulls = np.asarray(
            initial_pulls,
            dtype=float,
        ).copy()


    # If a warm start produces nonpositive predictions,
    # gradually shrink it toward zero.
    for _ in range(20):

        predicted = (
            nominal
            + A
            @ pulls
        )

        if np.all(
            predicted > 1.0e-10
        ):

            break

        pulls *= 0.5

    identity = np.eye(
        N_NUISANCE
    )


    for _ in range(
        MAX_PULL_ITERATIONS
    ):

        predicted = (
            nominal
            + A
            @ pulls
        )

        if np.any(
            predicted <= 1.0e-12
        ):

            pulls *= 0.5
            continue


        # ----------------------------------------------------
        # Gradient and Hessian of the data term
        # ----------------------------------------------------

        if chi2_type == "cnp":

            derivative = (
                2.0
                * predicted
                / (
                    3.0
                    * observed
                )
                - 2.0
                * observed**2
                / (
                    3.0
                    * predicted**2
                )
            )

            curvature = (
                2.0
                / (
                    3.0
                    * observed
                )
                + 4.0
                * observed**2
                / (
                    3.0
                    * predicted**3
                )
            )


        elif chi2_type == "poisson":

            derivative = (
                2.0
                * (
                    1.0
                    - observed
                    / predicted
                )
            )

            curvature = (
                2.0
                * observed
                / predicted**2
            )


        else:

            raise ValueError(
                "Unknown chi-square type."
            )


        gradient = (
            A.T
            @ derivative
            + 2.0
            * pulls
        )

        Hessian = (
            A.T
            @ (
                curvature[
                    :,
                    None
                ]
                * A
            )
            + 2.0
            * identity
        )

        Hessian += (
            1.0e-10
            * identity
        )


        if np.max(
            np.abs(
                gradient
            )
        ) < PULL_TOLERANCE:

            break


        try:

            step = np.linalg.solve(
                Hessian,
                -gradient,
            )

        except np.linalg.LinAlgError:

            step = np.linalg.lstsq(
                Hessian,
                -gradient,
                rcond=None,
            )[0]


        current_chi2 = chi2_value(
            observed,
            predicted,
            pulls,
            chi2_type,
        )

        directional_derivative = (
            gradient
            @ step
        )

        step_scale = 1.0

        accepted = False

        for _ in range(25):

            trial_pulls = (
                pulls
                + step_scale
                * step
            )

            trial_prediction = (
                nominal
                + A
                @ trial_pulls
            )

            if np.any(
                trial_prediction <= 1.0e-12
            ):

                step_scale *= 0.5
                continue

            trial_chi2 = chi2_value(
                observed,
                trial_prediction,
                trial_pulls,
                chi2_type,
            )

            armijo_limit = (
                current_chi2
                + 1.0e-4
                * step_scale
                * directional_derivative
            )

            if (
                np.isfinite(
                    trial_chi2
                )
                and trial_chi2
                <= armijo_limit
            ):

                pulls = trial_pulls
                accepted = True
                break

            step_scale *= 0.5


        if not accepted:

            break


        if np.linalg.norm(
            step_scale
            * step
        ) < (
            PULL_TOLERANCE
            * (
                1.0
                + np.linalg.norm(
                    pulls
                )
            )
        ):

            break


    predicted_fit = (
        nominal
        + A
        @ pulls
    )

    final_chi2 = chi2_value(
        observed,
        predicted_fit,
        pulls,
        chi2_type,
    )

    predicted_full = (
        nominal_full
        + A_full
        @ pulls
    )

    return (
        final_chi2,
        pulls,
        predicted_full,
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

def scan_configuration(
    config_name,
):
    """
    Scan sin^2(theta12) and dm21 while profiling all
    nuisance pulls at each grid point.
    """

    config = CONFIGURATIONS[
        config_name
    ]

    chi2_grid = np.zeros(
        (
            N_DM21,
            N_THETA12,
        ),
        dtype=float,
    )

    pulls_grid = np.zeros(
        (
            N_DM21,
            N_THETA12,
            N_NUISANCE,
        ),
        dtype=float,
    )

    print()
    print("=" * 72)
    print(
        f"Scanning configuration: {config_name}"
    )
    print(
        f"Statistical definition: "
        f"{config['chi2_type']}"
    )
    print("=" * 72)


    for i_dm, dm21_test in enumerate(
        dm21_grid
    ):

        # Snake through the grid so neighboring points
        # can be used as warm starts.
        if i_dm % 2 == 0:

            theta_indices = range(
                N_THETA12
            )

        else:

            theta_indices = range(
                N_THETA12 - 1,
                -1,
                -1,
            )


        previous_theta_index = None

        for i_theta in theta_indices:

            sin2_test = (
                sin2_theta12_grid[
                    i_theta
                ]
            )

            nominal, A = build_linearized_model(
                sin2_theta12=sin2_test,
                dm21=dm21_test,
                config_name=config_name,
            )


            if previous_theta_index is not None:

                initial_pulls = pulls_grid[
                    i_dm,
                    previous_theta_index,
                    :
                ]

            elif i_dm > 0:

                initial_pulls = pulls_grid[
                    i_dm - 1,
                    i_theta,
                    :
                ]

            else:

                initial_pulls = np.zeros(
                    N_NUISANCE
                )


            chi2, best_pulls, _ = profile_pulls(
                nominal_full=nominal,
                A_full=A,
                chi2_type=config[
                    "chi2_type"
                ],
                initial_pulls=initial_pulls,
            )

            chi2_grid[
                i_dm,
                i_theta
            ] = chi2

            pulls_grid[
                i_dm,
                i_theta,
                :
            ] = best_pulls

            previous_theta_index = (
                i_theta
            )


        print(
            f"Completed dm21 row "
            f"{i_dm + 1:3d}/{N_DM21}: "
            f"{dm21_test:.6e} eV^2"
        )


    # --------------------------------------------------------
    # Best grid point
    # --------------------------------------------------------

    best_grid_index = np.unravel_index(
        np.argmin(
            chi2_grid
        ),
        chi2_grid.shape,
    )

    best_grid_dm_index = (
        best_grid_index[0]
    )

    best_grid_theta_index = (
        best_grid_index[1]
    )

    best_theta = (
        sin2_theta12_grid[
            best_grid_theta_index
        ]
    )

    best_dm21 = (
        dm21_grid[
            best_grid_dm_index
        ]
    )

    best_pulls = pulls_grid[
        best_grid_dm_index,
        best_grid_theta_index,
        :
    ].copy()


    # --------------------------------------------------------
    # Optional continuous refinement
    # --------------------------------------------------------

    if REFINE_BEST_FIT:

        dm_scale = 1.0e5

        def profiled_physics_chi2(
            physics_parameters,
        ):

            theta_test = float(
                physics_parameters[0]
            )

            dm_test = float(
                physics_parameters[1]
                / dm_scale
            )

            nominal, A = build_linearized_model(
                sin2_theta12=theta_test,
                dm21=dm_test,
                config_name=config_name,
            )

            chi2, _, _ = profile_pulls(
                nominal_full=nominal,
                A_full=A,
                chi2_type=config[
                    "chi2_type"
                ],
                initial_pulls=best_pulls,
            )

            return chi2


        refinement = minimize(
            profiled_physics_chi2,
            x0=np.array([
                best_theta,
                best_dm21
                * dm_scale,
            ]),
            method="L-BFGS-B",
            bounds=[
                (
                    SIN2_THETA12_MIN,
                    SIN2_THETA12_MAX,
                ),
                (
                    DM21_MIN
                    * dm_scale,
                    DM21_MAX
                    * dm_scale,
                ),
            ],
            options={
                "maxiter": 80,
                "ftol": 1.0e-10,
            },
        )


        if refinement.success:

            best_theta = float(
                refinement.x[0]
            )

            best_dm21 = float(
                refinement.x[1]
                / dm_scale
            )

        else:

            print(
                "Continuous refinement did not fully converge. "
                "Using its final point."
            )

            best_theta = float(
                refinement.x[0]
            )

            best_dm21 = float(
                refinement.x[1]
                / dm_scale
            )


    # --------------------------------------------------------
    # Re-profile at final best-fit point
    # --------------------------------------------------------

    best_nominal, best_A = (
        build_linearized_model(
            sin2_theta12=best_theta,
            dm21=best_dm21,
            config_name=config_name,
        )
    )

    best_chi2, best_pulls, best_prediction = (
        profile_pulls(
            nominal_full=best_nominal,
            A_full=best_A,
            chi2_type=config[
                "chi2_type"
            ],
            initial_pulls=best_pulls,
        )
    )


    delta_chi2_grid = (
        chi2_grid
        - best_chi2
    )

    # Prevent tiny negative values caused by numerical
    # refinement below the discrete grid minimum.
    delta_chi2_grid = np.maximum(
        delta_chi2_grid,
        0.0,
    )


    profile_theta12 = np.min(
        delta_chi2_grid,
        axis=0,
    )

    profile_dm21 = np.min(
        delta_chi2_grid,
        axis=1,
    )


    print()
    print(
        f"Best fit for {config_name}"
    )
    print("-" * 72)

    print(
        f"sin^2(theta12) = "
        f"{best_theta:.8f}"
    )

    print(
        f"dm21            = "
        f"{best_dm21:.8e} eV^2"
    )

    print(
        f"chi2_min        = "
        f"{best_chi2:.6f}"
    )

    print(
        "Largest fitted nuisance pulls:"
    )

    pull_order = np.argsort(
        np.abs(
            best_pulls
        )
    )[::-1]

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
        "best_theta12": best_theta,
        "best_dm21": best_dm21,
        "best_chi2": best_chi2,
        "best_pulls": best_pulls,
        "best_prediction": best_prediction,
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


# ============================================================
# Correlated-Gaussian JUNO reference
# ============================================================

THETA_MESH, DM21_MESH = np.meshgrid(
    sin2_theta12_grid,
    dm21_grid,
)

juno_theta_standardized = (
    THETA_MESH
    - JUNO_BEST_SIN2_THETA12
) / JUNO_SIGMA_SIN2_THETA12

juno_dm21_standardized = (
    DM21_MESH
    - JUNO_BEST_DM21
) / JUNO_SIGMA_DM21

JUNO_DELTA_CHI2_GRID = (
    juno_theta_standardized**2
    - 2.0
    * JUNO_CORRELATION
    * juno_theta_standardized
    * juno_dm21_standardized
    + juno_dm21_standardized**2
) / (
    1.0
    - JUNO_CORRELATION**2
)

JUNO_PROFILE_THETA12 = (
    (
        sin2_theta12_grid
        - JUNO_BEST_SIN2_THETA12
    )
    / JUNO_SIGMA_SIN2_THETA12
) ** 2

JUNO_PROFILE_DM21 = (
    (
        dm21_grid
        - JUNO_BEST_DM21
    )
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


# ============================================================
# Plot JUNO reference as black dashed curves
# ============================================================

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

ax_contour.scatter(
    JUNO_BEST_SIN2_THETA12,
    JUNO_BEST_DM21 * 1.0e5,
    marker="+",
    s=65,
    linewidths=1.8,
    color="black",
    zorder=5,
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

    result = results[
        config_name
    ]

    axis.plot(
        E_prompt_bins[
            fit_mask
        ],
        result[
            "best_prediction"
        ][
            fit_mask
        ],
        lw=2.0,
        label=config_name,
    )

    axis.plot(
        E_prompt_bins[
            fit_mask
        ],
        observed_on_grid[
            fit_mask
        ],
        "--",
        color="black",
        lw=1.4,
        label="JUNO",
    )

    axis.set_xlim(
        FIT_ENERGY_MIN,
        FIT_ENERGY_MAX,
    )

    axis.set_ylabel(
        "Events per 0.1 MeV"
    )

    axis.grid(
        alpha=0.25
    )

    axis.legend(
        frameon=False,
        fontsize=9,
    )

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


ax_spectrum_cnf1.tick_params(
    labelbottom=False
)

ax_spectrum_cnf2.set_xlabel(
    r"$E_{\rm pr}\ [{\rm MeV}]$"
)


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