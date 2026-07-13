import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from scipy.interpolate import PchipInterpolator
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

USE_RANDOM_PULLS = True

# Same seed gives the same systematic realization.
# Use None for new random pulls every run.
PULL_SEED = 123

# Standard Gaussian pulls correspond to scale = 1.
PULL_SCALE = 1.0


# ------------------------------------------------------------
# Individual systematic switches
# ------------------------------------------------------------

USE_REACTOR_RATE_PULL = True
USE_BACKGROUND_NORM_PULLS = True
USE_LIHE_SHAPE_PULL = True
USE_FLUX_PULLS = True
USE_ENERGY_RESOLUTION_PULL = True

# The paper states that scale and bias have very similar effects,
# so usually only one is used at a time.
#
# Options:
#   "scale"
#   "bias"
#   "both"
#   "none"
#
ENERGY_SCALE_BIAS_MODE = "scale"


# ------------------------------------------------------------
# Absolute reactor normalization
# ------------------------------------------------------------

# Options:
#   "peak" -> match the maximum JUNO reactor bin
#   "area" -> match the total JUNO reactor events
#
REACTOR_NORMALIZATION_MODE = "peak"


# ------------------------------------------------------------
# Fixed configuration rescaling factors
# ------------------------------------------------------------

# Background rescaling, excluding geoneutrinos.
R_BG = 1.0

# Nonlinearity rescaling:
#   1.0 = nominal nonlinearity
#   0.0 = no nonlinearity
R_NL = 1.0

# Fixed resolution rescaling.
R_RES = 1.0


# ------------------------------------------------------------
# Output
# ------------------------------------------------------------

FIG_PATH = "img/osc_all_systematics.png"


# ============================================================
# Random-number generator
# ============================================================

rng = np.random.default_rng(PULL_SEED)


# ============================================================
# Pull helper functions
# ============================================================

def draw_pull(enabled=True):
    """
    Draw one Gaussian pull xi ~ N(0, PULL_SCALE).
    """

    if not USE_RANDOM_PULLS or not enabled:
        return 0.0

    return float(
        rng.normal(
            loc=0.0,
            scale=PULL_SCALE,
        )
    )


def draw_nonnegative_normalization_pull(
    fractional_uncertainty,
    enabled=True,
):
    """
    Draw a Gaussian normalization pull while requiring

        1 + sigma * xi >= 0.
    """

    if not USE_RANDOM_PULLS or not enabled:
        return 0.0

    while True:

        xi = draw_pull(enabled=True)

        factor = (
            1.0
            + fractional_uncertainty * xi
        )

        if factor >= 0.0:
            return xi


def draw_lihe_pulls(
    energy,
    sigma_norm,
    sigma_shape_at_1mev,
):
    """
    Draw the Li/He normalization and shape pulls while requiring

        1
        + sigma_norm * xi_norm
        + sigma_shape(E) * xi_shape >= 0

    in every prompt-energy bin.
    """

    if not USE_RANDOM_PULLS:

        return 0.0, 0.0

    while True:

        if USE_BACKGROUND_NORM_PULLS:
            xi_norm = draw_pull()
        else:
            xi_norm = 0.0

        if USE_LIHE_SHAPE_PULL:
            xi_shape = draw_pull()
        else:
            xi_shape = 0.0

        sigma_shape = (
            sigma_shape_at_1mev
            * energy
            / 1.0
        )

        factor = (
            1.0
            + sigma_norm * xi_norm
            + sigma_shape * xi_shape
        )

        if np.all(factor >= 0.0):
            return xi_norm, xi_shape


# ============================================================
# Physical constants
# ============================================================

kg_to_MeV = 5.61e29

m_p = 1.6726219e-27 * kg_to_MeV
m_n = 1.6749275e-27 * kg_to_MeV
m_e = 9.1093837e-31 * kg_to_MeV

Delta = m_n - m_p


# ============================================================
# Oscillation parameters
# ============================================================

sin2_theta12 = 0.308
sin2_theta13 = 0.02215

dm21 = 7.49e-5
dm31 = 2.513e-3


# ============================================================
# Detector settings
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

E_nu = np.linspace(
    1.81,
    10.0,
    2000,
)

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
# 20% at 1 MeV and linearly proportional to energy.
SIGMA_LIHE_SHAPE_AT_1MEV = 0.20


# The supplied page gives 0.5% for scale and bias.
SIGMA_ENERGY_SCALE = 0.005
SIGMA_ENERGY_BIAS = 0.005

# Energy-resolution uncertainty.
SIGMA_ENERGY_RESOLUTION = 0.05


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
# Build continuous Daya Bay flux model
# ============================================================

DYB_PATH = (
    "data/"
    "DYB_unfolded_spectra_tot_U235_Pu239.txt"
)

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

nbin = extras["nbin"]

print(
    f"Number of reactor flux pulls: {nbin}"
)


# ============================================================
# Load positron nonlinearity
# ============================================================

NONLINEARITY_PATH = (
    "data/positron_nonlinearity.csv"
)


def load_nonlinearity_curve(path):
    """
    Load the first two usable numerical columns as
    energy and nonlinearity factor.
    """

    df = pd.read_csv(path)

    numeric_df = df.apply(
        pd.to_numeric,
        errors="coerce",
    )

    valid_columns = [
        column
        for column in numeric_df.columns
        if numeric_df[column].notna().sum() > 1
    ]

    if len(valid_columns) < 2:

        df = pd.read_csv(
            path,
            header=None,
        )

        numeric_df = df.apply(
            pd.to_numeric,
            errors="coerce",
        )

        valid_columns = [
            column
            for column in numeric_df.columns
            if numeric_df[column].notna().sum() > 1
        ]

    if len(valid_columns) < 2:
        raise ValueError(
            "Could not identify two numerical columns "
            "in the nonlinearity file."
        )

    energy = numeric_df[
        valid_columns[0]
    ].to_numpy(dtype=float)

    factor = numeric_df[
        valid_columns[1]
    ].to_numpy(dtype=float)

    good = (
        np.isfinite(energy)
        & np.isfinite(factor)
    )

    energy = energy[good]
    factor = factor[good]

    order = np.argsort(
        energy
    )

    energy = energy[order]
    factor = factor[order]

    energy, unique_indices = np.unique(
        energy,
        return_index=True,
    )

    factor = factor[
        unique_indices
    ]

    return energy, factor


E_nl_points, F_nl_points = load_nonlinearity_curve(
    NONLINEARITY_PATH
)

F_nl_interpolator = PchipInterpolator(
    E_nl_points,
    F_nl_points,
    extrapolate=False,
)


def F_nl(E_prompt):
    """
    Evaluate the nominal nonlinearity factor.
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

    return F_nl_interpolator(
        E_clipped
    )


# ============================================================
# Detector response functions
# ============================================================

def sigma_prompt(E_prompt):
    """
    Nominal prompt-energy resolution width.
    """

    E_prompt = np.asarray(
        E_prompt,
        dtype=float,
    )

    E_safe = np.clip(
        E_prompt,
        1e-8,
        None,
    )

    return np.sqrt(
        res_a**2 * E_safe
        + res_b**2 * E_safe**2
        + res_c**2
    )


def compute_response_matrix_nl(
    xi_scale=0.0,
    xi_bias=0.0,
    xi_resolution=0.0,
):
    """
    Construct the nonlinear response matrix including

        energy scale,
        additive nonlinearity bias,
        fixed nonlinearity rescaling,
        energy resolution uncertainty,
        fixed resolution rescaling.
    """

    E_visible = (
        E_nu
        - Delta
        + m_e
    )

    E_prompt_0 = (
        prompt_alpha * E_visible
        + prompt_beta
    )

    nominal_nonlinearity = F_nl(
        E_prompt_0
    )

    # r_nl = 1 gives the full nominal nonlinearity.
    # r_nl = 0 removes the nonlinear deformation.
    effective_nonlinearity = (
        1.0
        + R_NL
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

    bias_term = (
        SIGMA_ENERGY_BIAS
        * xi_bias
    )

    # Equation corresponding to
    #
    # Ehat_pr = E_pr [
    #     (1 + scale pull) F_nl(E_pr)
    #     + bias pull
    # ]
    #
    mu = (
        E_prompt_0
        * (
            scale_factor
            * effective_nonlinearity
            + bias_term
        )
    )

    if np.any(mu <= 0.0):
        raise ValueError(
            "Energy-scale or bias pull produced "
            "nonpositive reconstructed energy."
        )

    resolution_factor = (
        1.0
        + SIGMA_ENERGY_RESOLUTION
        * xi_resolution
    )

    if resolution_factor <= 0.0:
        raise ValueError(
            "Energy-resolution pull produced "
            "a nonpositive resolution factor."
        )

    sigma_E = (
        R_RES
        * resolution_factor
        * sigma_prompt(mu)
    )

    sigma_E = np.clip(
        sigma_E,
        1e-12,
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

    response_matrix = 0.5 * (
        erf(z_upper)
        - erf(z_lower)
    )

    response_matrix[
        :,
        E_visible <= 0.0
    ] = 0.0

    return response_matrix


# ============================================================
# Trapezoid integration weights
# ============================================================

trap_weights = np.zeros_like(
    E_nu
)

trap_weights[1:-1] = 0.5 * (
    E_nu[2:]
    - E_nu[:-2]
)

trap_weights[0] = 0.5 * (
    E_nu[1]
    - E_nu[0]
)

trap_weights[-1] = 0.5 * (
    E_nu[-1]
    - E_nu[-2]
)


# ============================================================
# Reactor survival probability
# ============================================================

Pee_weighted = np.zeros_like(
    E_nu
)

for _, reactor in reactor_data.iterrows():

    L_km = reactor["L_km"]
    reactor_weight = reactor["w"]

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


sigma_IBD = sigma_ibd(
    E_nu,
    Delta,
    m_e,
)


# ============================================================
# Reactor-spectrum function
# ============================================================

def compute_reactor_spectrum(
    xi_flux,
    xi_scale=0.0,
    xi_bias=0.0,
    xi_resolution=0.0,
):
    """
    Compute the reactor prompt spectrum with flux and
    detector-response pulls.
    """

    xi_flux = np.asarray(
        xi_flux,
        dtype=float,
    )

    if xi_flux.size != nbin:
        raise ValueError(
            f"Expected {nbin} flux pulls, "
            f"received {xi_flux.size}."
        )

    phi_E = np.asarray(
        phi_cont(
            E_nu,
            xi_flux,
        ),
        dtype=float,
    ).ravel()

    phi_E = np.clip(
        phi_E,
        0.0,
        None,
    )

    response_matrix = compute_response_matrix_nl(
        xi_scale=xi_scale,
        xi_bias=xi_bias,
        xi_resolution=xi_resolution,
    )

    neutrino_kernel = (
        phi_E
        * sigma_IBD
        * Pee_weighted
        * trap_weights
    )

    spectrum = (
        response_matrix
        @ neutrino_kernel
    )

    return np.clip(
        spectrum,
        0.0,
        None,
    )


# ============================================================
# Draw reactor and detector pulls
# ============================================================

XI_REACTOR_RATE = (
    draw_nonnegative_normalization_pull(
        SIGMA_REACTOR_RATE,
        enabled=USE_REACTOR_RATE_PULL,
    )
)


if USE_RANDOM_PULLS and USE_FLUX_PULLS:

    XI_FLUX = rng.normal(
        loc=0.0,
        scale=PULL_SCALE,
        size=nbin,
    )

else:

    XI_FLUX = np.zeros(
        nbin,
        dtype=float,
    )


if ENERGY_SCALE_BIAS_MODE in {
    "scale",
    "both",
}:

    XI_ENERGY_SCALE = draw_pull()

else:

    XI_ENERGY_SCALE = 0.0


if ENERGY_SCALE_BIAS_MODE in {
    "bias",
    "both",
}:

    XI_ENERGY_BIAS = draw_pull()

else:

    XI_ENERGY_BIAS = 0.0


XI_ENERGY_RESOLUTION = (
    draw_nonnegative_normalization_pull(
        SIGMA_ENERGY_RESOLUTION,
        enabled=USE_ENERGY_RESOLUTION_PULL,
    )
)


# ============================================================
# Nominal and pulled reactor spectra
# ============================================================

reactor_raw_nominal = compute_reactor_spectrum(
    xi_flux=np.zeros(
        nbin,
        dtype=float,
    ),
    xi_scale=0.0,
    xi_bias=0.0,
    xi_resolution=0.0,
)

reactor_raw_pulled = compute_reactor_spectrum(
    xi_flux=XI_FLUX,
    xi_scale=XI_ENERGY_SCALE,
    xi_bias=XI_ENERGY_BIAS,
    xi_resolution=XI_ENERGY_RESOLUTION,
)


# ============================================================
# Load JUNO reference
# ============================================================

JUNO_PATH = "data/spect-fit.txt"

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

juno_order = np.argsort(
    juno_energy
)

juno_energy = juno_energy[
    juno_order
]

juno_reactor_signal = juno_reactor_signal[
    juno_order
]


# ============================================================
# Absolute reactor normalization
# ============================================================

if REACTOR_NORMALIZATION_MODE == "peak":

    model_peak = np.max(
        reactor_raw_nominal
    )

    juno_peak = np.max(
        juno_reactor_signal
    )

    if model_peak <= 0.0:
        raise ValueError(
            "Nominal model reactor spectrum "
            "has a nonpositive peak."
        )

    C_norm = (
        juno_peak
        / model_peak
    )


elif REACTOR_NORMALIZATION_MODE == "area":

    juno_on_model_grid = np.interp(
        Epr_centers,
        juno_energy,
        juno_reactor_signal,
        left=np.nan,
        right=np.nan,
    )

    common_mask = (
        np.isfinite(
            juno_on_model_grid
        )
        & np.isfinite(
            reactor_raw_nominal
        )
    )

    if not np.any(common_mask):
        raise ValueError(
            "JUNO and model spectra do not overlap."
        )

    model_total = np.sum(
        reactor_raw_nominal[
            common_mask
        ]
    )

    juno_total = np.sum(
        juno_on_model_grid[
            common_mask
        ]
    )

    if model_total <= 0.0:
        raise ValueError(
            "Nominal model reactor spectrum "
            "has a nonpositive total."
        )

    C_norm = (
        juno_total
        / model_total
    )


else:

    raise ValueError(
        "REACTOR_NORMALIZATION_MODE must be "
        "'peak' or 'area'."
    )


osc_spectra_nominal = (
    C_norm
    * reactor_raw_nominal
)


reactor_rate_factor = (
    1.0
    + SIGMA_REACTOR_RATE
    * XI_REACTOR_RATE
)


osc_spectra = (
    reactor_rate_factor
    * C_norm
    * reactor_raw_pulled
)


# ============================================================
# Background rates
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


# ============================================================
# Load digitized background shapes
# ============================================================

BG_PATH = (
    "data/digitized_backgrounds.csv"
)

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
    df_raw[
        "world_reactors_digitized"
    ].to_numpy(dtype=float),
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


# ============================================================
# Normalize backgrounds to Table 1
# ============================================================

Li_He_base = normalizeToTable(
    Li_He_shape,
    "Li_He",
    BIN_WIDTH,
    TABLE1_TOTAL_EVENTS,
)

geoneutrinos_base = normalizeToTable(
    geoneutrinos_shape,
    "geoneutrinos",
    BIN_WIDTH,
    TABLE1_TOTAL_EVENTS,
)

world_reactors_base = normalizeToTable(
    world_reactors_shape,
    "world_reactors",
    BIN_WIDTH,
    TABLE1_TOTAL_EVENTS,
)

bi_po_base = normalizeToTable(
    bi_po_shape,
    "bi_po",
    BIN_WIDTH,
    TABLE1_TOTAL_EVENTS,
)

others_base = normalizeToTable(
    others_shape,
    "others",
    BIN_WIDTH,
    TABLE1_TOTAL_EVENTS,
)


background_base = {
    "Li_He": Li_He_base,
    "geoneutrinos": geoneutrinos_base,
    "world_reactors": world_reactors_base,
    "bi_po": bi_po_base,
    "others": others_base,
}


# ============================================================
# Draw background normalization pulls
# ============================================================

XI_BACKGROUND = {}


XI_BACKGROUND["Li_He"], XI_LIHE_SHAPE = (
    draw_lihe_pulls(
        energy=E_prompt_bins,
        sigma_norm=BACKGROUND_NORM_SIGMAS[
            "Li_He"
        ],
        sigma_shape_at_1mev=(
            SIGMA_LIHE_SHAPE_AT_1MEV
        ),
    )
)


for name in [
    "geoneutrinos",
    "world_reactors",
    "bi_po",
    "others",
]:

    XI_BACKGROUND[name] = (
        draw_nonnegative_normalization_pull(
            BACKGROUND_NORM_SIGMAS[name],
            enabled=(
                USE_BACKGROUND_NORM_PULLS
            ),
        )
    )


# ============================================================
# Nominal backgrounds with fixed configuration rescaling
# ============================================================

background_nominal = {
    # r_BG affects Li/He.
    "Li_He": (
        R_BG
        * Li_He_base
    ),

    # The paper excludes geoneutrinos from r_BG.
    "geoneutrinos": geoneutrinos_base,

    "world_reactors": (
        R_BG
        * world_reactors_base
    ),

    "bi_po": (
        R_BG
        * bi_po_base
    ),

    "others": (
        R_BG
        * others_base
    ),
}


# ============================================================
# Apply background pulls
# ============================================================

background_pulled = {}


# ------------------------------------------------------------
# Li/He normalization plus shape uncertainty
#
# Paper-like additive form:
#
# 1
# + sigma_norm * xi_norm
# + sigma_shape(E) * xi_shape
#
# ------------------------------------------------------------

lihe_shape_fraction = (
    SIGMA_LIHE_SHAPE_AT_1MEV
    * E_prompt_bins
    / 1.0
)

lihe_factor = (
    1.0
    + BACKGROUND_NORM_SIGMAS["Li_He"]
    * XI_BACKGROUND["Li_He"]
    + lihe_shape_fraction
    * XI_LIHE_SHAPE
)

if np.any(lihe_factor < 0.0):
    raise ValueError(
        "Li/He normalization plus shape pull "
        "produced a negative factor."
    )

background_pulled["Li_He"] = (
    R_BG
    * lihe_factor
    * Li_He_base
)


# ------------------------------------------------------------
# Remaining background normalizations
# ------------------------------------------------------------

for name in [
    "geoneutrinos",
    "world_reactors",
    "bi_po",
    "others",
]:

    normalization_factor = (
        1.0
        + BACKGROUND_NORM_SIGMAS[name]
        * XI_BACKGROUND[name]
    )

    if normalization_factor < 0.0:
        raise ValueError(
            f"Background '{name}' became negative."
        )

    if name == "geoneutrinos":
        fixed_rescaling = 1.0
    else:
        fixed_rescaling = R_BG

    background_pulled[name] = (
        fixed_rescaling
        * normalization_factor
        * background_base[name]
    )


# ============================================================
# Total background spectra
# ============================================================

Total_Background_nominal = np.zeros_like(
    E_prompt_bins
)

Total_Background = np.zeros_like(
    E_prompt_bins
)

for name in background_nominal:

    Total_Background_nominal += (
        background_nominal[name]
    )

    Total_Background += (
        background_pulled[name]
    )


# ============================================================
# Complete predictions
# ============================================================

osc_spectra_background_nominal = (
    osc_spectra_nominal
    + Total_Background_nominal
)

osc_spectra_background = (
    osc_spectra
    + Total_Background
)


# ============================================================
# Pull chi-square penalty
# ============================================================

chi2_pull_reactor = (
    XI_REACTOR_RATE**2
)

chi2_pull_background = sum(
    xi**2
    for xi in XI_BACKGROUND.values()
)

chi2_pull_lihe_shape = (
    XI_LIHE_SHAPE**2
)

chi2_pull_flux = np.sum(
    XI_FLUX**2
)

chi2_pull_detector = (
    XI_ENERGY_SCALE**2
    + XI_ENERGY_BIAS**2
    + XI_ENERGY_RESOLUTION**2
)

chi2_pull_total = (
    chi2_pull_reactor
    + chi2_pull_background
    + chi2_pull_lihe_shape
    + chi2_pull_flux
    + chi2_pull_detector
)


# ============================================================
# Diagnostics
# ============================================================

print("\n")
print("=" * 82)
print("SYSTEMATIC PULL SUMMARY")
print("=" * 82)

print(
    f"Combined reactor-rate uncertainty: "
    f"{100.0 * SIGMA_REACTOR_RATE:.3f}%"
)

print(
    f"Reactor normalization mode: "
    f"{REACTOR_NORMALIZATION_MODE}"
)

print(
    f"Absolute reactor scale C_norm: "
    f"{C_norm:.6e}"
)

print(
    f"Reactor-rate pull: "
    f"{XI_REACTOR_RATE:+.4f}"
)

print(
    f"Reactor-rate factor: "
    f"{reactor_rate_factor:.6f}"
)

print(
    f"Applied reactor-rate shift: "
    f"{100.0 * (reactor_rate_factor - 1.0):+.3f}%"
)

print(
    f"Energy-scale pull: "
    f"{XI_ENERGY_SCALE:+.4f}"
)

print(
    f"Applied scale shift: "
    f"{100.0 * SIGMA_ENERGY_SCALE * XI_ENERGY_SCALE:+.3f}%"
)

print(
    f"Energy-bias pull: "
    f"{XI_ENERGY_BIAS:+.4f}"
)

print(
    f"Applied bias term: "
    f"{100.0 * SIGMA_ENERGY_BIAS * XI_ENERGY_BIAS:+.3f}%"
)

print(
    f"Energy-resolution pull: "
    f"{XI_ENERGY_RESOLUTION:+.4f}"
)

print(
    f"Applied resolution shift: "
    f"{100.0 * SIGMA_ENERGY_RESOLUTION * XI_ENERGY_RESOLUTION:+.3f}%"
)

print(
    f"Number of flux pulls: "
    f"{len(XI_FLUX)}"
)

print(
    f"Flux-pull RMS: "
    f"{np.sqrt(np.mean(XI_FLUX**2)):.4f}"
)

print(
    f"Li/He normalization pull: "
    f"{XI_BACKGROUND['Li_He']:+.4f}"
)

print(
    f"Li/He shape pull: "
    f"{XI_LIHE_SHAPE:+.4f}"
)

print(
    f"Li/He factor range: "
    f"{np.min(lihe_factor):.4f} "
    f"to {np.max(lihe_factor):.4f}"
)

print("-" * 82)

for name in [
    "geoneutrinos",
    "world_reactors",
    "bi_po",
    "others",
]:

    sigma_bg = (
        BACKGROUND_NORM_SIGMAS[name]
    )

    xi_bg = (
        XI_BACKGROUND[name]
    )

    factor_bg = (
        1.0
        + sigma_bg * xi_bg
    )

    print(
        f"{name:20s} | "
        f"xi = {xi_bg:+8.4f} | "
        f"sigma = {100.0 * sigma_bg:6.2f}% | "
        f"factor = {factor_bg:8.4f}"
    )

print("-" * 82)

print(
    f"Nominal reactor events: "
    f"{np.sum(osc_spectra_nominal):.3f}"
)

print(
    f"Pulled reactor events: "
    f"{np.sum(osc_spectra):.3f}"
)

print(
    f"Nominal background events: "
    f"{np.sum(Total_Background_nominal):.3f}"
)

print(
    f"Pulled background events: "
    f"{np.sum(Total_Background):.3f}"
)

print(
    f"Nominal total events: "
    f"{np.sum(osc_spectra_background_nominal):.3f}"
)

print(
    f"Pulled total events: "
    f"{np.sum(osc_spectra_background):.3f}"
)

print("-" * 82)

print(
    f"Reactor pull penalty: "
    f"{chi2_pull_reactor:.6f}"
)

print(
    f"Background normalization penalty: "
    f"{chi2_pull_background:.6f}"
)

print(
    f"Li/He shape penalty: "
    f"{chi2_pull_lihe_shape:.6f}"
)

print(
    f"Flux-pull penalty: "
    f"{chi2_pull_flux:.6f}"
)

print(
    f"Detector pull penalty: "
    f"{chi2_pull_detector:.6f}"
)

print(
    f"Total pull penalty: "
    f"{chi2_pull_total:.6f}"
)

print("=" * 82)


# ============================================================
# Plot
# ============================================================

plt.figure(
    figsize=(7.8, 5.0)
)

plt.plot(
    E_prompt_bins,
    osc_spectra_background,
    ":",
    linewidth=3,
    label="Model with all systematic pulls",
)

plt.plot(
    E_prompt_bins,
    osc_spectra_background_nominal,
    "-",
    linewidth=2,
    label="Nominal model",
)

plt.xlabel(
    r"$E_{\rm pr}$ [MeV]"
)

plt.ylabel(
    "Events per 0.1 MeV"
)

plt.title(
    "Oscillated Spectrum with Full Systematic Model"
)

plt.grid(
    True
)

plt.legend()

plt.tight_layout()


# ============================================================
# Save figure
# ============================================================

figure_path = Path(
    FIG_PATH
)

figure_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)

plt.savefig(
    figure_path,
    dpi=300,
    bbox_inches="tight",
)

print(
    f"\nSaved figure to: {figure_path}"
)

plt.show()