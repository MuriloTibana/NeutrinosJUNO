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

# Overall reactor event-rate normalization pull.
REACTOR_NORMALIZATION = True

# 25 reactor-flux covariance pulls.
REACTOR_FLUX_UNCERTAINTY = True

# Background normalization pulls.
BACKGROUND_NORMALIZATION = True

# Energy-dependent Li/He shape pull.
LIHE_SHAPE_UNCERTAINTY = True

# Prompt-energy scale pull xi_scl.
ENERGY_SCALE_UNCERTAINTY = True

# Additive bias in the nonlinear correction xi_bias.
#
# The reference notes that xi_scl and xi_bias have nearly the
# same spectral effect, so analyses normally activate only one.
ENERGY_BIAS_UNCERTAINTY = False

# Energy-resolution pull xi_res. This rescales the Gaussian
# detector width without shifting its mean.
ENERGY_RESOLUTION_UNCERTAINTY = True

# Same seed gives the same systematic realization.
PULL_SEED = 123

FIG_PATH = "img/osc_sys_norm_bg_lihe_flux_energy_resolution.png"


# ============================================================
# Random-number generator
# ============================================================

rng = np.random.default_rng(seed=PULL_SEED)


# ============================================================
# Pull helper functions
# ============================================================

def draw_standard_pull(enabled=True):
    """
    Draw a standardized Gaussian pull

        xi ~ N(0, 1).

    If the systematic is disabled, return zero.
    """

    if not enabled:
        return 0.0

    return float(
        rng.normal(
            loc=0.0,
            scale=1.0,
        )
    )


def draw_nonnegative_normalization_pull(
    fractional_uncertainty,
    enabled=True,
    max_tries=100000,
):
    """
    Draw xi ~ N(0,1), requiring

        1 + sigma * xi >= 0.

    This prevents negative normalization factors in
    illustrative random spectra.
    """

    if not enabled:
        return 0.0

    for _ in range(max_tries):

        xi = draw_standard_pull(enabled=True)

        factor = (
            1.0
            + fractional_uncertainty * xi
        )

        if factor >= 0.0:
            return xi

    raise RuntimeError(
        "Could not draw a nonnegative normalization pull."
    )


def draw_valid_lihe_pulls(
    energy,
    nominal_spectrum,
    sigma_norm,
    sigma_shape_at_1mev,
    use_norm=True,
    use_shape=True,
    max_tries=100000,
):
    """
    Draw independent Li/He normalization and shape pulls:

        xi_norm  ~ N(0,1)
        xi_shape ~ N(0,1)

    with multiplicative factor

        1
        + sigma_norm * xi_norm
        + sigma_shape(E) * xi_shape,

    where

        sigma_shape(E)
        = sigma_shape_at_1mev * E / 1 MeV.
    """

    energy = np.asarray(
        energy,
        dtype=float,
    )

    nominal_spectrum = np.asarray(
        nominal_spectrum,
        dtype=float,
    )

    active_bins = nominal_spectrum > 0.0

    if not np.any(active_bins):
        return 0.0, 0.0, np.ones_like(energy)

    shape_fraction = (
        sigma_shape_at_1mev
        * energy
        / 1.0
    )

    for _ in range(max_tries):

        xi_norm = draw_standard_pull(
            enabled=use_norm
        )

        xi_shape = draw_standard_pull(
            enabled=use_shape
        )

        factor = (
            1.0
            + sigma_norm * xi_norm
            + shape_fraction * xi_shape
        )

        if np.all(factor[active_bins] >= 0.0):
            return xi_norm, xi_shape, factor

    raise RuntimeError(
        "Could not draw valid Li/He pulls."
    )



def load_nonlinearity_interpolator(csv_path):
    """
    Read the positron nonlinearity correction F_nl(E).

    The routine accepts either a CSV with headers or a simple
    two-column CSV. Values outside the tabulated range are held
    at the nearest endpoint rather than extrapolated wildly.
    """

    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Could not find nonlinearity file: {csv_path}"
        )

    table = pd.read_csv(csv_path)
    numeric = table.apply(
        pd.to_numeric,
        errors="coerce",
    )

    usable_columns = [
        column
        for column in numeric.columns
        if numeric[column].notna().sum() >= 2
    ]

    if len(usable_columns) < 2:
        table = pd.read_csv(
            csv_path,
            header=None,
        )
        numeric = table.apply(
            pd.to_numeric,
            errors="coerce",
        )
        usable_columns = [
            column
            for column in numeric.columns
            if numeric[column].notna().sum() >= 2
        ]

    if len(usable_columns) < 2:
        raise ValueError(
            "The nonlinearity CSV must contain at least "
            "two numeric columns: energy and F_nl."
        )

    energy_column = usable_columns[0]
    factor_column = usable_columns[1]

    energy = numeric[
        energy_column
    ].to_numpy(dtype=float)

    factor = numeric[
        factor_column
    ].to_numpy(dtype=float)

    good = (
        np.isfinite(energy)
        & np.isfinite(factor)
    )

    energy = energy[good]
    factor = factor[good]

    order = np.argsort(energy)
    energy = energy[order]
    factor = factor[order]

    energy, unique_indices = np.unique(
        energy,
        return_index=True,
    )
    factor = factor[unique_indices]

    if energy.size < 2:
        raise ValueError(
            "The nonlinearity table needs at least "
            "two distinct energy points."
        )

    interpolator = PchipInterpolator(
        energy,
        factor,
        extrapolate=False,
    )

    def F_nl(energy_query):
        energy_query = np.asarray(
            energy_query,
            dtype=float,
        )

        energy_clipped = np.clip(
            energy_query,
            energy[0],
            energy[-1],
        )

        return np.asarray(
            interpolator(energy_clipped),
            dtype=float,
        )

    return F_nl, energy, factor


def gaussian_bin_probabilities(
    bin_edges,
    mean_energy,
    sigma_energy,
):
    """
    Gaussian probability for each reconstructed prompt-energy
    bin and each neutrino-energy point.

    Output shape:

        (number of prompt bins, number of neutrino points)
    """

    bin_edges = np.asarray(
        bin_edges,
        dtype=float,
    )

    mean_energy = np.asarray(
        mean_energy,
        dtype=float,
    )

    sigma_energy = np.asarray(
        sigma_energy,
        dtype=float,
    )

    sigma_energy = np.maximum(
        sigma_energy,
        1.0e-12,
    )

    z_low = (
        bin_edges[:-1, None]
        - mean_energy[None, :]
    ) / (
        np.sqrt(2.0)
        * sigma_energy[None, :]
    )

    z_high = (
        bin_edges[1:, None]
        - mean_energy[None, :]
    ) / (
        np.sqrt(2.0)
        * sigma_energy[None, :]
    )

    probabilities = 0.5 * (
        erf(z_high)
        - erf(z_low)
    )

    return np.clip(
        probabilities,
        0.0,
        1.0,
    )


def compute_spectrum_with_energy_scale_pulls(
    neutrino_energy,
    neutrino_integrand,
    prompt_edges,
    nonlinearity_function,
    xi_scale,
    xi_bias,
    xi_resolution,
    sigma_scale,
    sigma_bias,
    sigma_resolution,
    resolution_a,
    resolution_b,
    resolution_c,
    calibration_alpha=1.0,
    calibration_beta=0.0,
):
    r"""
    Fold the neutrino spectrum into reconstructed prompt energy.

    The energy-scale model is

        E_pr_tilde
        = E_pr * [
            (1 + sigma_scale * xi_scale) F_nl(E_pr)
            + sigma_bias * xi_bias
          ].

    Here xi_scale and xi_bias are standardized pulls:

        xi_scale, xi_bias ~ N(0, 1).

    Therefore sigma_scale * xi_scale and
    sigma_bias * xi_bias are the physical fractional shifts.

    The energy-resolution uncertainty is implemented as

        sigma_E_tilde
        = (1 + sigma_resolution * xi_resolution) sigma_E,

    where xi_resolution is also a standardized N(0,1) pull.
    """

    neutrino_energy = np.asarray(
        neutrino_energy,
        dtype=float,
    )

    neutrino_integrand = np.asarray(
        neutrino_integrand,
        dtype=float,
    )

    prompt_edges = np.asarray(
        prompt_edges,
        dtype=float,
    )

    if neutrino_energy.shape != neutrino_integrand.shape:
        raise ValueError(
            "neutrino_energy and neutrino_integrand "
            "must have the same shape."
        )

    # IBD visible prompt energy before detector nonlinearity:
    #
    # E_prompt = E_e + m_e
    #          = E_nu - Delta + m_e.
    prompt_energy = (
        neutrino_energy
        - Delta
        + m_e
    )

    prompt_energy = (
        calibration_alpha
        * prompt_energy
        + calibration_beta
    )

    F_nl = nonlinearity_function(
        prompt_energy
    )

    fractional_scale_shift = (
        sigma_scale
        * xi_scale
    )

    fractional_bias_shift = (
        sigma_bias
        * xi_bias
    )

    nonlinear_correction = (
        (1.0 + fractional_scale_shift)
        * F_nl
        + fractional_bias_shift
    )

    if np.any(nonlinear_correction <= 0.0):
        raise ValueError(
            "The energy-scale pulls produced a nonpositive "
            "nonlinearity correction."
        )

    reconstructed_mean = (
        prompt_energy
        * nonlinear_correction
    )

    # sigma_E / E = sqrt(a^2/E + b^2 + c^2/E^2)
    #
    # Therefore:
    #
    # sigma_E^2 = a^2 E + b^2 E^2 + c^2.
    resolution_variance = (
        resolution_a ** 2
        * reconstructed_mean
        + resolution_b ** 2
        * reconstructed_mean ** 2
        + resolution_c ** 2
    )

    resolution_sigma = np.sqrt(
        np.clip(
            resolution_variance,
            1.0e-24,
            None,
        )
    )

    # Energy-resolution pull:
    #
    #     sigma_E -> (1 + sigma_res * xi_res) sigma_E.
    #
    # This changes only the Gaussian width. The reconstructed
    # mean energy remains controlled by the scale/bias pulls.
    resolution_factor = (
        1.0
        + sigma_resolution
        * xi_resolution
    )

    if resolution_factor <= 0.0:
        raise ValueError(
            "The energy-resolution pull produced a "
            "nonpositive Gaussian width factor."
        )

    resolution_sigma = (
        resolution_factor
        * resolution_sigma
    )

    response_probabilities = (
        gaussian_bin_probabilities(
            prompt_edges,
            reconstructed_mean,
            resolution_sigma,
        )
    )

    prompt_spectrum = np.trapz(
        response_probabilities
        * neutrino_integrand[None, :],
        x=neutrino_energy,
        axis=1,
    )

    prompt_centers = 0.5 * (
        prompt_edges[:-1]
        + prompt_edges[1:]
    )

    return (
        prompt_centers,
        prompt_spectrum,
        reconstructed_mean,
        resolution_sigma,
        F_nl,
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
# Oscillation parameters
# ============================================================

sin2_theta12 = 0.308
sin2_theta13 = 0.02215

dm21 = 7.49e-5
dm31 = 2.513e-3


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
            list(
                REACTOR_RATE_UNCERTAINTIES.values()
            ),
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


# Prompt-energy scale and nonlinear-bias uncertainties.
#
# Both are 0.5%, as stated in the reference. The pulls themselves
# remain standardized N(0,1) variables.
SIGMA_ENERGY_SCALE = 0.005
SIGMA_ENERGY_BIAS = 0.005


# Energy-resolution uncertainty. The reference adopts 5%.
# With a standardized pull xi_res ~ N(0,1), the physical
# resolution-width factor is
#
#     1 + SIGMA_ENERGY_RESOLUTION * xi_res.
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

reactor_data = pd.DataFrame(reactors)

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
# Build the continuous Daya Bay flux model
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

E_low = df_total[
    "E_low"
].to_numpy(dtype=float)

E_high = df_total[
    "E_high"
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
    f"Number of reactor-flux pulls: {nbin}"
)

if nbin != 25:
    print(
        "Warning: the loaded Daya Bay model contains "
        f"{nbin} pulls rather than 25."
    )


# ============================================================
# Energy grids
# ============================================================

E_nu = np.linspace(
    1.81,
    10.0,
    2000,
)

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


# ============================================================
# Reactor-weighted survival probability
# ============================================================

Pee_weighted = np.zeros_like(
    E_nu
)

for _, reactor in reactor_data.iterrows():

    L_km = reactor["L_km"]
    weight = reactor["w"]

    Pee_reactor = neutrino_oscillation(
        E_nu,
        L_km,
        sin2_theta12,
        sin2_theta13,
        dm21,
        dm31,
    )

    Pee_weighted += (
        weight
        * Pee_reactor
    )


sigma_IBD = sigma_ibd(
    E_nu,
    Delta,
    m_e,
)


# ============================================================
# Detector energy-response settings
# ============================================================

NONLINEARITY_PATH = (
    "data/positron_nonlinearity.csv"
)

res_a = 0.033
res_b = 0.01
res_c = 0.0

prompt_alpha = 1.0
prompt_beta = 0.0


F_NL, F_NL_ENERGY_POINTS, F_NL_VALUES = (
    load_nonlinearity_interpolator(
        NONLINEARITY_PATH
    )
)


if (
    ENERGY_SCALE_UNCERTAINTY
    and ENERGY_BIAS_UNCERTAINTY
):
    print(
        "Warning: both energy-scale pulls are active. "
        "The reference normally uses only one at a time "
        "because their spectral effects are very similar."
    )


# ============================================================
# Draw the reactor pulls
# ============================================================

# Overall reactor event-rate normalization pull.
XI_REACTOR_RATE = (
    draw_nonnegative_normalization_pull(
        fractional_uncertainty=SIGMA_REACTOR_RATE,
        enabled=REACTOR_NORMALIZATION,
    )
)


# Prompt-energy scale and nonlinear-bias pulls.
#
# The physical shifts entering Eq. (2.4) are
#
#     delta_scl  = SIGMA_ENERGY_SCALE * XI_ENERGY_SCALE
#     delta_bias = SIGMA_ENERGY_BIAS  * XI_ENERGY_BIAS.
XI_ENERGY_SCALE = draw_standard_pull(
    enabled=ENERGY_SCALE_UNCERTAINTY
)

XI_ENERGY_BIAS = draw_standard_pull(
    enabled=ENERGY_BIAS_UNCERTAINTY
)


# Energy-resolution pull. Rejection sampling guarantees that
# the multiplicative Gaussian-width factor remains nonnegative.
XI_ENERGY_RESOLUTION = (
    draw_nonnegative_normalization_pull(
        fractional_uncertainty=SIGMA_ENERGY_RESOLUTION,
        enabled=ENERGY_RESOLUTION_UNCERTAINTY,
    )
)


# The 25 covariance pulls are standard-normal variables.
#
# No extra sigma is required because Psi_ik already contains
# the uncertainty sizes and correlations.
if REACTOR_FLUX_UNCERTAINTY:

    XI_FLUX = rng.normal(
        loc=0.0,
        scale=1.0,
        size=nbin,
    )

else:

    XI_FLUX = np.zeros(
        nbin,
        dtype=float,
    )


XI_FLUX_NOMINAL = np.zeros(
    nbin,
    dtype=float,
)


# ============================================================
# Reactor prompt-spectrum function
# ============================================================

def compute_reactor_prompt_spectrum(
    xi_flux,
    xi_energy_scale=0.0,
    xi_energy_bias=0.0,
    xi_energy_resolution=0.0,
):
    """
    Compute the reactor prompt spectrum for a given set of
    Daya Bay flux pulls and prompt-energy response pulls.

    The continuous flux model performs

        phi(E, xi)
        = phi_0(E) + sum_k psi_k(E) xi_k.

    The detector response then performs

        E_pr_tilde
        = E_pr [
            (1 + sigma_scl xi_scl) F_nl(E_pr)
            + sigma_bias xi_bias
          ].

    The Gaussian detector width is simultaneously changed by

        sigma_E_tilde
        = (1 + sigma_res xi_res) sigma_E.
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

    # Prevent small negative flux values caused by extreme
    # random covariance fluctuations.
    phi_E = np.clip(
        phi_E,
        0.0,
        None,
    )

    integrand = (
        phi_E
        * sigma_IBD
        * Pee_weighted
    )

    (
        energy_centers,
        prompt_spectrum,
        reconstructed_mean,
        resolution_sigma,
        F_nl_values,
    ) = compute_spectrum_with_energy_scale_pulls(
        neutrino_energy=E_nu,
        neutrino_integrand=integrand,
        prompt_edges=Epr_edges,
        nonlinearity_function=F_NL,
        xi_scale=xi_energy_scale,
        xi_bias=xi_energy_bias,
        xi_resolution=xi_energy_resolution,
        sigma_scale=SIGMA_ENERGY_SCALE,
        sigma_bias=SIGMA_ENERGY_BIAS,
        sigma_resolution=SIGMA_ENERGY_RESOLUTION,
        resolution_a=res_a,
        resolution_b=res_b,
        resolution_c=res_c,
        calibration_alpha=prompt_alpha,
        calibration_beta=prompt_beta,
    )

    return (
        np.asarray(
            energy_centers,
            dtype=float,
        ),
        np.asarray(
            prompt_spectrum,
            dtype=float,
        ),
        phi_E,
        np.asarray(
            reconstructed_mean,
            dtype=float,
        ),
        np.asarray(
            resolution_sigma,
            dtype=float,
        ),
        np.asarray(
            F_nl_values,
            dtype=float,
        ),
    )


# ============================================================
# Nominal and systematically pulled reactor spectra
# ============================================================

(
    Epr_centers_nominal,
    reactor_raw_nominal,
    phi_E_nominal,
    reconstructed_mean_nominal,
    resolution_sigma_nominal,
    F_nl_nominal,
) = compute_reactor_prompt_spectrum(
    XI_FLUX_NOMINAL,
    xi_energy_scale=0.0,
    xi_energy_bias=0.0,
    xi_energy_resolution=0.0,
)


(
    Epr_centers_pulled,
    reactor_raw_pulled,
    phi_E_pulled,
    reconstructed_mean_pulled,
    resolution_sigma_pulled,
    F_nl_pulled,
) = compute_reactor_prompt_spectrum(
    XI_FLUX,
    xi_energy_scale=XI_ENERGY_SCALE,
    xi_energy_bias=XI_ENERGY_BIAS,
    xi_energy_resolution=XI_ENERGY_RESOLUTION,
)


if not np.allclose(
    Epr_centers_nominal,
    Epr_centers_pulled,
):
    raise ValueError(
        "Nominal and pulled prompt-energy grids differ."
    )

E_prompt_bins = Epr_centers_nominal.copy()


# ============================================================
# Load JUNO reference spectrum
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


juno_reactor_signal = df_JUNO[
    "reactor_signal"
].to_numpy(dtype=float)


# ============================================================
# Fix absolute normalization using the nominal model
# ============================================================

model_peak = np.max(
    reactor_raw_nominal
)

juno_peak = np.max(
    juno_reactor_signal
)

if model_peak <= 0.0:
    raise ValueError(
        "The nominal reactor spectrum has a "
        "nonpositive maximum."
    )

# Using a ratio makes the normalization correct whether or not
# the response routine returns a unit-normalized curve.
C_norm = (
    juno_peak
    / model_peak
)


# The same fixed C_norm must be used for both spectra.
# Otherwise part of the flux-pull effect would be normalized away.
osc_spectra_nominal = (
    C_norm
    * reactor_raw_nominal
)

osc_spectra_systematics_pulled = (
    C_norm
    * reactor_raw_pulled
)


# ============================================================
# Apply the reactor event-rate normalization pull
# ============================================================

reactor_rate_factor = (
    1.0
    + SIGMA_REACTOR_RATE
    * XI_REACTOR_RATE
)

osc_spectra = (
    reactor_rate_factor
    * osc_spectra_systematics_pulled
)


# ============================================================
# Background event rates
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
    for name, rate in TABLE1_RATES_CPD.items()
}


# ============================================================
# Load digitized background shapes
# ============================================================

BG_PATH = "data/digitized_backgrounds.csv"

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
# Normalize the backgrounds to Table 1
# ============================================================

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


background_nominal = {
    "Li_He": Li_He,
    "geoneutrinos": geoneutrinos,
    "world_reactors": world_reactors,
    "bi_po": bi_po,
    "others": others,
}


# ============================================================
# Draw Li/He normalization and shape pulls
# ============================================================

(
    XI_LIHE_NORM,
    XI_LIHE_SHAPE,
    lihe_factor,
) = draw_valid_lihe_pulls(
    energy=E_prompt_bins,
    nominal_spectrum=background_nominal["Li_He"],
    sigma_norm=BACKGROUND_NORM_SIGMAS["Li_He"],
    sigma_shape_at_1mev=SIGMA_LIHE_SHAPE_AT_1MEV,
    use_norm=BACKGROUND_NORMALIZATION,
    use_shape=LIHE_SHAPE_UNCERTAINTY,
)


# ============================================================
# Draw remaining background normalization pulls
# ============================================================

XI_BACKGROUND = {
    "Li_He": XI_LIHE_NORM,
}

for name in [
    "geoneutrinos",
    "world_reactors",
    "bi_po",
    "others",
]:

    XI_BACKGROUND[name] = (
        draw_nonnegative_normalization_pull(
            fractional_uncertainty=(
                BACKGROUND_NORM_SIGMAS[name]
            ),
            enabled=BACKGROUND_NORMALIZATION,
        )
    )


# ============================================================
# Apply background pulls
# ============================================================

background_pulled = {}
background_factors = {}


# Li/He normalization plus energy-dependent shape pull.
background_factors["Li_He"] = (
    lihe_factor
)

background_pulled["Li_He"] = (
    lihe_factor
    * background_nominal["Li_He"]
)


# Remaining background normalization pulls.
for name in [
    "geoneutrinos",
    "world_reactors",
    "bi_po",
    "others",
]:

    sigma_bg = BACKGROUND_NORM_SIGMAS[name]
    xi_bg = XI_BACKGROUND[name]

    normalization_factor = (
        1.0
        + sigma_bg * xi_bg
    )

    background_factors[name] = (
        normalization_factor
    )

    background_pulled[name] = (
        normalization_factor
        * background_nominal[name]
    )


# ============================================================
# Total background spectra
# ============================================================

Total_Background_nominal = np.zeros_like(
    E_prompt_bins,
    dtype=float,
)

Total_Background = np.zeros_like(
    E_prompt_bins,
    dtype=float,
)

for name in background_nominal:

    Total_Background_nominal += (
        background_nominal[name]
    )

    Total_Background += (
        background_pulled[name]
    )


# ============================================================
# Complete nominal and pulled predictions
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
# Pull chi-square penalties
# ============================================================

chi2_pull_reactor_rate = (
    XI_REACTOR_RATE ** 2
)

chi2_pull_flux = np.sum(
    XI_FLUX ** 2
)

chi2_pull_background_norm = sum(
    xi ** 2
    for xi in XI_BACKGROUND.values()
)

chi2_pull_lihe_shape = (
    XI_LIHE_SHAPE ** 2
)

chi2_pull_energy_scale = (
    XI_ENERGY_SCALE ** 2
)

chi2_pull_energy_bias = (
    XI_ENERGY_BIAS ** 2
)

chi2_pull_energy_resolution = (
    XI_ENERGY_RESOLUTION ** 2
)

chi2_pull_total = (
    chi2_pull_reactor_rate
    + chi2_pull_flux
    + chi2_pull_background_norm
    + chi2_pull_lihe_shape
    + chi2_pull_energy_scale
    + chi2_pull_energy_bias
    + chi2_pull_energy_resolution
)


# ============================================================
# Diagnostics
# ============================================================

print("\n")
print("=" * 100)
print("SYSTEMATIC PULL SUMMARY")
print("=" * 100)

print(
    "Combined reactor event-rate uncertainty: "
    f"{100.0 * SIGMA_REACTOR_RATE:.3f}%"
)

print(
    "Reactor-rate pull: "
    f"{XI_REACTOR_RATE:+.6f}"
)

print(
    "Reactor-rate factor: "
    f"{reactor_rate_factor:.6f}"
)

print(
    "Applied reactor-rate shift: "
    f"{100.0 * (reactor_rate_factor - 1.0):+.3f}%"
)

print("-" * 100)

energy_scale_shift = (
    SIGMA_ENERGY_SCALE
    * XI_ENERGY_SCALE
)

energy_bias_shift = (
    SIGMA_ENERGY_BIAS
    * XI_ENERGY_BIAS
)

energy_resolution_shift = (
    SIGMA_ENERGY_RESOLUTION
    * XI_ENERGY_RESOLUTION
)

energy_resolution_factor = (
    1.0
    + energy_resolution_shift
)

print(
    "Energy-scale pull xi_scl: "
    f"{XI_ENERGY_SCALE:+.6f}"
)

print(
    "Physical energy-scale shift: "
    f"{100.0 * energy_scale_shift:+.4f}%"
)

print(
    "Nonlinearity-bias pull xi_bias: "
    f"{XI_ENERGY_BIAS:+.6f}"
)

print(
    "Physical nonlinearity-bias shift: "
    f"{100.0 * energy_bias_shift:+.4f}%"
)

print(
    "Energy-resolution pull xi_res: "
    f"{XI_ENERGY_RESOLUTION:+.6f}"
)

print(
    "Physical resolution-width shift: "
    f"{100.0 * energy_resolution_shift:+.4f}%"
)

print(
    "Resolution-width factor: "
    f"{energy_resolution_factor:.6f}"
)

active_prompt = (
    (E_nu - Delta + m_e) > 0.0
)

if np.any(active_prompt):

    energy_ratio = (
        reconstructed_mean_pulled[active_prompt]
        / reconstructed_mean_nominal[active_prompt]
    )

    print(
        "Pulled/nominal reconstructed-energy ratio: "
        f"{np.min(energy_ratio):.6f} "
        f"to {np.max(energy_ratio):.6f}"
    )

    resolution_ratio = (
        resolution_sigma_pulled[active_prompt]
        / resolution_sigma_nominal[active_prompt]
    )

    print(
        "Pulled/nominal resolution-width ratio: "
        f"{np.min(resolution_ratio):.6f} "
        f"to {np.max(resolution_ratio):.6f}"
    )

print("-" * 100)

print(
    f"Number of reactor-flux pulls: {len(XI_FLUX)}"
)

print(
    "Flux-pull RMS: "
    f"{np.sqrt(np.mean(XI_FLUX**2)):.6f}"
)

print(
    "Flux-pull minimum: "
    f"{np.min(XI_FLUX):+.6f}"
)

print(
    "Flux-pull maximum: "
    f"{np.max(XI_FLUX):+.6f}"
)

print("\nIndividual reactor-flux pulls")

for index, xi_flux in enumerate(
    XI_FLUX,
    start=1,
):
    print(
        f"xi_flux[{index:02d}] = "
        f"{xi_flux:+.6f}"
    )

print("-" * 100)

print(
    "Li/He normalization pull: "
    f"{XI_LIHE_NORM:+.6f}"
)

print(
    "Li/He shape pull: "
    f"{XI_LIHE_SHAPE:+.6f}"
)

print(
    "Li/He normalization contribution: "
    f"{100.0 * BACKGROUND_NORM_SIGMAS['Li_He'] * XI_LIHE_NORM:+.3f}%"
)

print(
    "Li/He shape contribution at 1 MeV: "
    f"{100.0 * SIGMA_LIHE_SHAPE_AT_1MEV * XI_LIHE_SHAPE:+.3f}%"
)

active_lihe = (
    background_nominal["Li_He"] > 0.0
)

if np.any(active_lihe):

    print(
        "Li/He factor range in active bins: "
        f"{np.min(lihe_factor[active_lihe]):.6f} "
        f"to {np.max(lihe_factor[active_lihe]):.6f}"
    )

print("-" * 100)

print("Background normalization pulls")

for name in [
    "geoneutrinos",
    "world_reactors",
    "bi_po",
    "others",
]:

    xi_bg = XI_BACKGROUND[name]
    sigma_bg = BACKGROUND_NORM_SIGMAS[name]
    factor_bg = background_factors[name]

    print(
        f"{name:20s} | "
        f"xi = {xi_bg:+8.4f} | "
        f"sigma = {100.0 * sigma_bg:6.2f}% | "
        f"factor = {factor_bg:8.4f}"
    )

print("-" * 100)

print("Event totals")

print(
    "Nominal reactor events: "
    f"{np.sum(osc_spectra_nominal):.3f}"
)

print(
    "Flux/energy-pulled reactor events before rate pull: "
    f"{np.sum(osc_spectra_systematics_pulled):.3f}"
)

print(
    "Fully pulled reactor events: "
    f"{np.sum(osc_spectra):.3f}"
)

print(
    "Nominal background events: "
    f"{np.sum(Total_Background_nominal):.3f}"
)

print(
    "Pulled background events: "
    f"{np.sum(Total_Background):.3f}"
)

print(
    "Nominal total events: "
    f"{np.sum(osc_spectra_background_nominal):.3f}"
)

print(
    "Pulled total events: "
    f"{np.sum(osc_spectra_background):.3f}"
)

print("-" * 100)

print("Pull penalties")

print(
    "Reactor-rate penalty: "
    f"{chi2_pull_reactor_rate:.6f}"
)

print(
    "Reactor-flux penalty: "
    f"{chi2_pull_flux:.6f}"
)

print(
    "Background normalization penalty: "
    f"{chi2_pull_background_norm:.6f}"
)

print(
    "Li/He shape penalty: "
    f"{chi2_pull_lihe_shape:.6f}"
)

print(
    "Energy-scale penalty: "
    f"{chi2_pull_energy_scale:.6f}"
)

print(
    "Nonlinearity-bias penalty: "
    f"{chi2_pull_energy_bias:.6f}"
)

print(
    "Energy-resolution penalty: "
    f"{chi2_pull_energy_resolution:.6f}"
)

print(
    "Total pull penalty: "
    f"{chi2_pull_total:.6f}"
)

print("=" * 100)


# ============================================================
# Plot labels
# ============================================================

active_systematics = []

if REACTOR_NORMALIZATION:
    active_systematics.append(
        "reactor normalization"
    )

if REACTOR_FLUX_UNCERTAINTY:
    active_systematics.append(
        "25 flux"
    )

if BACKGROUND_NORMALIZATION:
    active_systematics.append(
        "background normalization"
    )

if LIHE_SHAPE_UNCERTAINTY:
    active_systematics.append(
        "Li/He shape"
    )

if ENERGY_SCALE_UNCERTAINTY:
    active_systematics.append(
        "energy scale"
    )

if ENERGY_BIAS_UNCERTAINTY:
    active_systematics.append(
        "nonlinearity bias"
    )

if ENERGY_RESOLUTION_UNCERTAINTY:
    active_systematics.append(
        "energy resolution"
    )

if active_systematics:

    label_pulls = (
        "Model with "
        + ", ".join(active_systematics)
        + " pulls"
    )

else:

    label_pulls = "Nominal model"

label_nominal = (
    "Nominal reactor and backgrounds"
)

label_title = (
    "Oscillated Spectrum with Systematic Uncertainties"
)


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
    color="darkgoldenrod",
    linewidth=3,
    label=label_pulls,
)

plt.plot(
    E_prompt_bins,
    osc_spectra_background_nominal,
    "-",
    color="darkorange",
    linewidth=2,
    label=label_nominal,
)

plt.xlabel(
    r"$E_{\rm pr}$ [MeV]"
)

plt.ylabel(
    "Events per 0.1 MeV"
)

plt.title(
    label_title
)

plt.grid(
    True,
    alpha=0.3,
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