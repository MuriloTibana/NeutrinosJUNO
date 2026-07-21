import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from src.readDayaBay import (
    read_total_flux,
    read_covariance_matrix,
    recast_covariance_matrix,
)
from src.deltaSplines import create_delta_basis
from src.continuous_flux import build_continuous_flux_model
from src.phiHuber import phi_huber_weighted
from src.crossSectionIBD import sigma_ibd
from src.energyResponse import compute_spectrum_with_response
from src.oscillation import neutrino_oscillation
from src.background import normalizeToTable, interpolateToBins


# ============================================================
# User options
# ============================================================

# Include the reactor event-rate normalization pull.
REACTOR_NORMALIZATION = True
BACKGROUND_NORMALIZATION = True
LIHE_SHAPE_UNCERTAINTY = True

# Same seed produces the same random realization.
PULL_SEED = 123

if REACTOR_NORMALIZATION:
    FIG_PATH = "img/osc_sys_norm_bg_lihe_shape.png"
else:
    FIG_PATH = "img/osc_sys_bg_lihe_shape.png"

# ============================================================
# Random-number generator
# ============================================================
rng = np.random.default_rng(seed=PULL_SEED)

# ============================================================
# Pull helper functions
# ============================================================

def draw_standard_pull(enabled=True):
    """
    Draw a standardized Gaussian nuisance parameter:

        xi ~ N(0, 1)

    If the systematic is disabled, return zero.
    """
    if not enabled:
        return 0.0

    return float(rng.normal(loc=0.0, scale=1.0))


def draw_nonnegative_normalization_pull(
    fractional_uncertainty,
    enabled=True,
    max_tries=100000,
):
    """
    Draw a standardized pull xi ~ N(0,1), requiring

        1 + sigma * xi >= 0.

    This prevents an illustrative random realization from
    producing a negative background normalization.
    """

    if not enabled:
        return 0.0

    for _ in range(max_tries):

        xi = draw_standard_pull(enabled=True)

        factor = (1.0 + fractional_uncertainty * xi)

        if factor >= 0.0:
            return xi

    raise RuntimeError("Could not draw a nonnegative normalization pull.")


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

    The Li/He factor is

        1
        + sigma_norm * xi_norm
        + sigma_shape(E) * xi_shape,

    where

        sigma_shape(E)
        = sigma_shape_at_1mev * E / 1 MeV.

    The draw is accepted only when the factor is nonnegative
    in bins containing a nonzero nominal Li/He contribution.
    """

    active_bins = nominal_spectrum > 0.0

    if not np.any(active_bins):
        return 0.0, 0.0, np.ones_like(energy)

    shape_fraction = (sigma_shape_at_1mev * energy / 1.0)

    for _ in range(max_tries):

        xi_norm = draw_standard_pull(enabled=use_norm)
        xi_shape = draw_standard_pull(enabled=use_shape)

        factor = (1.0 + sigma_norm * xi_norm + shape_fraction * xi_shape)

        if np.all(factor[active_bins] >= 0.0):
            return xi_norm, xi_shape, factor

    raise RuntimeError(
        "Could not draw valid Li/He normalization "
        "and shape pulls.")


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
# Reactor information
# ============================================================
reactors = [
    {"name": "Taishan-1",  "P_GWth": 4.6, "L_km": 52.77},
    {"name": "Taishan-2",  "P_GWth": 4.6, "L_km": 52.64},
    {"name": "Yangjiang-1","P_GWth": 2.9, "L_km": 52.74},
    {"name": "Yangjiang-2","P_GWth": 2.9, "L_km": 52.82},
    {"name": "Yangjiang-3","P_GWth": 2.9, "L_km": 52.41},
    {"name": "Yangjiang-4","P_GWth": 2.9, "L_km": 52.49},
    {"name": "Yangjiang-5","P_GWth": 2.9, "L_km": 52.11},
    {"name": "Yangjiang-6","P_GWth": 2.9, "L_km": 52.19},
    {"name": "DayaBay-effective", "P_GWth": 17.4, "L_km": 215.0}]

reactor_data = pd.DataFrame(reactors)
reactor_data["w"] = reactor_data["P_GWth"] / (4*np.pi*reactor_data["L_km"]**2)


# ============================================================
# Huber coefficients and fission fractions
# ============================================================

alpha = {
    "U235":  np.array([4.367, -4.577, 2.100, -5.294e-1, 6.186e-2, -2.777e-3]),
    "Pu239": np.array([4.757, -5.392, 2.563, -6.596e-1, 7.820e-2, -3.536e-3]),
    "Pu241": np.array([2.990, -2.882, 1.278, -3.343e-1, 3.905e-2, -1.754e-3])}

frac = {
    "U235": 0.564,
    "U238": 0.076,
    "Pu239": 0.304,
    "Pu241": 0.056}

# ============================================================
# Build continuous Daya Bay flux model
# ============================================================

DYB_PATH = "data/DYB_unfolded_spectra_tot_U235_Pu239.txt"

df_total = read_total_flux(DYB_PATH, "Total")
C_ij = read_covariance_matrix(DYB_PATH)
Psi_ik = recast_covariance_matrix(C_ij)

Phi0 = df_total["Flux"].to_numpy(dtype=float)
E_high = df_total["E_high"].to_numpy(dtype=float)
E_low = df_total["E_low"].to_numpy(dtype=float)
E_center = df_total["E_center"].to_numpy(dtype=float)

delta, splines, I = create_delta_basis(E_center)

phi_cont, extras = build_continuous_flux_model(
    E_center, E_low, E_high,
    Phi0, Psi_ik, delta,
    phi_huber_weighted, sigma_ibd,
    frac, alpha, Delta, m_e, 500)

nbin = extras["nbin"]

# ============================================================
# Neutrino and prompt-energy grids
# ============================================================

E_nu = np.linspace(1.81, 10.0, 2000)

BIN_WIDTH = 0.1
Epr_edges = np.arange(0.0, 10.0 + BIN_WIDTH, BIN_WIDTH)
Epr_centers = 0.5 * (Epr_edges[:-1] + Epr_edges[1:])
E_prompt_bins = Epr_centers.copy()

# ============================================================
# Reactor-weighted survival probability
# ============================================================

Pee_weighted = np.zeros_like(E_nu)

for _, reactor in reactor_data.iterrows():

    L_km = reactor["L_km"]
    w = reactor["w"]
    Pee_r = neutrino_oscillation(E_nu, L_km, sin2_theta12, sin2_theta13, dm21, dm31)

    Pee_weighted += w * Pee_r
    
# ============================================================
# Evaluate nominal continuous flux
# ============================================================

xi_flux_nominal = np.zeros(
    nbin,
    dtype=float,
)

phi_E = np.asarray(
    phi_cont(
        E_nu,
        xi_flux_nominal,
    ),
    dtype=float,
).ravel()

phi_E = np.clip(
    phi_E,
    0.0,
    None,
)


# ============================================================
# Reactor spectrum before detector response
# ============================================================

sigma = sigma_ibd(
    E_nu,
    Delta,
    m_e,
)

integrand_common = (
    phi_E
    * sigma
    * Pee_weighted
)


# ============================================================
# Detector energy response
# ============================================================

NONLINEARITY_PATH = (
    "data/positron_nonlinearity.csv"
)

res_a = 0.033
res_b = 0.01
res_c = 0.0

prompt_alpha = 1.0
prompt_beta = 0.0

Epr_centers, Ni_nl = compute_spectrum_with_response(
    E_nu,
    integrand_common,
    Epr_edges,
    "nonlinear",
    True,
    res_a,
    res_b,
    res_c,
    prompt_alpha,
    prompt_beta,
    NONLINEARITY_PATH,
)


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

C_norm = np.max(
    df_JUNO["reactor_signal"]
)

osc_spectra_nominal = (
    C_norm
    * Ni_nl
)


# ============================================================
# Reactor event-rate normalization uncertainty
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

XI_REACTOR_RATE = (
    draw_nonnegative_normalization_pull(
        fractional_uncertainty=SIGMA_REACTOR_RATE,
        enabled=REACTOR_NORMALIZATION,
    )
)

reactor_rate_factor = (
    1.0
    + SIGMA_REACTOR_RATE
    * XI_REACTOR_RATE
)

osc_spectra = (
    reactor_rate_factor
    * osc_spectra_nominal
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
# Background systematic uncertainties
# ============================================================

BACKGROUND_NORM_SIGMAS = {
    "Li_He": 0.33,
    "geoneutrinos": 0.56,
    "world_reactors": 0.10,
    "bi_po": 0.56,
    "others": 1.00,
}

# Li/He shape uncertainty:
# 20% at 1 MeV, increasing linearly with prompt energy.
SIGMA_LIHE_SHAPE_AT_1MEV = 0.20


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
# Normalize background shapes to Table 1
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


# ------------------------------------------------------------
# Li/He normalization and shape uncertainty
#
# factor_i =
#
#   1
#   + sigma_norm * xi_norm
#   + sigma_shape(E_i) * xi_shape
#
# ------------------------------------------------------------

background_factors["Li_He"] = lihe_factor

background_pulled["Li_He"] = (
    lihe_factor
    * background_nominal["Li_He"]
)


# ------------------------------------------------------------
# Remaining background normalization uncertainties
# ------------------------------------------------------------

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

chi2_pull_reactor = (
    XI_REACTOR_RATE ** 2
)

chi2_pull_background_norm = sum(
    xi ** 2
    for xi in XI_BACKGROUND.values()
)

chi2_pull_lihe_shape = (
    XI_LIHE_SHAPE ** 2
)

chi2_pull_total = (
    chi2_pull_reactor
    + chi2_pull_background_norm
    + chi2_pull_lihe_shape
)


# ============================================================
# Diagnostics
# ============================================================

print("\n")
print("=" * 96)
print("SYSTEMATIC PULL SUMMARY")
print("=" * 96)

print(
    "Combined reactor event-rate uncertainty: "
    f"{100.0 * SIGMA_REACTOR_RATE:.3f}%"
)

print(
    "Reactor-rate pull: "
    f"xi_norm = {XI_REACTOR_RATE:+.4f}"
)

print(
    "Reactor-rate factor: "
    f"{reactor_rate_factor:.6f}"
)

print(
    "Applied reactor-rate shift: "
    f"{100.0 * (reactor_rate_factor - 1.0):+.3f}%"
)

print("-" * 96)

print(
    "Li/He normalization pull: "
    f"{XI_LIHE_NORM:+.4f}"
)

print(
    "Li/He shape pull: "
    f"{XI_LIHE_SHAPE:+.4f}"
)

print(
    "Li/He normalization contribution: "
    f"{100.0 * BACKGROUND_NORM_SIGMAS['Li_He'] * XI_LIHE_NORM:+.3f}%"
)

print(
    "Li/He shape uncertainty at 1 MeV: "
    f"{100.0 * SIGMA_LIHE_SHAPE_AT_1MEV * XI_LIHE_SHAPE:+.3f}%"
)

active_lihe = (
    background_nominal["Li_He"] > 0.0
)

if np.any(active_lihe):

    print(
        "Li/He factor range in active bins: "
        f"{np.min(lihe_factor[active_lihe]):.4f} "
        f"to {np.max(lihe_factor[active_lihe]):.4f}"
    )

print("-" * 96)

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

print("-" * 96)

print("Background event totals")

for name in background_nominal:

    nominal_events = np.sum(
        background_nominal[name]
    )

    pulled_events = np.sum(
        background_pulled[name]
    )

    print(
        f"{name:20s} | "
        f"{nominal_events:10.3f} "
        f"-> {pulled_events:10.3f}"
    )

print("-" * 96)

print(
    "Nominal reactor events: "
    f"{np.sum(osc_spectra_nominal):.3f}"
)

print(
    "Pulled reactor events: "
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

print("-" * 96)

print(
    "Reactor pull penalty: "
    f"{chi2_pull_reactor:.6f}"
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
    "Total pull penalty: "
    f"{chi2_pull_total:.6f}"
)

print("=" * 96)


# ============================================================
# Plot labels
# ============================================================

if REACTOR_NORMALIZATION:

    label_pulls = (
        "Reactor, background, and Li/He shape pulls"
    )

    label_nominal = (
        "Nominal reactor and backgrounds"
    )

    label_title = (
        "Reactor and Background Systematic Uncertainties"
    )

else:

    label_pulls = (
        "Nominal reactor with background pulls"
    )

    label_nominal = (
        "Nominal reactor and backgrounds"
    )

    label_title = (
        "Background Normalization and Li/He Shape Uncertainties"
    )


# ============================================================
# Plot total prediction
# ============================================================

plt.figure(
    figsize=(7.5, 4.8)
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