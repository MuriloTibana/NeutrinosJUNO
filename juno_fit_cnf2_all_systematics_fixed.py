
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.optimize import LinearConstraint, minimize
from scipy.special import erf

from src.continuous_flux import build_continuous_flux_model
from src.crossSectionIBD import sigma_ibd
from src.deltaSplines import create_delta_basis
from src.oscillation import neutrino_oscillation
from src.phiHuber import phi_huber_weighted
from src.readDayaBay import read_covariance_matrix, read_total_flux, recast_covariance_matrix


# ============================================================
# Configuration: cnf 2
# ============================================================

CONFIGURATION = "cnf2"

N_THETA12 = 45
N_DM21 = 45

SIN2_THETA12_MIN = 0.27
SIN2_THETA12_MAX = 0.35
DM21_MIN = 7.0e-5
DM21_MAX = 8.0e-5

SIN2_THETA12_REFERENCE = 0.309
DM21_REFERENCE = 7.53e-5

EPSILON = 1.0e-12
LARGE_CHI2 = 1.0e30
MIN_COMPONENT_FACTOR = 1.0e-6
NEGATIVE_SPECTRUM_PENALTY = 1.0e6

JUNO_FIRST_BIN_EDGE = 0.7

JUNO_PATH = Path("data/spect-fit.txt")
BG_PATH = Path("data/digitized_backgrounds.csv")
DYB_PATH = Path("data/DYB_unfolded_spectra_tot_U235_Pu239.txt")
NONLINEARITY_PATH = Path("data/positron_nonlinearity.csv")

RESULTS_PATH = Path(
    f"results/{N_THETA12}_{N_DM21}_data_{CONFIGURATION}_all_systematics_fixed_NEW.npz"
)
RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

# cnf 2 values from the supplied table
CHI2_TYPE = "poisson"
R_BG = 1.15
R_NL = 1.0
R_RES = 1.0

SIGMA_REACTOR_RATE = 0.018
SIGMA_ENERGY_SCALE = 0.005
SIGMA_ENERGY_BIAS = 0.0
SIGMA_ENERGY_RESOLUTION = 0.05

# Numerical optimization controls
PULL_BOUND = 5.0
SLSQP_EPS = 1.0e-5
SLSQP_FTOL = 1.0e-9
SLSQP_MAXITER_LINEAR = 500
SLSQP_MAXITER_FULL = 900
STATIONARITY_STEP = 1.0e-3
STATIONARITY_TOL = 1.0e-3

print("\nAnalysis configuration")
print("=" * 86)
print(f"Configuration                 = {CONFIGURATION}")
print(f"Data chi-square               = {CHI2_TYPE}")
print(f"r_BG                          = {R_BG:.4f}")
print(f"r_nl                          = {R_NL:.4f}")
print(f"r_res                         = {R_RES:.4f}")
print(f"Reactor-rate uncertainty      = {100.0 * SIGMA_REACTOR_RATE:.3f}%")
print(f"Energy-scale uncertainty      = {100.0 * SIGMA_ENERGY_SCALE:.3f}%")
print(f"Energy-bias uncertainty       = {100.0 * SIGMA_ENERGY_BIAS:.3f}%")
print(f"Energy-resolution uncertainty = {100.0 * SIGMA_ENERGY_RESOLUTION:.3f}%")


# ============================================================
# Physical constants and reactor configuration
# ============================================================

kg_to_MeV = 5.61e29
m_p = 1.6726219e-27 * kg_to_MeV
m_n = 1.6749275e-27 * kg_to_MeV
m_e = 9.1093837e-31 * kg_to_MeV
Delta = m_n - m_p

SIN2_THETA13 = 0.02215
DM31 = 2.513e-3

RES_A = 0.033
RES_B = 0.01
RES_C = 0.0

PROMPT_ALPHA = 1.0
PROMPT_BETA = 0.0

LIVE_DAYS = 59.1

REACTORS = [
    {"name": "Taishan-1", "P_GWth": 4.6, "L_km": 52.77},
    {"name": "Taishan-2", "P_GWth": 4.6, "L_km": 52.64},
    {"name": "Yangjiang-1", "P_GWth": 2.9, "L_km": 52.74},
    {"name": "Yangjiang-2", "P_GWth": 2.9, "L_km": 52.82},
    {"name": "Yangjiang-3", "P_GWth": 2.9, "L_km": 52.41},
    {"name": "Yangjiang-4", "P_GWth": 2.9, "L_km": 52.49},
    {"name": "Yangjiang-5", "P_GWth": 2.9, "L_km": 52.11},
    {"name": "Yangjiang-6", "P_GWth": 2.9, "L_km": 52.19},
    {"name": "DayaBay-effective", "P_GWth": 17.4, "L_km": 215.0},
]

reactor_data = pd.DataFrame(REACTORS)
reactor_data["weight"] = reactor_data["P_GWth"] / (4.0 * np.pi * reactor_data["L_km"] ** 2)

HUBER_COEFFICIENTS = {
    "U235": np.array([4.367, -4.577, 2.100, -0.5294, 0.06186, -0.002777]),
    "Pu239": np.array([4.757, -5.392, 2.563, -0.6596, 0.07820, -0.003536]),
    "Pu241": np.array([2.990, -2.882, 1.278, -0.3343, 0.03905, -0.001754]),
}

FISSION_FRACTIONS = {
    "U235": 0.564,
    "U238": 0.076,
    "Pu239": 0.304,
    "Pu241": 0.056,
}

E_NU = np.linspace(1.81, 10.0, 2000)

TRAP_WEIGHTS = np.zeros_like(E_NU)
TRAP_WEIGHTS[1:-1] = 0.5 * (E_NU[2:] - E_NU[:-2])
TRAP_WEIGHTS[0] = 0.5 * (E_NU[1] - E_NU[0])
TRAP_WEIGHTS[-1] = 0.5 * (E_NU[-1] - E_NU[-2])


# ============================================================
# Statistical functions
# ============================================================

def chi2_poisson(observed, predicted):
    observed = np.asarray(observed, dtype=float)
    predicted = np.clip(np.asarray(predicted, dtype=float), EPSILON, None)

    log_term = np.zeros_like(observed)
    positive = observed > 0.0
    log_term[positive] = observed[positive] * np.log(observed[positive] / predicted[positive])

    return 2.0 * np.sum(predicted - observed + log_term)


def chi2_cnp(observed, predicted):
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    observed_safe = np.clip(observed, EPSILON, None)
    predicted_safe = np.clip(predicted, EPSILON, None)
    variance = 3.0 / (1.0 / observed_safe + 2.0 / predicted_safe)

    return np.sum((predicted_safe - observed_safe) ** 2 / variance)


def calculate_data_chi2(observed, predicted):
    if CHI2_TYPE == "poisson":
        return chi2_poisson(observed, predicted)
    if CHI2_TYPE == "cnp":
        return chi2_cnp(observed, predicted)
    raise ValueError("CHI2_TYPE must be 'poisson' or 'cnp'.")


# ============================================================
# JUNO data and exact 66-bin edges
# ============================================================

def reconstruct_bin_edges(bin_centers, first_edge):
    bin_centers = np.asarray(bin_centers, dtype=float)
    edges = np.empty(bin_centers.size + 1, dtype=float)
    edges[0] = first_edge

    for index, center in enumerate(bin_centers):
        edges[index + 1] = 2.0 * center - edges[index]

    if not np.all(np.diff(edges) > 0.0):
        raise ValueError("Reconstructed JUNO bin edges are not strictly increasing.")

    reconstructed_centers = 0.5 * (edges[:-1] + edges[1:])

    if not np.allclose(reconstructed_centers, bin_centers, rtol=0.0, atol=1.0e-10):
        raise ValueError("The reconstructed JUNO bin edges do not reproduce the supplied centers.")

    return edges


df_JUNO = pd.read_csv(JUNO_PATH, sep=r"\s+", header=None)
df_JUNO.columns = ["energy", "reactor_signal", "reactor_background", "data", "unoscillated_signal"]
df_JUNO = df_JUNO.apply(pd.to_numeric, errors="coerce").dropna().sort_values("energy").reset_index(drop=True)

JUNO_energy = df_JUNO["energy"].to_numpy(dtype=float)
JUNO_reactor = df_JUNO["reactor_signal"].to_numpy(dtype=float)
JUNO_total = df_JUNO["reactor_background"].to_numpy(dtype=float)
JUNO_data = df_JUNO["data"].to_numpy(dtype=float)
JUNO_background = JUNO_total - JUNO_reactor

JUNO_BIN_EDGES = reconstruct_bin_edges(JUNO_energy, JUNO_FIRST_BIN_EDGE)
JUNO_BIN_WIDTHS = np.diff(JUNO_BIN_EDGES)
N_JUNO_BINS = JUNO_energy.size

if N_JUNO_BINS != 66:
    raise ValueError(f"Expected 66 JUNO bins, received {N_JUNO_BINS}.")

observed_spectrum = JUNO_data.copy()

print("\nJUNO binning")
print("=" * 86)
print(f"Number of fitted bins = {N_JUNO_BINS}")
print(f"First bin             = [{JUNO_BIN_EDGES[0]:.3f}, {JUNO_BIN_EDGES[1]:.3f}] MeV")
print(f"Last bin              = [{JUNO_BIN_EDGES[-2]:.3f}, {JUNO_BIN_EDGES[-1]:.3f}] MeV")
print(f"Minimum bin width     = {JUNO_BIN_WIDTHS.min():.3f} MeV")
print(f"Maximum bin width     = {JUNO_BIN_WIDTHS.max():.3f} MeV")


# ============================================================
# Backgrounds integrated into the exact JUNO bins
# ============================================================

TABLE1_RATES_CPD = {
    "Li_He": 4.3,
    "geoneutrinos": 2.4,
    "world_reactors": 0.88,
    "bi_po": 0.18,
    "others": 0.04 + 0.02 + 0.05 + 0.08 + 4.9e-2,
}

TABLE1_TOTAL_EVENTS = {name: rate * LIVE_DAYS for name, rate in TABLE1_RATES_CPD.items()}

BACKGROUND_UNCERTAINTIES = {
    "Li_He_norm": 0.33,
    "Li_He_shape": 0.20,
    "geoneutrinos": 0.56,
    "world_reactors": 0.10,
    "bi_po": 0.56,
    "others": 1.00,
}


def integrate_digitized_shape(E_raw, y_raw, target_edges, target_total):
    E_raw = np.asarray(E_raw, dtype=float)
    y_raw = np.asarray(y_raw, dtype=float)

    valid = np.isfinite(E_raw) & np.isfinite(y_raw)
    E_raw = E_raw[valid]
    y_raw = np.clip(y_raw[valid], 0.0, None)

    order = np.argsort(E_raw)
    E_raw = E_raw[order]
    y_raw = y_raw[order]

    E_raw, unique_indices = np.unique(E_raw, return_index=True)
    y_raw = y_raw[unique_indices]

    counts = np.zeros(target_edges.size - 1, dtype=float)

    for index, (lower, upper) in enumerate(zip(target_edges[:-1], target_edges[1:])):
        number_of_points = max(80, int(np.ceil((upper - lower) / 0.002)) + 1)
        integration_grid = np.linspace(lower, upper, number_of_points)
        interpolated_shape = np.interp(integration_grid, E_raw, y_raw, left=0.0, right=0.0)
        counts[index] = np.trapz(interpolated_shape, integration_grid)

    shape_total = counts.sum()

    if not np.isfinite(shape_total) or shape_total <= 0.0:
        raise ValueError("A digitized background shape has a nonpositive integral.")

    return counts * (target_total / shape_total)


df_raw = pd.read_csv(BG_PATH)
df_raw.columns = [str(column).strip() for column in df_raw.columns]

for column in df_raw.columns:
    df_raw[column] = pd.to_numeric(df_raw[column], errors="coerce")

df_raw = df_raw.dropna(subset=["E_prompt"]).sort_values("E_prompt")
E_raw = df_raw["E_prompt"].to_numpy(dtype=float)

background_columns = {
    "Li_He": "Li_He",
    "geoneutrinos": "geoneutrinos",
    "world_reactors": "world_reactors_digitized",
    "bi_po": "bi_po",
    "others": "others",
}

missing_columns = set(background_columns.values()).difference(df_raw.columns)

if missing_columns:
    raise KeyError(f"Missing digitized background columns: {sorted(missing_columns)}")

BACKGROUND_COMPONENTS = {
    name: integrate_digitized_shape(
        E_raw,
        df_raw[column].to_numpy(dtype=float),
        JUNO_BIN_EDGES,
        TABLE1_TOTAL_EVENTS[name],
    )
    for name, column in background_columns.items()
}

Li_He = BACKGROUND_COMPONENTS["Li_He"]
geoneutrinos = BACKGROUND_COMPONENTS["geoneutrinos"]
world_reactors = BACKGROUND_COMPONENTS["world_reactors"]
bi_po = BACKGROUND_COMPONENTS["bi_po"]
others = BACKGROUND_COMPONENTS["others"]

Total_background_nominal = (
    R_BG * Li_He
    + geoneutrinos
    + R_BG * world_reactors
    + R_BG * bi_po
    + R_BG * others
)

print("\nBackground totals before nuisance pulls")
print("=" * 86)

for name, component in BACKGROUND_COMPONENTS.items():
    print(f"{name:24s} = {component.sum():.6f} events")

print(f"{'cnf2 total after r_BG':24s} = {Total_background_nominal.sum():.6f} events")


# ============================================================
# Daya Bay continuous flux and 25 orthogonal pulls
# ============================================================

df_total = read_total_flux(DYB_PATH, "Total")
covariance = read_covariance_matrix(DYB_PATH)
psi_ik = recast_covariance_matrix(covariance)

phi0 = df_total["Flux"].to_numpy(dtype=float)
E_low = df_total["E_low"].to_numpy(dtype=float)
E_high = df_total["E_high"].to_numpy(dtype=float)
E_center = df_total["E_center"].to_numpy(dtype=float)

delta_basis, _, _ = create_delta_basis(E_center)

phi_continuous, flux_extras = build_continuous_flux_model(
    E_center,
    E_low,
    E_high,
    phi0,
    psi_ik,
    delta_basis,
    phi_huber_weighted,
    sigma_ibd,
    FISSION_FRACTIONS,
    HUBER_COEFFICIENTS,
    Delta,
    m_e,
    500,
)

N_FLUX_PULLS = int(flux_extras["nbin"])

if N_FLUX_PULLS != 25:
    raise ValueError(f"Expected 25 reactor-flux pulls, received {N_FLUX_PULLS}.")

ZERO_FLUX_PULLS = np.zeros(N_FLUX_PULLS, dtype=float)
PHI_NOMINAL_E = np.asarray(phi_continuous(E_NU, ZERO_FLUX_PULLS), dtype=float).ravel()
PHI_BASIS_E = np.empty((N_FLUX_PULLS, E_NU.size), dtype=float)

for index in range(N_FLUX_PULLS):
    unit_pull = np.zeros(N_FLUX_PULLS, dtype=float)
    unit_pull[index] = 1.0
    PHI_BASIS_E[index] = (
        np.asarray(phi_continuous(E_NU, unit_pull), dtype=float).ravel()
        - PHI_NOMINAL_E
    )

FLUX_PULL_NAMES = np.asarray([f"reactor_flux_{index + 1:02d}" for index in range(N_FLUX_PULLS)])


# ============================================================
# Nominal nonlinearity
# ============================================================

def load_nonlinearity_curve(path):
    dataframe = pd.read_csv(path)
    numeric = dataframe.apply(pd.to_numeric, errors="coerce")
    usable_columns = [column for column in numeric.columns if numeric[column].notna().sum() > 1]

    if len(usable_columns) < 2:
        numeric = pd.read_csv(path, header=None).apply(pd.to_numeric, errors="coerce")
        usable_columns = [column for column in numeric.columns if numeric[column].notna().sum() > 1]

    if len(usable_columns) < 2:
        raise ValueError("Could not identify two numerical columns in the nonlinearity file.")

    energy = numeric[usable_columns[0]].to_numpy(dtype=float)
    factor = numeric[usable_columns[1]].to_numpy(dtype=float)

    valid = np.isfinite(energy) & np.isfinite(factor)
    energy = energy[valid]
    factor = factor[valid]

    order = np.argsort(energy)
    energy = energy[order]
    factor = factor[order]

    energy, unique_indices = np.unique(energy, return_index=True)
    factor = factor[unique_indices]

    return energy, factor


E_NL_POINTS, F_NL_POINTS = load_nonlinearity_curve(NONLINEARITY_PATH)
F_NL_INTERPOLATOR = PchipInterpolator(E_NL_POINTS, F_NL_POINTS, extrapolate=False)


def F_nl(E_prompt):
    E_prompt = np.asarray(E_prompt, dtype=float)
    E_clipped = np.clip(E_prompt, E_NL_POINTS[0], E_NL_POINTS[-1])
    return np.asarray(F_NL_INTERPOLATOR(E_clipped), dtype=float)


# ============================================================
# Pull layout
# ============================================================

IDX_REACTOR_NORM = 0
IDX_LIHE_NORM = 1
IDX_LIHE_SHAPE = 2
IDX_GEO = 3
IDX_WORLD = 4
IDX_BIPO = 5
IDX_OTHER = 6
IDX_ENERGY_SCALE = 7
IDX_ENERGY_RESOLUTION = 8
IDX_FLUX_START = 9
IDX_FLUX_STOP = IDX_FLUX_START + N_FLUX_PULLS

BASE_PULL_NAMES = np.asarray([
    "reactor_norm",
    "Li_He_norm",
    "Li_He_shape",
    "geoneutrinos",
    "world_reactors",
    "bi_po",
    "others",
])

DETECTOR_PULL_NAMES = np.asarray(["energy_scale", "energy_resolution"])
BACKGROUND_PULL_NAMES = BASE_PULL_NAMES[1:].copy()
PULL_NAMES = np.concatenate([BASE_PULL_NAMES, DETECTOR_PULL_NAMES, FLUX_PULL_NAMES])
N_PULLS = PULL_NAMES.size

if N_PULLS != 34:
    raise RuntimeError(f"cnf2 should contain 34 pulls, but the layout contains {N_PULLS}.")


# ============================================================
# Bounds and smooth linear positivity constraints
# ============================================================

PULL_BOUNDS = [(-PULL_BOUND, PULL_BOUND)] * N_PULLS

# Simple normalization factors receive physical lower bounds.
PULL_BOUNDS[IDX_GEO] = (max(-PULL_BOUND, -(1.0 - MIN_COMPONENT_FACTOR) / 0.56), PULL_BOUND)
PULL_BOUNDS[IDX_WORLD] = (max(-PULL_BOUND, -(1.0 - MIN_COMPONENT_FACTOR) / 0.10), PULL_BOUND)
PULL_BOUNDS[IDX_BIPO] = (max(-PULL_BOUND, -(1.0 - MIN_COMPONENT_FACTOR) / 0.56), PULL_BOUND)
PULL_BOUNDS[IDX_OTHER] = (max(-PULL_BOUND, -(1.0 - MIN_COMPONENT_FACTOR) / 1.00), PULL_BOUND)

# Li/He has a coupled normalization-plus-shape condition:
# 1 + 0.33 z_norm + 0.20 E_i z_shape >= MIN_COMPONENT_FACTOR.
constraint_matrix = np.zeros((N_JUNO_BINS, N_PULLS), dtype=float)
constraint_matrix[:, IDX_LIHE_NORM] = BACKGROUND_UNCERTAINTIES["Li_He_norm"]
constraint_matrix[:, IDX_LIHE_SHAPE] = BACKGROUND_UNCERTAINTIES["Li_He_shape"] * JUNO_energy

constraint_lower = np.full(N_JUNO_BINS, -1.0 + MIN_COMPONENT_FACTOR)
constraint_upper = np.full(N_JUNO_BINS, np.inf)

FULL_LINEAR_CONSTRAINT = LinearConstraint(
    constraint_matrix,
    constraint_lower,
    constraint_upper,
)

LINEAR_PULL_INDICES = np.concatenate([
    np.arange(IDX_REACTOR_NORM, IDX_ENERGY_SCALE),
    np.arange(IDX_FLUX_START, IDX_FLUX_STOP),
])

LINEAR_PULL_INDICES = LINEAR_PULL_INDICES.astype(int)
LINEAR_BOUNDS = [PULL_BOUNDS[index] for index in LINEAR_PULL_INDICES]

LINEAR_CONSTRAINT = LinearConstraint(
    constraint_matrix[:, LINEAR_PULL_INDICES],
    constraint_lower,
    constraint_upper,
)


# ============================================================
# Background prediction
# ============================================================

def calculate_background_with_pulls(pulls, return_components=False):
    lihe_factor = (
        1.0
        + BACKGROUND_UNCERTAINTIES["Li_He_norm"] * pulls[IDX_LIHE_NORM]
        + BACKGROUND_UNCERTAINTIES["Li_He_shape"] * JUNO_energy * pulls[IDX_LIHE_SHAPE]
    )

    components = {
        "Li_He": R_BG * lihe_factor * Li_He,
        "geoneutrinos": (
            1.0 + BACKGROUND_UNCERTAINTIES["geoneutrinos"] * pulls[IDX_GEO]
        ) * geoneutrinos,
        "world_reactors": R_BG * (
            1.0 + BACKGROUND_UNCERTAINTIES["world_reactors"] * pulls[IDX_WORLD]
        ) * world_reactors,
        "bi_po": R_BG * (
            1.0 + BACKGROUND_UNCERTAINTIES["bi_po"] * pulls[IDX_BIPO]
        ) * bi_po,
        "others": R_BG * (
            1.0 + BACKGROUND_UNCERTAINTIES["others"] * pulls[IDX_OTHER]
        ) * others,
    }

    total = sum(components.values())

    if return_components:
        return total, components

    return total


# ============================================================
# Detector response directly into the exact 66 JUNO bins
# ============================================================

def sigma_prompt(E_prompt):
    E_prompt = np.asarray(E_prompt, dtype=float)
    E_safe = np.clip(E_prompt, EPSILON, None)
    return np.sqrt(RES_A**2 * E_safe + RES_B**2 * E_safe**2 + RES_C**2)


def compute_response_matrix(z_scale=0.0, z_resolution=0.0):
    E_visible = E_NU - Delta + m_e
    E_prompt_0 = PROMPT_ALPHA * E_visible + PROMPT_BETA

    xi_scale = SIGMA_ENERGY_SCALE * z_scale
    xi_resolution = SIGMA_ENERGY_RESOLUTION * z_resolution

    mu = E_prompt_0 * R_NL * (1.0 + xi_scale) * F_nl(E_prompt_0)
    resolution_factor = (1.0 + xi_resolution) * R_RES

    if resolution_factor <= 0.0:
        raise ValueError("The resolution pull produced a nonpositive response width.")

    sigma_E = np.clip(resolution_factor * sigma_prompt(mu), EPSILON, None)

    lower_edges = JUNO_BIN_EDGES[:-1, None]
    upper_edges = JUNO_BIN_EDGES[1:, None]
    mu_matrix = mu[None, :]
    sigma_matrix = sigma_E[None, :]

    z_upper = (upper_edges - mu_matrix) / (np.sqrt(2.0) * sigma_matrix)
    z_lower = (lower_edges - mu_matrix) / (np.sqrt(2.0) * sigma_matrix)

    response_matrix = 0.5 * (erf(z_upper) - erf(z_lower))
    response_matrix[:, E_visible <= 0.0] = 0.0

    return response_matrix


# ============================================================
# Reactor probability and flux templates
# ============================================================

def calculate_weighted_survival_probability(sin2_theta12, dm21):
    weighted_probability = np.zeros_like(E_NU)

    for _, reactor in reactor_data.iterrows():
        probability = neutrino_oscillation(
            E_NU,
            reactor["L_km"],
            sin2_theta12,
            SIN2_THETA13,
            dm21,
            DM31,
        )

        weighted_probability += reactor["weight"] * probability

    return weighted_probability / reactor_data["weight"].sum()


def calculate_weighted_integrands(sin2_theta12, dm21):
    probability = calculate_weighted_survival_probability(sin2_theta12, dm21)
    common = sigma_ibd(E_NU, Delta, m_e) * probability * TRAP_WEIGHTS

    nominal_weighted = PHI_NOMINAL_E * common
    basis_weighted = PHI_BASIS_E * common[None, :]

    return nominal_weighted, basis_weighted


def project_reactor_components(nominal_weighted, basis_weighted, z_scale, z_resolution):
    response_matrix = compute_response_matrix(z_scale, z_resolution)
    nominal_raw = response_matrix @ nominal_weighted
    templates_raw = (response_matrix @ basis_weighted.T).T

    return nominal_raw, templates_raw


# ============================================================
# Fixed normalization on all 66 bins
# ============================================================

reference_nominal_weighted, reference_basis_weighted = calculate_weighted_integrands(
    SIN2_THETA12_REFERENCE,
    DM21_REFERENCE,
)

reference_response = compute_response_matrix(0.0, 0.0)
reference_raw = reference_response @ reference_nominal_weighted

if reference_raw.sum() <= 0.0:
    raise ValueError("The reference reactor spectrum has a nonpositive total.")

REFERENCE_NORMALIZATION = JUNO_reactor.sum() / reference_raw.sum()

print("\nFixed reactor normalization")
print("=" * 86)
print(f"Raw reference total       = {reference_raw.sum():.8e}")
print(f"JUNO reactor total        = {JUNO_reactor.sum():.6f}")
print(f"Reference normalization   = {REFERENCE_NORMALIZATION:.8e}")
print(f"Normalized model total    = {(REFERENCE_NORMALIZATION * reference_raw).sum():.6f}")


# ============================================================
# Oscillation scan arrays
# ============================================================

sin2_theta12_grid = np.linspace(SIN2_THETA12_MIN, SIN2_THETA12_MAX, N_THETA12)
dm21_grid = np.linspace(DM21_MIN, DM21_MAX, N_DM21)

chi2_grid = np.full((N_DM21, N_THETA12), np.nan, dtype=float)
pull_grid = np.full((N_DM21, N_THETA12, N_PULLS), np.nan, dtype=float)
success_grid = np.zeros((N_DM21, N_THETA12), dtype=bool)
nfev_grid = np.full((N_DM21, N_THETA12), -1, dtype=int)
constraint_violation_grid = np.full((N_DM21, N_THETA12), np.nan, dtype=float)
stationarity_improvement_grid = np.full((N_DM21, N_THETA12), np.nan, dtype=float)
optimizer_message_grid = np.full((N_DM21, N_THETA12), "", dtype="<U160")

total_points = N_THETA12 * N_DM21
completed_points = 0


# ============================================================
# Profile cnf2 nuisance parameters
# ============================================================

for i_dm, dm21_test in enumerate(dm21_grid):
    for i_theta, sin2_theta12_test in enumerate(sin2_theta12_grid):
        nominal_weighted, basis_weighted = calculate_weighted_integrands(
            sin2_theta12_test,
            dm21_test,
        )

        detector_cache = {}

        def get_projected_reactor(pulls):
            z_scale = float(pulls[IDX_ENERGY_SCALE])
            z_resolution = float(pulls[IDX_ENERGY_RESOLUTION])
            cache_key = (round(z_scale, 8), round(z_resolution, 8))

            if cache_key not in detector_cache:
                if len(detector_cache) >= 128:
                    detector_cache.clear()

                nominal_raw, templates_raw = project_reactor_components(
                    nominal_weighted,
                    basis_weighted,
                    z_scale,
                    z_resolution,
                )

                detector_cache[cache_key] = (
                    REFERENCE_NORMALIZATION * nominal_raw,
                    REFERENCE_NORMALIZATION * templates_raw,
                )

            return detector_cache[cache_key]

        def chi2_for_this_point(pulls):
            pulls = np.asarray(pulls, dtype=float)

            if pulls.shape != (N_PULLS,) or not np.all(np.isfinite(pulls)):
                return LARGE_CHI2

            reactor_factor = 1.0 + SIGMA_REACTOR_RATE * pulls[IDX_REACTOR_NORM]

            if reactor_factor <= 0.0:
                return LARGE_CHI2

            try:
                reactor_nominal, reactor_templates = get_projected_reactor(pulls)
            except (FloatingPointError, ValueError):
                return LARGE_CHI2

            flux_pulls = pulls[IDX_FLUX_START:IDX_FLUX_STOP]
            reactor_shape = reactor_nominal + flux_pulls @ reactor_templates

            # Do not reject infinitesimal Daya Bay pull variations because the
            # continuous basis can become slightly negative outside its support.
            # Penalize only a negative binned reactor prediction.
            negative_reactor = np.minimum(reactor_shape, 0.0)
            negative_reactor_penalty = (
                NEGATIVE_SPECTRUM_PENALTY * np.sum(negative_reactor**2)
            )

            reactor_prediction = reactor_factor * reactor_shape
            background_prediction = calculate_background_with_pulls(pulls)
            prediction = reactor_prediction + background_prediction

            negative_total = np.minimum(prediction, 0.0)
            negative_total_penalty = (
                NEGATIVE_SPECTRUM_PENALTY * np.sum(negative_total**2)
            )

            prediction_safe = np.clip(prediction, EPSILON, None)

            chi2_data = calculate_data_chi2(observed_spectrum, prediction_safe)
            chi2_pull = np.sum(pulls**2)

            total_chi2 = (
                chi2_data
                + chi2_pull
                + negative_reactor_penalty
                + negative_total_penalty
            )

            return total_chi2 if np.isfinite(total_chi2) else LARGE_CHI2

        def full_from_linear(linear_pulls, detector_reference):
            full_pulls = detector_reference.copy()
            full_pulls[LINEAR_PULL_INDICES] = linear_pulls
            return full_pulls

        if i_theta > 0 and success_grid[i_dm, i_theta - 1]:
            initial_pulls = pull_grid[i_dm, i_theta - 1].copy()
        elif i_dm > 0 and success_grid[i_dm - 1, i_theta]:
            initial_pulls = pull_grid[i_dm - 1, i_theta].copy()
        else:
            initial_pulls = np.zeros(N_PULLS, dtype=float)

        # Stage 1: profile all linear pulls with detector pulls held at the
        # warm-start values. This prevents the 34-dimensional line search from
        # declaring the all-zero vector converged.
        detector_reference = initial_pulls.copy()

        linear_result = minimize(
            lambda linear_pulls: chi2_for_this_point(
                full_from_linear(linear_pulls, detector_reference)
            ),
            x0=initial_pulls[LINEAR_PULL_INDICES],
            method="SLSQP",
            bounds=LINEAR_BOUNDS,
            constraints=[LINEAR_CONSTRAINT],
            options={
                "ftol": SLSQP_FTOL,
                "eps": SLSQP_EPS,
                "maxiter": SLSQP_MAXITER_LINEAR,
                "disp": False,
            },
        )

        staged_initial = initial_pulls.copy()

        if np.isfinite(linear_result.fun):
            staged_initial[LINEAR_PULL_INDICES] = linear_result.x

        # Stage 2: release the detector-response pulls and profile all 34 pulls.
        fit_result = minimize(
            chi2_for_this_point,
            x0=staged_initial,
            method="SLSQP",
            bounds=PULL_BOUNDS,
            constraints=[FULL_LINEAR_CONSTRAINT],
            options={
                "ftol": SLSQP_FTOL,
                "eps": SLSQP_EPS,
                "maxiter": SLSQP_MAXITER_FULL,
                "disp": False,
            },
        )

        candidate = fit_result.x.copy()
        candidate_chi2 = float(fit_result.fun)

        # Check whether simple coordinate perturbations can still lower chi2.
        check_indices = [
            IDX_REACTOR_NORM,
            IDX_LIHE_NORM,
            IDX_LIHE_SHAPE,
            IDX_GEO,
            IDX_ENERGY_SCALE,
            IDX_ENERGY_RESOLUTION,
            IDX_FLUX_START,
            IDX_FLUX_START + N_FLUX_PULLS // 2,
            IDX_FLUX_STOP - 1,
        ]

        best_directional_chi2 = candidate_chi2

        for index in check_indices:
            for sign in (-1.0, 1.0):
                trial = candidate.copy()
                trial[index] += sign * STATIONARITY_STEP

                lower, upper = PULL_BOUNDS[index]

                if trial[index] < lower or trial[index] > upper:
                    continue

                constraint_values = constraint_matrix @ trial

                if np.any(constraint_values < constraint_lower):
                    continue

                trial_chi2 = chi2_for_this_point(trial)
                best_directional_chi2 = min(best_directional_chi2, trial_chi2)

        stationarity_improvement = candidate_chi2 - best_directional_chi2

        # A second SLSQP pass with a larger differentiation step repairs rare
        # cases where numerical derivatives are too small.
        if stationarity_improvement > STATIONARITY_TOL:
            retry_result = minimize(
                chi2_for_this_point,
                x0=candidate,
                method="SLSQP",
                bounds=PULL_BOUNDS,
                constraints=[FULL_LINEAR_CONSTRAINT],
                options={
                    "ftol": 1.0e-10,
                    "eps": 1.0e-4,
                    "maxiter": 1400,
                    "disp": False,
                },
            )

            if np.isfinite(retry_result.fun) and retry_result.fun < candidate_chi2:
                fit_result = retry_result
                candidate = retry_result.x.copy()
                candidate_chi2 = float(retry_result.fun)

        constraint_violation = float(
            np.max(np.maximum(constraint_lower - constraint_matrix @ candidate, 0.0))
        )

        point_success = (
            np.isfinite(candidate_chi2)
            and constraint_violation <= 1.0e-7
            and candidate_chi2 < LARGE_CHI2
        )

        chi2_grid[i_dm, i_theta] = candidate_chi2
        pull_grid[i_dm, i_theta] = candidate
        success_grid[i_dm, i_theta] = point_success
        nfev_grid[i_dm, i_theta] = getattr(fit_result, "nfev", -1)
        constraint_violation_grid[i_dm, i_theta] = constraint_violation
        stationarity_improvement_grid[i_dm, i_theta] = stationarity_improvement
        optimizer_message_grid[i_dm, i_theta] = str(fit_result.message)[:160]

        completed_points += 1
        flux_norm = np.linalg.norm(candidate[IDX_FLUX_START:IDX_FLUX_STOP])

        print(
            f"Completed {completed_points:4d}/{total_points} | "
            f"theta={sin2_theta12_test:.6f} | "
            f"dm21={dm21_test:.6e} | "
            f"chi2={candidate_chi2:.6f} | "
            f"z_norm={candidate[IDX_REACTOR_NORM]:+.4f} | "
            f"z_LiHe_norm={candidate[IDX_LIHE_NORM]:+.4f} | "
            f"z_LiHe_shape={candidate[IDX_LIHE_SHAPE]:+.4f} | "
            f"z_scale={candidate[IDX_ENERGY_SCALE]:+.4f} | "
            f"z_res={candidate[IDX_ENERGY_RESOLUTION]:+.4f} | "
            f"||z_flux||={flux_norm:.4f} | "
            f"success={point_success}"
        )


# ============================================================
# Best-fit point
# ============================================================

valid_grid = success_grid & np.isfinite(chi2_grid)

if not np.any(valid_grid):
    raise RuntimeError("No oscillation-grid point produced a valid fit.")

chi2_for_minimum = np.where(valid_grid, chi2_grid, np.inf)
best_flat_index = np.argmin(chi2_for_minimum)
best_dm_index, best_theta_index = np.unravel_index(best_flat_index, chi2_grid.shape)

best_sin2_theta12 = sin2_theta12_grid[best_theta_index]
best_dm21 = dm21_grid[best_dm_index]
best_chi2 = chi2_grid[best_dm_index, best_theta_index]
best_pulls = pull_grid[best_dm_index, best_theta_index].copy()

best_z_norm = best_pulls[IDX_REACTOR_NORM]
best_background_pulls = best_pulls[IDX_LIHE_NORM:IDX_ENERGY_SCALE].copy()
best_detector_pulls = best_pulls[IDX_ENERGY_SCALE:IDX_FLUX_START].copy()
best_flux_pulls = best_pulls[IDX_FLUX_START:IDX_FLUX_STOP].copy()

best_z_scale = best_pulls[IDX_ENERGY_SCALE]
best_z_resolution = best_pulls[IDX_ENERGY_RESOLUTION]
best_reactor_factor = 1.0 + SIGMA_REACTOR_RATE * best_z_norm

delta_chi2_grid = chi2_grid - best_chi2
z_norm_grid = pull_grid[:, :, IDX_REACTOR_NORM].copy()
background_pull_grid = pull_grid[:, :, IDX_LIHE_NORM:IDX_ENERGY_SCALE].copy()
detector_pull_grid = pull_grid[:, :, IDX_ENERGY_SCALE:IDX_FLUX_START].copy()
flux_pull_grid = pull_grid[:, :, IDX_FLUX_START:IDX_FLUX_STOP].copy()


# ============================================================
# Reconstruct best-fit spectra and chi-square decomposition
# ============================================================

best_nominal_weighted, best_basis_weighted = calculate_weighted_integrands(
    best_sin2_theta12,
    best_dm21,
)

best_flux_weighted = best_nominal_weighted + best_flux_pulls @ best_basis_weighted

response_nominal = compute_response_matrix(0.0, 0.0)
response_detector = compute_response_matrix(best_z_scale, best_z_resolution)

spectrum_best_nominal = REFERENCE_NORMALIZATION * (response_nominal @ best_nominal_weighted)
spectrum_best_flux_only = REFERENCE_NORMALIZATION * (response_nominal @ best_flux_weighted)
spectrum_best_detector_only = REFERENCE_NORMALIZATION * (response_detector @ best_nominal_weighted)
spectrum_best_flux_detector = REFERENCE_NORMALIZATION * (response_detector @ best_flux_weighted)

best_reactor_nominal = spectrum_best_nominal.copy()
best_reactor_flux_only = spectrum_best_flux_only.copy()
best_reactor_detector_only = spectrum_best_detector_only.copy()
best_reactor_flux_detector = spectrum_best_flux_detector.copy()
best_reactor_prediction = best_reactor_factor * best_reactor_flux_detector

best_background_prediction, best_background_components = calculate_background_with_pulls(
    best_pulls,
    return_components=True,
)

best_background_nominal = Total_background_nominal.copy()
best_total_prediction = best_reactor_prediction + best_background_prediction

best_chi2_data = calculate_data_chi2(observed_spectrum, best_total_prediction)
best_chi2_reactor_norm_pull = best_z_norm**2
best_chi2_background_pulls = np.sum(best_background_pulls**2)
best_chi2_detector_pulls = np.sum(best_detector_pulls**2)
best_chi2_flux_pulls = np.sum(best_flux_pulls**2)
best_chi2_pull = np.sum(best_pulls**2)


# ============================================================
# Diagnostics
# ============================================================

zero_pull_points = np.sum(np.linalg.norm(pull_grid, axis=2) < 1.0e-10)

print("\nBest-fit cnf 2 result")
print("=" * 92)
print(f"sin²(theta12)                    = {best_sin2_theta12:.8f}")
print(f"Delta m²21                       = {best_dm21:.8e} eV²")
print(f"Minimum chi²                     = {best_chi2:.6f}")
print(f"Data contribution                = {best_chi2_data:.6f}")
print(f"Pull contribution                = {best_chi2_pull:.6f}")
print(f"  reactor normalization          = {best_chi2_reactor_norm_pull:.6f}")
print(f"  background pulls               = {best_chi2_background_pulls:.6f}")
print(f"  detector pulls                 = {best_chi2_detector_pulls:.6f}")
print(f"  25 reactor-flux pulls          = {best_chi2_flux_pulls:.6f}")
print(f"Successful grid points           = {success_grid.sum()}/{success_grid.size}")
print(f"All-zero pull grid points        = {zero_pull_points}/{success_grid.size}")
print(f"Reactor physical rate shift      = {100.0 * SIGMA_REACTOR_RATE * best_z_norm:+.4f}%")
print(f"Energy-scale physical shift      = {100.0 * SIGMA_ENERGY_SCALE * best_z_scale:+.4f}%")
print(f"Energy-resolution physical shift = {100.0 * SIGMA_ENERGY_RESOLUTION * best_z_resolution:+.4f}%")

print("\nBest-fit non-flux pulls")
print("=" * 92)

for name, value in zip(PULL_NAMES[:IDX_FLUX_START], best_pulls[:IDX_FLUX_START]):
    print(f"{name:30s} = {value:+.6f} sigma")

print("\nBest-fit flux pulls")
print("=" * 92)

for name, value in zip(FLUX_PULL_NAMES, best_flux_pulls):
    print(f"{name:30s} = {value:+.6f} sigma")


# ============================================================
# Save
# ============================================================

np.savez_compressed(
    RESULTS_PATH,

    configuration=np.asarray(CONFIGURATION),
    fit_juno=np.asarray("data"),
    chi2_type=np.asarray(CHI2_TYPE),

    r_background=R_BG,
    r_nonlinearity=R_NL,
    r_resolution=R_RES,

    JUNO_bin_edges=JUNO_BIN_EDGES,
    JUNO_bin_widths=JUNO_BIN_WIDTHS,
    JUNO_energy=JUNO_energy,
    observed_spectrum=observed_spectrum,
    JUNO_data=JUNO_data,
    JUNO_total=JUNO_total,
    JUNO_reactor=JUNO_reactor,
    JUNO_background=JUNO_background,

    sin2_theta12_grid=sin2_theta12_grid,
    dm21_grid=dm21_grid,
    chi2_grid=chi2_grid,
    delta_chi2_grid=delta_chi2_grid,
    success_grid=success_grid,
    nfev_grid=nfev_grid,
    optimizer_message_grid=optimizer_message_grid,
    constraint_violation_grid=constraint_violation_grid,
    stationarity_improvement_grid=stationarity_improvement_grid,

    pull_names=PULL_NAMES,
    pull_grid=pull_grid,
    best_pulls=best_pulls,

    z_norm_grid=z_norm_grid,
    best_z_norm=best_z_norm,
    best_reactor_factor=best_reactor_factor,
    sigma_reactor_rate=SIGMA_REACTOR_RATE,

    background_pull_names=BACKGROUND_PULL_NAMES,
    background_pull_grid=background_pull_grid,
    best_background_pulls=best_background_pulls,

    detector_pull_names=DETECTOR_PULL_NAMES,
    detector_pull_grid=detector_pull_grid,
    best_detector_pulls=best_detector_pulls,
    best_z_energy_scale=best_z_scale,
    best_z_energy_bias=0.0,
    best_z_energy_resolution=best_z_resolution,
    sigma_energy_scale=SIGMA_ENERGY_SCALE,
    sigma_energy_bias=SIGMA_ENERGY_BIAS,
    sigma_energy_resolution=SIGMA_ENERGY_RESOLUTION,

    reactor_flux_pull_names=FLUX_PULL_NAMES,
    reactor_flux_pull_grid=flux_pull_grid,
    best_reactor_flux_pulls=best_flux_pulls,

    best_sin2_theta12=best_sin2_theta12,
    best_dm21=best_dm21,
    best_chi2=best_chi2,
    best_chi2_data=best_chi2_data,
    best_chi2_pull=best_chi2_pull,
    best_chi2_reactor_norm_pull=best_chi2_reactor_norm_pull,
    best_chi2_background_pulls=best_chi2_background_pulls,
    best_chi2_detector_pulls=best_chi2_detector_pulls,
    best_chi2_flux_pulls=best_chi2_flux_pulls,

    spectrum_best_nominal=spectrum_best_nominal,
    spectrum_best_flux_only=spectrum_best_flux_only,
    spectrum_best_detector_only=spectrum_best_detector_only,
    spectrum_best_flux_detector=spectrum_best_flux_detector,

    best_reactor_nominal=best_reactor_nominal,
    best_reactor_flux_only=best_reactor_flux_only,
    best_reactor_detector_only=best_reactor_detector_only,
    best_reactor_flux_detector=best_reactor_flux_detector,
    best_reactor_prediction=best_reactor_prediction,

    best_background_nominal=best_background_nominal,
    best_background_prediction=best_background_prediction,
    best_total_prediction=best_total_prediction,

    best_Li_He=best_background_components["Li_He"],
    best_geoneutrinos=best_background_components["geoneutrinos"],
    best_world_reactors=best_background_components["world_reactors"],
    best_bi_po=best_background_components["bi_po"],
    best_others=best_background_components["others"],

    reference_normalization=REFERENCE_NORMALIZATION,
    n_reactor_flux_pulls=N_FLUX_PULLS,
    n_total_pulls=N_PULLS,
    zero_pull_points=zero_pull_points,
)

print(f"\nResults saved to {RESULTS_PATH}")
