#!/usr/bin/env python3
from __future__ import annotations

# ============================================================
# Corrected JUNO Figure-2-style fit using cnf 1 only
#
# This script:
#   1. Builds a JUNO-like reactor spectrum
#   2. Includes the full systematic-pull model
#   3. Uses cnf 1 only:
#
#        cnf 1: CNP chi2, r_BG = 1.00
#
#   4. Scans sin^2(theta12) and Delta m21^2
#   5. Profiles over all nuisance pulls at each scan point
#   6. Produces a Fig. 2-like figure:
#        left: cnf 1 contours in solar-parameter space
#        right: cnf 1 best-fit spectrum
#
# Run from project root:
#
#   python configuration_cnf1_only.py
# ============================================================

from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.special import erf
from scipy.interpolate import PchipInterpolator
from scipy.signal import savgol_filter
from scipy.optimize import minimize
from matplotlib.lines import Line2D


# ============================================================
# Project path
# ============================================================
from src.readDayaBay import read_total_flux, read_covariance_matrix, recast_covariance_matrix
from src.deltaSplines import create_delta_basis
from src.continuous_flux import build_continuous_flux_model
from src.phiHuber import phi_huber_weighted
from src.crossSectionIBD import sigma_ibd

# ============================================================
# User options
# ============================================================
BACKGROUND_SOURCE = "digitized"
# Options:
#   "digitized"
#   "juno_reference"

RUN_GLOBAL_BEST_FIT = True
RUN_SOLAR_SCAN = True

# Start with 25 x 25 while testing. Use 45 x 45 or higher for a smoother paper-style contour.
# Final contours need a reasonably fine grid.
N_SIN2_POINTS = 10
N_DM21_POINTS = 10

FIT_MAXITER_GLOBAL = 160
FIT_MAXITER_SCAN = 200

# Figure 2-style plotting/scanning window.
SIN2_THETA12_RANGE = (0.26, 0.36)
DM21_RANGE = (6.8e-5, 8.2e-5)

# Prompt-energy bins included in the fit.
FIT_ENERGY_MIN = 0.8
FIT_ENERGY_MAX = 10.0

# Fit both smooth JUNO reference curves:
#   reactor_signal      -> reactor-only prediction
#   reactor_background  -> reactor + backgrounds prediction
#
# The right-hand spectrum plot displays both smooth JUNO reference curves:
#   reactor_signal      -> lower black dashed line
#   reactor_background  -> upper black dashed line
# IMPORTANT:
# reactor_signal and reactor_background are not independent data sets.
# Fitting both double-counts the same reactor events and makes the
# confidence regions artificially small.
FIT_SMOOTH_JUNO_REACTOR = False
FIT_SMOOTH_JUNO_TOTAL = True

# Use the measured JUNO data column for the likelihood.  The smooth
# reactor/background curves remain available for plotting only.
FIT_TO_MEASURED_DATA = True

REACTOR_SPECTRUM_WEIGHT = 0.0
TOTAL_SPECTRUM_WEIGHT = 1.0

# Robust profiling controls.  Retrying from more than one pull seed
# helps prevent failed/local nuisance fits from shrinking contours.
PROFILE_MULTI_START = True
PROFILE_MAX_STARTS = 3

# 1.0 gives the standard Gaussian nuisance penalty sum(xi_k^2).
# Set to 0.0 only for a strict curve-closure diagnostic.
PULL_PENALTY_WEIGHT = 1.0

# Correlated-Gaussian JUNO plotting reference used only when an
# exact external JUNO Delta-chi-square grid is unavailable.
JUNO_BEST_SIN2_THETA12 = 0.3092
JUNO_SIGMA_SIN2_THETA12 = 0.0087
JUNO_BEST_DM21 = 7.50e-5
JUNO_SIGMA_DM21 = 0.12e-5
JUNO_CORRELATION = -0.23

PULL_BOUNDS = (-5.0, 5.0)

SAVE_RESULTS = True
SAVE_FIGURES = True

CONFIG_COMPARISON_FIGURE_PATH = "cnf1_fit.png"
CONFIG_COMPARISON_RESULTS_PATH = "cnf1_fit.npz"
# ============================================================
# Detector and prompt-energy settings
# ============================================================
res_a = 0.033
res_b = 0.01

# The response is evaluated on a fine true-neutrino-energy grid.
E_nu = np.linspace(1.806, 10.0, 2000)

# Use the exact JUNO reconstructed-energy centers from spect-fit.txt.
# x_model, Epr_edges, Epr_centers, and bin_width are defined after
# loading that file. This avoids interpolating measured event counts.
x_model: np.ndarray
Epr_edges: np.ndarray
Epr_centers: np.ndarray
bin_width: float

# Physics refinements used in the paper-style model.
USE_MATTER_EFFECTS = True
MATTER_DENSITY_G_CM3 = 2.55
ELECTRON_FRACTION = 0.50

# The paper notes that the dominant neutron-recoil effect is captured
# accurately by using the midpoint of the exact allowed positron-energy
# interval. This is much faster than integrating the full top-hat at
# every nuisance-fit evaluation.
RECOIL_MODE = "midpoint"
# Options:
#   "midpoint"  -> exact two-body endpoints, then use their midpoint
#   "none"      -> recoilless E_e = E_nu - Delta

# ============================================================
# Digitized background settings
# ============================================================
LIVE_DAYS = 59.1

# JUNO Table-1 pre-fit rates in counts/day.
TABLE1_RATE_CPD = {
    "LiHe": 4.30,
    "Geo": 1.20,
    "World": 0.88,
    "BiPo": 0.18,
    "Other": 0.04 + 0.02 + 0.05 + 0.08 + 0.049,
}

# Expected paper data diagnostics. The script warns, rather than
# silently rescaling the observed counts, when another digitization is used.
EXPECTED_JUNO_BINS = 66
EXPECTED_JUNO_DATA_TOTAL = 2379.0
DATA_TOTAL_TOLERANCE = 1.0

# ============================================================
# Nominal oscillation parameters
# ============================================================
sin2_theta12_nominal = 0.308
sin2_theta13_nominal = 0.02200

dm21_nominal = 7.49e-5
dm31 = 2.513e-3


# ============================================================
# Constants
# ============================================================

kg_to_MeV = 5.61e29

m_p = 1.6726219e-27 * kg_to_MeV
m_n = 1.6749275e-27 * kg_to_MeV
m_e = 9.1093837e-31 * kg_to_MeV

Delta = m_n - m_p


# ============================================================
# Base systematic uncertainties
# ============================================================

SIGMA_REACTOR_NORM = 0.018

SIGMA_BG_NORM = {
    "LiHe": 0.33,
    "Geo": 0.42,
    "World": 0.10,
    "BiPo": 0.56,
    "Other": 1.00,
}

SIGMA_LIHE_SHAPE_AT_1MEV = 0.20

SIGMA_ENERGY_SCALE = 0.005
SIGMA_SIN2_THETA13 = 0.00056

# These two are configuration-dependent.
SIGMA_ENERGY_BIAS = 0.0
SIGMA_ENERGY_RES = 0.05

ORTHOGONALIZE_LIHE_SHAPE_TO_NORM = False


# ============================================================
# Analysis configurations
# ============================================================

CONFIGURATIONS = [
    {
        "key": "cnf1",
        "label": "cnf 1",
        "chi2_kind": "cnp",
        "r_bg": 1.00,
        "r_nl": 1.00,
        "sigma_bias": 0.00,
        "r_res": 1.00,
        "sigma_res": 0.05,
    },
]

CONFIG_COLORS = {
    "cnf1": "magenta",
}

CHI2_KIND = "cnp"
CONFIG_R_NL = 1.0
CONFIG_R_RES = 1.0


# ============================================================
# Utility functions
# ============================================================
def normalize_column_name(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def find_digitized_column(df: pd.DataFrame, possible_names: list[str]) -> str | None:
    normalized_columns = {
        normalize_column_name(col): col
        for col in df.columns
    }

    for name in possible_names:
        key = normalize_column_name(name)
        if key in normalized_columns:
            return normalized_columns[key]

    for col_norm, col_original in normalized_columns.items():
        for name in possible_names:
            key = normalize_column_name(name)
            if key in col_norm or col_norm in key:
                return col_original

    return None


def interpolate_digitized_component(
    df: pd.DataFrame,
    x_source: np.ndarray,
    x_target: np.ndarray,
    possible_names: list[str],
    label: str,
    allow_missing: bool = True,
) -> np.ndarray:
    col = find_digitized_column(df, possible_names)

    if col is None:
        if allow_missing:
            print(f"Warning: no digitized column found for {label}. Using zeros.")
            return np.zeros_like(x_target, dtype=float)

        raise KeyError(
            f"No digitized column found for {label}. "
            f"Available columns: {list(df.columns)}"
        )

    y_source = df[col].to_numpy(dtype=float)
    y_source = np.nan_to_num(y_source, nan=0.0)

    y_interp = np.interp(
        x_target,
        x_source,
        y_source,
        left=0.0,
        right=0.0,
    )

    y_interp = np.clip(y_interp, 0.0, None)

    print(f"{label:12s}: using digitized column '{col}'")

    return y_interp


def positive_scale(sigma: float, xi: float) -> float:
    return max(0.0, 1.0 + sigma * xi)


# ============================================================
# Load JUNO reference spectrum and measured data
# ============================================================

total_t0 = perf_counter()

JUNO_PATH = "data/spect-fit.txt"

df_JUNO = pd.read_csv(JUNO_PATH, sep=r"\s+", header=None)
df_JUNO.columns = [
    "energy",
    "reactor_signal",
    "reactor_background",
    "data",
    "unoscillated_signal",
]

df_JUNO = (
    df_JUNO
    .replace([np.inf, -np.inf], np.nan)
    .dropna()
    .sort_values("energy")
    .reset_index(drop=True)
)

E_juno = df_JUNO["energy"].to_numpy(dtype=float)
juno_react = df_JUNO["reactor_signal"].to_numpy(dtype=float)
juno_noosc = df_JUNO["unoscillated_signal"].to_numpy(dtype=float)
juno_react_bk = df_JUNO["reactor_background"].to_numpy(dtype=float)
juno_data = df_JUNO["data"].to_numpy(dtype=float)


def centers_to_edges(centers: np.ndarray) -> np.ndarray:
    """Construct bin edges from strictly increasing bin centers."""
    centers = np.asarray(centers, dtype=float)

    if centers.ndim != 1 or centers.size < 2:
        raise ValueError("At least two one-dimensional bin centers are required.")

    spacing = np.diff(centers)
    if np.any(spacing <= 0.0):
        raise ValueError("JUNO energy centers must be strictly increasing.")

    edges = np.empty(centers.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - 0.5 * spacing[0]
    edges[-1] = centers[-1] + 0.5 * spacing[-1]

    return edges


# No interpolation of measured counts: the likelihood uses the exact rows.
x_model = E_juno.copy()
Epr_centers = x_model.copy()
Epr_edges = centers_to_edges(Epr_centers)
bin_width = float(np.median(np.diff(Epr_centers)))

# Keep the old names as direct aliases so the remainder of the script is clear.
juno_react_interp = juno_react.copy()
juno_noosc_interp = juno_noosc.copy()
juno_react_bk_interp = juno_react_bk.copy()
juno_data_interp = juno_data.copy()

fit_target_for_mask = (
    juno_data_interp
    if FIT_TO_MEASURED_DATA
    else juno_react_bk_interp
)

fit_mask = (
    np.isfinite(fit_target_for_mask)
    & (x_model >= FIT_ENERGY_MIN)
    & (x_model <= FIT_ENERGY_MAX)
    & (fit_target_for_mask >= 0.0)
)

print("\nJUNO data diagnostics:")
print(f"  rows in spect-fit.txt = {len(E_juno)}")
print(f"  measured event total  = {np.sum(juno_data):.6f}")
print(f"  fitted rows            = {np.count_nonzero(fit_mask)}")
print(f"  median bin spacing     = {bin_width:.6f} MeV")

if len(E_juno) != EXPECTED_JUNO_BINS:
    print(
        "WARNING: the paper fit used 66 reconstructed-energy points, "
        f"but this file contains {len(E_juno)}."
    )

if abs(np.sum(juno_data) - EXPECTED_JUNO_DATA_TOTAL) > DATA_TOTAL_TOLERANCE:
    print(
        "WARNING: the paper reports 2379 measured events, but this data "
        f"column sums to {np.sum(juno_data):.3f}. Contours cannot match "
        "exactly until the same digitized data set is used."
    )

# ============================================================
# Reactor model
# ============================================================

reactors = [
    {"name": "Taishan-1", "P_GWth": 4.6, "L_km": 52.77},
    {"name": "Taishan-2", "P_GWth": 4.6, "L_km": 52.64},
    {"name": "Yangjiang-1", "P_GWth": 2.9, "L_km": 52.74},
    {"name": "Yangjiang-2", "P_GWth": 2.9, "L_km": 52.82},
    {"name": "Yangjiang-3", "P_GWth": 2.9, "L_km": 52.41},
    {"name": "Yangjiang-4", "P_GWth": 2.9, "L_km": 52.49},
    {"name": "Yangjiang-5", "P_GWth": 2.9, "L_km": 52.11},
    {"name": "Yangjiang-6", "P_GWth": 2.9, "L_km": 52.19},
    {"name": "DayaBay-effective", "P_GWth": 17.4, "L_km": 215.0}]

reactor_data = pd.DataFrame(reactors)

km_to_cm = 1.0e5
reactor_data["L_cm"] = reactor_data["L_km"] * km_to_cm
reactor_data["w"] = reactor_data["P_GWth"] / (4.0 * np.pi * reactor_data["L_cm"] ** 2)
reactor_L = reactor_data["L_km"].to_numpy(dtype=float)
reactor_w = reactor_data["w"].to_numpy(dtype=float)


# ============================================================
# Huber coefficients and fission fractions
# ============================================================

alpha_huber = {
    "U235":  np.array([4.367, -4.577, 2.100, -5.294e-1, 6.186e-2, -2.777e-3]),
    "Pu239": np.array([4.757, -5.392, 2.563, -6.596e-1, 7.820e-2, -3.536e-3]),
    "Pu241": np.array([2.990, -2.882, 1.278, -3.343e-1, 3.905e-2, -1.754e-3])}

frac = {
    "U235": 0.564,
    "U238": 0.076,
    "Pu239": 0.304,
    "Pu241": 0.056}

# ============================================================
# Build continuous flux model from Daya Bay unfolded spectrum
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
    frac, alpha_huber, Delta, m_e, 500)

nbin = int(extras["nbin"])
print(f"Number of flux pulls: {nbin}")

sig_ibd = sigma_ibd(E_nu, Delta, m_e)
# ============================================================
# Nonlinearity model
# ============================================================

def load_or_make_nonlinearity_points() -> tuple[np.ndarray, np.ndarray]:
    try:
        path = "data/positron_nonlinearity.csv"

        df = pd.read_csv(path)

        if "E_pr" not in df.columns or "F_nl" not in df.columns:
            raise ValueError(f"{path} must contain columns 'E_pr' and 'F_nl'.")

        E_pts = df["E_pr"].to_numpy(dtype=float)
        F_pts = df["F_nl"].to_numpy(dtype=float)

        print(f"\nLoaded positron nonlinearity from {path}")

    except FileNotFoundError:
        print("\nWarning: positron_nonlinearity.csv not found.")
        print("Using placeholder nonlinearity points.")

        E_pts = np.array([
            0.5,
            0.7,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
            9.0,
            10.0,
        ])

        F_pts = np.array([
            0.90,
            0.92,
            0.94,
            0.965,
            0.98,
            0.995,
            1.005,
            1.012,
            1.018,
            1.022,
            1.025,
            1.027,
            1.028,
        ])

    order = np.argsort(E_pts)
    E_pts = E_pts[order]
    F_pts = F_pts[order]

    unique_E, unique_idx = np.unique(E_pts, return_index=True)

    return unique_E, F_pts[unique_idx]


E_pts, F_pts = load_or_make_nonlinearity_points()

if len(F_pts) >= 7:
    F_pts_sm = savgol_filter(
        F_pts,
        window_length=7,
        polyorder=2,
        mode="interp",
    )
else:
    F_pts_sm = F_pts.copy()

_Fnl = PchipInterpolator(E_pts, F_pts_sm, extrapolate=False)


def F_nl(Epr: np.ndarray) -> np.ndarray:
    Epr = np.asarray(Epr, dtype=float)
    Ecl = np.clip(Epr, E_pts[0], E_pts[-1])

    return _Fnl(Ecl)


def sigma_E(E: np.ndarray, a: float = res_a, b: float = res_b) -> np.ndarray:
    E = np.asarray(E, dtype=float)
    E = np.clip(E, 1e-6, None)

    return np.sqrt(a * a * E + b * b * E * E)


def ibd_positron_energy_endpoints(E_nu_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Exact two-body laboratory-frame positron-energy endpoints for

        anti-nu_e + p -> e+ + n

    with the proton initially at rest. Returned energies are total
    positron energies, including m_e.
    """
    E = np.asarray(E_nu_values, dtype=float)
    s = m_p * m_p + 2.0 * m_p * E
    sqrt_s = np.sqrt(np.clip(s, 1e-30, None))

    Ee_star = (s + m_e * m_e - m_n * m_n) / (2.0 * sqrt_s)
    pe_star2 = np.clip(Ee_star * Ee_star - m_e * m_e, 0.0, None)
    pe_star = np.sqrt(pe_star2)

    beta_cm = E / (E + m_p)
    gamma_cm = (E + m_p) / sqrt_s

    Ee_min = gamma_cm * (Ee_star - beta_cm * pe_star)
    Ee_max = gamma_cm * (Ee_star + beta_cm * pe_star)

    below_threshold = Ee_star < m_e
    Ee_min = np.where(below_threshold, m_e, Ee_min)
    Ee_max = np.where(below_threshold, m_e, Ee_max)

    return Ee_min, Ee_max


def nominal_prompt_energy_from_neutrino(E_nu_values: np.ndarray) -> np.ndarray:
    """Map true neutrino energy to prompt energy, including recoil."""
    E = np.asarray(E_nu_values, dtype=float)

    if RECOIL_MODE == "none":
        Ee = E - Delta
    elif RECOIL_MODE == "midpoint":
        Ee_min, Ee_max = ibd_positron_energy_endpoints(E)
        Ee = 0.5 * (Ee_min + Ee_max)
    else:
        raise ValueError("RECOIL_MODE must be 'midpoint' or 'none'.")

    return Ee + m_e


def compute_response_matrix_nl(
    xi_scl: float = 0.0,
    xi_bias: float = 0.0,
    xi_res: float = 0.0,
) -> np.ndarray:
    """
    Nonlinear binned response matrix on the exact JUNO bin edges.

    Standard-normal pulls are converted to physical fractional shifts
    with their corresponding one-sigma uncertainties.
    """
    Epr0 = nominal_prompt_energy_from_neutrino(E_nu)

    scale_factor = 1.0 + SIGMA_ENERGY_SCALE * xi_scl
    bias_term = SIGMA_ENERGY_BIAS * xi_bias

    F_nom = F_nl(Epr0)

    # Paper convention:
    # Epr -> Epr * [r_nl * (1 + xi_scl) * F_nl(Epr) + xi_bias]
    mu = Epr0 * (
        CONFIG_R_NL * scale_factor * F_nom
        + bias_term
    )

    sigE = sigma_E(mu, a=res_a, b=res_b)
    sigE = CONFIG_R_RES * sigE
    sigE = (1.0 + SIGMA_ENERGY_RES * xi_res) * sigE
    sigE = np.clip(sigE, 1e-12, None)

    lo = Epr_edges[:-1][:, None]
    hi = Epr_edges[1:][:, None]

    mu2 = mu[None, :]
    sig2 = sigE[None, :]

    z_hi = (hi - mu2) / (np.sqrt(2.0) * sig2)
    z_lo = (lo - mu2) / (np.sqrt(2.0) * sig2)

    Rmat = 0.5 * (erf(z_hi) - erf(z_lo))

    bad = E_nu < 1.806
    if np.any(bad):
        Rmat[:, bad] = 0.0

    return np.clip(Rmat, 0.0, 1.0)

# ============================================================
# Trapezoid integration weights
# ============================================================

trap_w = np.zeros_like(E_nu)
trap_w[1:-1] = 0.5 * (E_nu[2:] - E_nu[:-2])
trap_w[0] = 0.5 * (E_nu[1] - E_nu[0])
trap_w[-1] = 0.5 * (E_nu[-1] - E_nu[-2])


# ============================================================
# Flux with pull modes
# ============================================================

def evaluate_flux_with_pulls(xi_flux: np.ndarray | None = None) -> np.ndarray:
    if xi_flux is None:
        xi_flux = np.zeros(nbin, dtype=float)

    xi_flux = np.asarray(xi_flux, dtype=float)

    if xi_flux.size != nbin:
        raise ValueError(f"Expected {nbin} flux pulls, got {xi_flux.size}.")

    phi_E = np.asarray(phi_cont(E_nu, xi_flux), dtype=float).ravel()
    phi_E = np.clip(phi_E, 0.0, None)

    return phi_E


# ============================================================
# Reactor spectrum and bin-by-bin JUNO closure correction
# ============================================================

def reactor_survival_probability(
    E_MeV: np.ndarray,
    L_km: float,
    sin2_theta12_fit: float,
    sin2_theta13_fit: float,
    dm21_fit: float,
    dm31_fit: float,
) -> np.ndarray:
    """Three-flavour reactor survival probability with 1-2 matter effects."""
    E = np.asarray(E_MeV, dtype=float)

    s12_sq = float(np.clip(sin2_theta12_fit, 1e-8, 1.0 - 1e-8))
    s13_sq = float(np.clip(sin2_theta13_fit, 1e-8, 1.0 - 1e-8))

    c12_sq = 1.0 - s12_sq
    c13_sq = 1.0 - s13_sq

    sin2_2theta12 = 4.0 * s12_sq * c12_sq
    cos2theta12 = c12_sq - s12_sq
    sin2_2theta13 = 4.0 * s13_sq * c13_sq

    dm21_eff = dm21_fit * np.ones_like(E)
    sin2_2theta12_eff = sin2_2theta12 * np.ones_like(E)

    if USE_MATTER_EFFECTS:
        # For antineutrinos the matter potential has the opposite sign.
        # a = 2 sqrt(2) G_F N_e E
        #   = 1.52e-4 eV^2 * Y_e * rho[g/cm^3] * E[GeV].
        # The sign is negative for antineutrinos.
        matter_mass2 = (
            -1.52e-4
            * MATTER_DENSITY_G_CM3
            * ELECTRON_FRACTION
            * (E / 1000.0)
        )
        A = matter_mass2 / dm21_fit
        D = np.sqrt(
            (cos2theta12 - A) ** 2
            + sin2_2theta12
        )
        dm21_eff = dm21_fit * D
        sin2_2theta12_eff = sin2_2theta12 / np.clip(D * D, 1e-15, None)

    dm32_fit = dm31_fit - dm21_fit

    phase21 = 1.267 * dm21_eff * L_km * 1000.0 / E
    phase31 = 1.267 * dm31_fit * L_km * 1000.0 / E
    phase32 = 1.267 * dm32_fit * L_km * 1000.0 / E

    Pee = (
        1.0
        - c13_sq**2 * sin2_2theta12_eff * np.sin(phase21) ** 2
        - sin2_2theta13 * (
            c12_sq * np.sin(phase31) ** 2
            + s12_sq * np.sin(phase32) ** 2
        )
    )

    return np.clip(Pee, 0.0, 1.0)


def reactor_spectrum_raw(
    sin2_theta12_fit: float,
    dm21_fit: float,
    sin2_theta13_fit: float = sin2_theta13_nominal,
    xi_flux: np.ndarray | None = None,
    xi_scl: float = 0.0,
    xi_bias: float = 0.0,
    xi_res: float = 0.0,
    use_osc: bool = True,
) -> np.ndarray:
    phi_E = evaluate_flux_with_pulls(xi_flux)

    Rmat = compute_response_matrix_nl(
        xi_scl=xi_scl,
        xi_bias=xi_bias,
        xi_res=xi_res,
    )

    base_kernel = phi_E * sig_ibd * trap_w
    spectrum = np.zeros(len(Epr_centers), dtype=float)

    for L_km, w in zip(reactor_L, reactor_w):
        if use_osc:
            Pee = reactor_survival_probability(
                E_MeV=E_nu,
                L_km=L_km,
                sin2_theta12_fit=sin2_theta12_fit,
                sin2_theta13_fit=sin2_theta13_fit,
                dm21_fit=dm21_fit,
                dm31_fit=dm31,
            )
        else:
            Pee = np.ones_like(E_nu)

        spectrum += w * (Rmat @ (base_kernel * Pee))

    return spectrum


print("\nComputing nominal reactor spectra and bin-by-bin closure correction...")

raw_noosc_nominal = reactor_spectrum_raw(
    sin2_theta12_fit=sin2_theta12_nominal,
    dm21_fit=dm21_nominal,
    sin2_theta13_fit=sin2_theta13_nominal,
    use_osc=False,
)

valid_correction = (
    np.isfinite(raw_noosc_nominal)
    & np.isfinite(juno_noosc_interp)
    & (raw_noosc_nominal > 0.0)
    & (juno_noosc_interp >= 0.0)
)

if not np.any(valid_correction):
    raise RuntimeError("No valid bins are available for the bin-by-bin reactor correction.")

BINWISE_REACTOR_CORRECTION = np.ones_like(raw_noosc_nominal)
BINWISE_REACTOR_CORRECTION[valid_correction] = (
    juno_noosc_interp[valid_correction]
    / raw_noosc_nominal[valid_correction]
)

# Fill any invalid edge bins with the nearest valid correction value.
if np.any(~valid_correction):
    BINWISE_REACTOR_CORRECTION[~valid_correction] = np.interp(
        x_model[~valid_correction],
        x_model[valid_correction],
        BINWISE_REACTOR_CORRECTION[valid_correction],
    )

Ni_noosc_nominal = BINWISE_REACTOR_CORRECTION * raw_noosc_nominal

raw_osc_nominal = reactor_spectrum_raw(
    sin2_theta12_fit=sin2_theta12_nominal,
    dm21_fit=dm21_nominal,
    sin2_theta13_fit=sin2_theta13_nominal,
    use_osc=True,
)
Ni_reactor_nominal = BINWISE_REACTOR_CORRECTION * raw_osc_nominal

closure_error = np.max(
    np.abs(
        Ni_noosc_nominal[fit_mask]
        - juno_noosc_interp[fit_mask]
    )
)

print(f"  correction range = [{np.min(BINWISE_REACTOR_CORRECTION):.6g}, "
      f"{np.max(BINWISE_REACTOR_CORRECTION):.6g}]")
print(f"  max no-osc closure error = {closure_error:.6e} events/bin")

# ============================================================
# Background model: digitized shapes normalized to JUNO Table 1
# ============================================================

def normalize_shape_to_total(
    shape: np.ndarray,
    target_total: float,
    label: str,
) -> np.ndarray:
    shape = np.clip(np.asarray(shape, dtype=float), 0.0, None)
    current_total = float(np.sum(shape))

    if current_total <= 0.0 or not np.isfinite(current_total):
        raise ValueError(
            f"Cannot normalize background '{label}': its digitized shape "
            "has zero or invalid total."
        )

    return shape * (target_total / current_total)


def group_backgrounds(
    B_Geo: np.ndarray,
    B_Acc: np.ndarray,
    B_LiHe: np.ndarray,
    B_BiPo: np.ndarray,
    B_AtmNC: np.ndarray,
    B_FastN: np.ndarray,
    B_DoubleN: np.ndarray,
    B_C13an: np.ndarray,
) -> dict[str, np.ndarray]:
    B_Other = B_C13an + B_FastN + B_DoubleN + B_AtmNC + B_Acc

    return {
        "LiHe": B_LiHe,
        "Geo": B_Geo,
        "BiPo": B_BiPo,
        "Other": B_Other,
    }


def build_juno_reference_backgrounds() -> dict[str, np.ndarray]:
    """
    Diagnostic fallback only. Each non-world component borrows the
    official smooth total-background shape before Table-1 normalization.
    """
    B_total_ref = np.clip(
        juno_react_bk_interp - juno_react_interp,
        0.0,
        None,
    )

    return {
        "LiHe": B_total_ref.copy(),
        "Geo": B_total_ref.copy(),
        "BiPo": B_total_ref.copy(),
        "Other": B_total_ref.copy(),
    }


def build_digitized_backgrounds() -> dict[str, np.ndarray]:
    csv_path = "data/digitized_backgrounds.csv"

    print(f"\nUsing digitized background shapes: {csv_path}")

    df = pd.read_csv(csv_path).copy()
    df.columns = [str(c).strip() for c in df.columns]

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(axis=0, how="all")
    df = df.sort_values(df.columns[0]).reset_index(drop=True)

    x_source = df.iloc[:, 0].to_numpy(dtype=float)

    B_Geo_raw = interpolate_digitized_component(
        df, x_source, x_model,
        possible_names=["Geoneutrinos", "Geo", "GeoNeutrinos"],
        label="Geo",
        allow_missing=False,
    )

    B_Acc_raw = interpolate_digitized_component(
        df, x_source, x_model,
        possible_names=["Accidentals", "Accidental", "Acc"],
        label="Acc",
    )

    B_LiHe_raw = interpolate_digitized_component(
        df, x_source, x_model,
        possible_names=["Li9_He8", "9Li_8He", "LiHe", "9Li8He"],
        label="LiHe",
        allow_missing=False,
    )

    B_C13an_raw = interpolate_digitized_component(
        df, x_source, x_model,
        possible_names=["C13_alpha_n_O16", "13C_alpha_n_16O", "C13an", "13C"],
        label="C13an",
    )

    B_FastN_raw = interpolate_digitized_component(
        df, x_source, x_model,
        possible_names=["FastNeutrons", "Fast_neutrons", "FastN"],
        label="FastN",
    )

    B_AtmNC_raw = interpolate_digitized_component(
        df, x_source, x_model,
        possible_names=["AtmosphericNC", "Atmospheric_NC", "AtmNC"],
        label="AtmNC",
    )

    B_BiPo_raw = interpolate_digitized_component(
        df, x_source, x_model,
        possible_names=["BiPo", "Bi214Po214", "214Bi214Po"],
        label="BiPo",
        allow_missing=False,
    )

    B_DoubleN_raw = interpolate_digitized_component(
        df, x_source, x_model,
        possible_names=["DoubleNeutrons", "Double_neutrons", "DoubleN"],
        label="DoubleN",
    )

    return group_backgrounds(
        B_Geo=B_Geo_raw,
        B_Acc=B_Acc_raw,
        B_LiHe=B_LiHe_raw,
        B_BiPo=B_BiPo_raw,
        B_AtmNC=B_AtmNC_raw,
        B_FastN=B_FastN_raw,
        B_DoubleN=B_DoubleN_raw,
        B_C13an=B_C13an_raw,
    )


if BACKGROUND_SOURCE == "digitized":
    raw_B_components = build_digitized_backgrounds()
elif BACKGROUND_SOURCE == "juno_reference":
    raw_B_components = build_juno_reference_backgrounds()
else:
    raise ValueError("BACKGROUND_SOURCE must be 'digitized' or 'juno_reference'.")

# The paper uses the unoscillated reactor spectrum as the world-reactor shape.
raw_B_components["World"] = Ni_noosc_nominal.copy()

B_components = {}
for name in ["LiHe", "Geo", "World", "BiPo", "Other"]:
    target_events = TABLE1_RATE_CPD[name] * LIVE_DAYS
    B_components[name] = normalize_shape_to_total(
        raw_B_components[name],
        target_events,
        name,
    )

BASE_B_COMPONENTS = {
    name: arr.copy()
    for name, arr in B_components.items()
}

B_total = sum(B_components.values())

print("\nTable-1-normalized background totals:")
for name, arr in B_components.items():
    target = TABLE1_RATE_CPD[name] * LIVE_DAYS
    print(
        f"  {name:8s}: {np.sum(arr):10.5f} events "
        f"(target {target:10.5f})"
    )
print(f"  {'Total':8s}: {np.sum(B_total):10.5f} events")

# ============================================================
# Observation and smooth plotting references
# ============================================================

# Smooth official curves are plotting references only.
N_plot_reactor = juno_react_interp.copy()
N_plot_total = juno_react_bk_interp.copy()
N_obs_reactor = N_plot_reactor.copy()

# Use exactly one statistically independent observation vector.
if FIT_TO_MEASURED_DATA:
    N_obs_total = juno_data_interp.copy()
    FIT_TARGET_NAME = "JUNO measured data"
else:
    N_obs_total = juno_react_bk_interp.copy()
    FIT_TARGET_NAME = "JUNO smooth reactor + background curve"

N_obs = N_obs_total.copy()

print(f"Likelihood target: {FIT_TARGET_NAME}")
print(f"Number of fitted energy bins: {np.count_nonzero(fit_mask)}")

# ============================================================
# Pull bookkeeping
# ============================================================

pull_names = (
    [
        "Reactor norm",
        "LiHe norm",
        "Geo norm",
        "World norm",
        "BiPo norm",
        "Other norm",
        "LiHe shape",
    ]
    + [f"Flux mode {k + 1}" for k in range(nbin)]
    + [
        "Energy scale",
        "Energy bias",
        "Energy resolution",
        "sin2 theta13",
    ]
)

pull_index = {name: i for i, name in enumerate(pull_names)}
n_pulls = len(pull_names)

print("\nFit model:")
print("  parameters of interest: sin^2(theta12), dm21")
print(f"  nuisance pulls: {n_pulls}")


# ============================================================
# Configuration machinery
# ============================================================

lihe_shape_fraction = SIGMA_LIHE_SHAPE_AT_1MEV * x_model


def make_config_backgrounds(r_bg: float) -> dict[str, np.ndarray]:
    """
    Apply r_BG to all backgrounds except geoneutrinos.
    """

    B_cfg = {}

    for name, arr in BASE_B_COMPONENTS.items():
        if name == "Geo":
            B_cfg[name] = arr.copy()
        else:
            B_cfg[name] = r_bg * arr

    return B_cfg


def update_lihe_shape_fraction() -> None:
    global lihe_shape_fraction

    lihe_shape_fraction = SIGMA_LIHE_SHAPE_AT_1MEV * x_model

    if ORTHOGONALIZE_LIHE_SHAPE_TO_NORM and np.sum(B_components["LiHe"]) > 0:
        lihe_mean = (
            np.sum(B_components["LiHe"] * lihe_shape_fraction)
            / np.sum(B_components["LiHe"])
        )
        lihe_shape_fraction = lihe_shape_fraction - lihe_mean


def set_analysis_configuration(cfg: dict) -> None:
    """
    Select configuration-specific analysis settings.
    """

    global CHI2_KIND
    global CONFIG_R_NL
    global CONFIG_R_RES
    global SIGMA_ENERGY_BIAS
    global SIGMA_ENERGY_RES
    global B_components
    global B_total

    CHI2_KIND = cfg["chi2_kind"]
    CONFIG_R_NL = cfg["r_nl"]
    CONFIG_R_RES = cfg["r_res"]
    SIGMA_ENERGY_BIAS = cfg["sigma_bias"]
    SIGMA_ENERGY_RES = cfg["sigma_res"]

    B_components = make_config_backgrounds(cfg["r_bg"])
    B_total = sum(B_components.values())

    update_lihe_shape_fraction()

    print("\n" + "=" * 72)
    print(f"Configuration: {cfg['label']}")
    print("=" * 72)
    print(f"  chi2 kind    = {CHI2_KIND}")
    print(f"  r_BG         = {cfg['r_bg']:.4f}")
    print(f"  r_nl         = {cfg['r_nl']:.4f}")
    print(f"  sigma_bias   = {100.0 * SIGMA_ENERGY_BIAS:.3f}%")
    print(f"  r_res        = {cfg['r_res']:.4f}")
    print(f"  sigma_res    = {100.0 * SIGMA_ENERGY_RES:.3f}%")

    print("\nBackground sums after configuration scaling:")
    for name, arr in B_components.items():
        print(f"  {name:8s}: {np.sum(arr):.6g}")
    print(f"  {'Total':8s}: {np.sum(B_total):.6g}")


# ============================================================
# Fit-vector helpers
# ============================================================

def unpack_fit_vector(x: np.ndarray) -> dict[str, object]:
    x = np.asarray(x, dtype=float)

    sin2_theta12_fit = x[0]
    dm21_fit = x[1]

    theta = x[2:]

    if theta.size != n_pulls:
        raise ValueError(f"Expected {n_pulls} pulls, got {theta.size}.")

    i = 0

    xi_reactor = theta[i]
    i += 1

    xi_LiHe = theta[i]
    i += 1

    xi_Geo = theta[i]
    i += 1

    xi_World = theta[i]
    i += 1

    xi_BiPo = theta[i]
    i += 1

    xi_Other = theta[i]
    i += 1

    xi_LiHe_shape = theta[i]
    i += 1

    xi_flux = theta[i:i + nbin]
    i += nbin

    xi_scl = theta[i]
    i += 1

    xi_bias = theta[i]
    i += 1

    xi_res = theta[i]
    i += 1

    xi_theta13 = theta[i]
    i += 1

    if i != theta.size:
        raise RuntimeError("Internal pull bookkeeping mismatch.")

    return {
        "sin2_theta12": sin2_theta12_fit,
        "dm21": dm21_fit,
        "theta": theta,
        "xi_reactor": xi_reactor,
        "xi_LiHe": xi_LiHe,
        "xi_Geo": xi_Geo,
        "xi_World": xi_World,
        "xi_BiPo": xi_BiPo,
        "xi_Other": xi_Other,
        "xi_LiHe_shape": xi_LiHe_shape,
        "xi_flux": xi_flux,
        "xi_scl": xi_scl,
        "xi_bias": xi_bias,
        "xi_res": xi_res,
        "xi_theta13": xi_theta13,
    }


def make_x_from_osc_and_pulls(
    sin2_theta12: float,
    dm21: float,
    theta: np.ndarray | None = None,
) -> np.ndarray:
    if theta is None:
        theta = np.zeros(n_pulls, dtype=float)

    x = np.zeros(2 + n_pulls, dtype=float)
    x[0] = sin2_theta12
    x[1] = dm21
    x[2:] = theta

    return x


# ============================================================
# Prediction and chi-square
# ============================================================

def predict_components_from_fit_vector(
    x: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Recalculate the full nonlinear prediction at the current
    oscillation parameters and nuisance pulls.
    """

    p = unpack_fit_vector(x)

    S_raw = reactor_spectrum_raw(
        sin2_theta12_fit=p["sin2_theta12"],
        dm21_fit=p["dm21"],
        sin2_theta13_fit=(
            sin2_theta13_nominal
            + SIGMA_SIN2_THETA13 * p["xi_theta13"]
        ),
        xi_flux=p["xi_flux"],
        xi_scl=p["xi_scl"],
        xi_bias=p["xi_bias"],
        xi_res=p["xi_res"],
        use_osc=True,
    )

    S_reactor = BINWISE_REACTOR_CORRECTION * S_raw

    S_sys = (
        positive_scale(
            SIGMA_REACTOR_NORM,
            p["xi_reactor"],
        )
        * S_reactor
    )

    B_LiHe_sys = (
        positive_scale(
            SIGMA_BG_NORM["LiHe"],
            p["xi_LiHe"],
        )
        * B_components["LiHe"]
    )

    B_Geo_sys = (
        positive_scale(
            SIGMA_BG_NORM["Geo"],
            p["xi_Geo"],
        )
        * B_components["Geo"]
    )

    B_World_sys = (
        positive_scale(
            SIGMA_BG_NORM["World"],
            p["xi_World"],
        )
        * B_components["World"]
    )

    B_BiPo_sys = (
        positive_scale(
            SIGMA_BG_NORM["BiPo"],
            p["xi_BiPo"],
        )
        * B_components["BiPo"]
    )

    B_Other_sys = (
        positive_scale(
            SIGMA_BG_NORM["Other"],
            p["xi_Other"],
        )
        * B_components["Other"]
    )

    B_LiHe_sys = B_LiHe_sys * (
        1.0
        + p["xi_LiHe_shape"]
        * lihe_shape_fraction
    )
    B_LiHe_sys = np.clip(B_LiHe_sys, 0.0, None)

    B_total_sys = (
        B_LiHe_sys
        + B_Geo_sys
        + B_World_sys
        + B_BiPo_sys
        + B_Other_sys
    )

    N_pred = S_sys + B_total_sys

    return {
        "reactor": np.clip(S_sys, 1e-12, None),
        "background": np.clip(B_total_sys, 0.0, None),
        "total": np.clip(N_pred, 1e-12, None),
    }


def predict_from_fit_vector(x: np.ndarray) -> np.ndarray:
    return predict_components_from_fit_vector(x)["total"]


def chi2_data_term(N_obs: np.ndarray, N_pred: np.ndarray) -> float:
    O = np.asarray(N_obs, dtype=float)
    P = np.asarray(N_pred, dtype=float)

    P = np.clip(P, 1e-12, None)

    if CHI2_KIND == "gaussian":
        var = np.maximum(O, 1.0)
        return float(np.sum((O - P) ** 2 / var))

    if CHI2_KIND == "poisson":
        O_safe = np.clip(O, 1e-12, None)
        term = P - O + O_safe * np.log(O_safe / P)
        return float(2.0 * np.sum(term))

    if CHI2_KIND == "cnp":
        O_safe = np.clip(O, 1e-12, None)
        sigma2 = 3.0 / (1.0 / O_safe + 2.0 / P)
        sigma2 = np.clip(sigma2, 1e-12, None)
        return float(np.sum((P - O) ** 2 / sigma2))

    raise ValueError("CHI2_KIND must be 'gaussian', 'poisson', or 'cnp'.")


def chi2_full(x: np.ndarray) -> float:
    """
    Profile-likelihood objective using one statistically independent
    JUNO spectrum.

    The full nonlinear prediction is rebuilt on every call.  The smooth
    reactor-only reference is deliberately not added as a second data
    term, because it is already contained in the total spectrum.
    """

    p = unpack_fit_vector(x)
    components = predict_components_from_fit_vector(x)

    chi2_stat = (
        TOTAL_SPECTRUM_WEIGHT
        * chi2_data_term(
            N_obs_total[fit_mask],
            components["total"][fit_mask],
        )
    )

    chi2_pull = (
        PULL_PENALTY_WEIGHT
        * np.sum(p["theta"] ** 2)
    )

    return float(chi2_stat + chi2_pull)


# ============================================================
# Global fit and solar scan
# ============================================================

bounds_full = [
    SIN2_THETA12_RANGE,
    DM21_RANGE,
]
bounds_full += [PULL_BOUNDS] * n_pulls


def run_global_fit_current_config(cfg: dict) -> dict:
    x0 = make_x_from_osc_and_pulls(
        sin2_theta12=sin2_theta12_nominal,
        dm21=dm21_nominal,
    )

    x_nominal = x0.copy()
    N_nominal = predict_from_fit_vector(x_nominal)
    chi2_nominal = chi2_full(x_nominal)

    print("\nNominal point:")
    print(f"  sin^2(theta12) = {x_nominal[0]:.8f}")
    print(f"  dm21           = {x_nominal[1]:.8e} eV^2")
    print(f"  chi2           = {chi2_nominal:.6f}")

    if RUN_GLOBAL_BEST_FIT:
        print("\nRunning global best fit...")

        fit_t0 = perf_counter()

        result_global = minimize(
            chi2_full,
            x0=x0,
            method="L-BFGS-B",
            bounds=bounds_full,
            options={
                "maxiter": FIT_MAXITER_GLOBAL,
                "ftol": 1e-7,
                "gtol": 1e-5,
                "maxls": 30,
            },
        )

        fit_elapsed = perf_counter() - fit_t0

        x_best = result_global.x
        components_best = predict_components_from_fit_vector(x_best)
        N_best = components_best["total"]
        N_reactor_best = components_best["reactor"]
        chi2_best = chi2_full(x_best)
        p_best = unpack_fit_vector(x_best)

        print("\nGlobal best fit:")
        print(f"  success = {result_global.success}")
        print(f"  message = {result_global.message}")
        print(f"  time    = {fit_elapsed:.2f} s")
        print(f"  chi2    = {chi2_best:.6f}")
        print(f"  Delta chi2 nominal-best = {chi2_nominal - chi2_best:.6f}")
        print(f"  sin^2(theta12) = {p_best['sin2_theta12']:.8f}")
        print(f"  dm21           = {p_best['dm21']:.8e} eV^2")

    else:
        result_global = None
        x_best = x_nominal.copy()
        components_best = predict_components_from_fit_vector(x_best)
        N_best = components_best["total"]
        N_reactor_best = components_best["reactor"]
        chi2_best = chi2_nominal
        p_best = unpack_fit_vector(x_best)

    print("\nBest-fit main pulls:")
    theta_best = p_best["theta"]

    for name in [
        "Reactor norm",
        "LiHe norm",
        "Geo norm",
        "World norm",
        "BiPo norm",
        "Other norm",
        "LiHe shape",
        "Energy scale",
        "Energy bias",
        "Energy resolution",
        "sin2 theta13",
    ]:
        val = theta_best[pull_index[name]]
        print(f"  {name:20s}: {val:+.5f}")

    return {
        "cfg": cfg,
        "x_nominal": x_nominal,
        "N_nominal": N_nominal,
        "chi2_nominal": chi2_nominal,
        "result_global": result_global,
        "x_best_global": x_best,
        "N_best_global": N_best,
        "N_reactor_best_global": N_reactor_best,
        "chi2_best_global": chi2_best,
        "p_best_global": p_best,
    }


def chi2_profile_pulls_fixed_osc(
    theta: np.ndarray,
    sin2_theta12_fixed: float,
    dm21_fixed: float,
) -> float:
    x = make_x_from_osc_and_pulls(
        sin2_theta12=sin2_theta12_fixed,
        dm21=dm21_fixed,
        theta=theta,
    )

    return chi2_full(x)


def minimize_profiled_pulls(
    sin2_theta12_fixed: float,
    dm21_fixed: float,
    theta_warm: np.ndarray,
    theta_global: np.ndarray,
):
    """
    Minimize nuisance pulls at one oscillation-grid point.

    A second start is useful because an unsuccessful or local nuisance
    minimum artificially raises chi-square away from the best fit and
    therefore makes the confidence contours too small.
    """

    starts = [np.asarray(theta_warm, dtype=float)]

    if PROFILE_MULTI_START:
        starts.append(np.asarray(theta_global, dtype=float))
        starts.append(np.zeros(n_pulls, dtype=float))

    # Remove duplicate starts while preserving order.
    unique_starts = []
    for candidate in starts:
        if not any(np.allclose(candidate, old) for old in unique_starts):
            unique_starts.append(candidate)

    starts = unique_starts[:PROFILE_MAX_STARTS]

    best_result = None

    for start_theta in starts:
        result = minimize(
            chi2_profile_pulls_fixed_osc,
            x0=start_theta,
            args=(sin2_theta12_fixed, dm21_fixed),
            method="L-BFGS-B",
            bounds=[PULL_BOUNDS] * n_pulls,
            options={
                "maxiter": FIT_MAXITER_SCAN,
                "ftol": 1e-9,
                "gtol": 1e-7,
                "maxls": 40,
            },
        )

        if np.isfinite(result.fun):
            if best_result is None or result.fun < best_result.fun:
                best_result = result

    if best_result is None:
        raise RuntimeError(
            "Nuisance profiling returned no finite result at "
            f"sin²(theta12)={sin2_theta12_fixed:.8f}, "
            f"dm21={dm21_fixed:.8e}."
        )

    if not best_result.success:
        print(
            "Warning: nuisance profiling did not fully converge at "
            f"sin²(theta12)={sin2_theta12_fixed:.8f}, "
            f"dm21={dm21_fixed:.8e}: "
            f"{best_result.message}"
        )

    return best_result


def run_solar_scan_current_config(cfg: dict, global_fit_result: dict) -> dict:
    print("\nRunning solar contour scan...")
    print(f"  sin^2(theta12) points: {N_SIN2_POINTS}")
    print(f"  dm21 points:           {N_DM21_POINTS}")
    print(f"  total grid points:     {N_SIN2_POINTS * N_DM21_POINTS}")
    print(f"  pulls per point:       {n_pulls}")

    scan_t0 = perf_counter()

    sin2_theta12_grid = np.linspace(
        SIN2_THETA12_RANGE[0],
        SIN2_THETA12_RANGE[1],
        N_SIN2_POINTS,
    )

    dm21_grid = np.linspace(
        DM21_RANGE[0],
        DM21_RANGE[1],
        N_DM21_POINTS,
    )

    chi2_grid = np.zeros(
        (len(dm21_grid), len(sin2_theta12_grid)),
        dtype=float,
    )

    pull_grid = np.zeros(
        (len(dm21_grid), len(sin2_theta12_grid), n_pulls),
        dtype=float,
    )

    theta_warm_global = global_fit_result["p_best_global"]["theta"].copy()

    n_total = len(dm21_grid) * len(sin2_theta12_grid)
    n_done = 0
    best_seen_chi2 = np.inf

    for iy, dm21_test in enumerate(dm21_grid):
        row_t0 = perf_counter()

        theta_warm = theta_warm_global.copy()

        for ix, sin2_test in enumerate(sin2_theta12_grid):
            result_scan = minimize_profiled_pulls(
                sin2_theta12_fixed=sin2_test,
                dm21_fixed=dm21_test,
                theta_warm=theta_warm,
                theta_global=theta_warm_global,
            )

            theta_best_grid = result_scan.x
            chi2_best_grid = float(result_scan.fun)

            chi2_grid[iy, ix] = chi2_best_grid
            pull_grid[iy, ix, :] = theta_best_grid

            theta_warm = theta_best_grid.copy()

            if chi2_best_grid < best_seen_chi2:
                best_seen_chi2 = chi2_best_grid
                theta_warm_global = theta_best_grid.copy()

            n_done += 1

        row_elapsed = perf_counter() - row_t0
        elapsed = perf_counter() - scan_t0
        avg_time = elapsed / max(n_done, 1)
        eta = avg_time * (n_total - n_done)

        print(
            f"  row {iy + 1:3d}/{len(dm21_grid)} "
            f"| dm21 = {dm21_test:.5e} "
            f"| row time = {row_elapsed:.2f} s "
            f"| elapsed = {elapsed / 60:.2f} min "
            f"| ETA = {eta / 60:.2f} min"
        )

    scan_elapsed = perf_counter() - scan_t0

    chi2_min_scan = float(np.min(chi2_grid))
    dchi2_grid = chi2_grid - chi2_min_scan

    iy_best_scan, ix_best_scan = np.unravel_index(
        np.argmin(chi2_grid),
        chi2_grid.shape,
    )

    best_sin2_scan = sin2_theta12_grid[ix_best_scan]
    best_dm21_scan = dm21_grid[iy_best_scan]
    best_pulls_scan = pull_grid[iy_best_scan, ix_best_scan, :]

    x_best_scan = make_x_from_osc_and_pulls(
        sin2_theta12=best_sin2_scan,
        dm21=best_dm21_scan,
        theta=best_pulls_scan,
    )

    best_scan_components = predict_components_from_fit_vector(
        x_best_scan
    )
    N_best_scan = best_scan_components["total"]
    N_reactor_best_scan = best_scan_components["reactor"]

    print("\nBest fit from 2D solar scan:")
    print(f"  sin^2(theta12) = {best_sin2_scan:.8f}")
    print(f"  dm21           = {best_dm21_scan:.8e} eV^2")
    print(f"  chi2_min       = {chi2_min_scan:.6f}")

    print("\nBest-fit scan pulls:")
    for name, val in zip(pull_names, best_pulls_scan):
        print(f"  {name:20s}: {val:+.5f}")

    print("\nSolar scan timing:")
    print(f"  total time = {scan_elapsed:.2f} s")
    print(f"  total time = {scan_elapsed / 60:.2f} min")

    return {
        "cfg": cfg,
        "sin2_theta12_grid": sin2_theta12_grid,
        "dm21_grid": dm21_grid,
        "chi2_grid": chi2_grid,
        "dchi2_grid": dchi2_grid,
        "pull_grid": pull_grid,
        "best_sin2": best_sin2_scan,
        "best_dm21": best_dm21_scan,
        "best_pulls": best_pulls_scan,
        "x_best_scan": x_best_scan,
        "N_best_scan": N_best_scan,
        "N_reactor_best_scan": N_reactor_best_scan,
        "chi2_min": chi2_min_scan,
        "scan_elapsed": scan_elapsed,
    }


# ============================================================
# Run cnf 1 only
# ============================================================

configuration_results = []

for cfg in CONFIGURATIONS:
    set_analysis_configuration(cfg)

    global_fit_result = run_global_fit_current_config(cfg)

    if RUN_SOLAR_SCAN:
        scan_result = run_solar_scan_current_config(cfg, global_fit_result)
    else:
        scan_result = None

    configuration_results.append(
        {
            "cfg": cfg,
            "global": global_fit_result,
            "scan": scan_result,
        }
    )


# ============================================================
# Save numerical results
# ============================================================

if SAVE_RESULTS and RUN_SOLAR_SCAN:
    save_dict = {
        "x_model": x_model,
        "fit_mask": fit_mask,
        "N_obs": N_obs,
        "N_obs_reactor": N_obs_reactor,
        "N_obs_total": N_obs_total,
        "N_plot_reactor": N_plot_reactor,
        "N_plot_total": N_plot_total,
        "binwise_reactor_correction": BINWISE_REACTOR_CORRECTION,
        "pull_penalty_weight": np.array(
            PULL_PENALTY_WEIGHT,
            dtype=float,
        ),
        "reactor_spectrum_weight": np.array(
            REACTOR_SPECTRUM_WEIGHT,
            dtype=float,
        ),
        "total_spectrum_weight": np.array(
            TOTAL_SPECTRUM_WEIGHT,
            dtype=float,
        ),
        "pull_names": np.array(pull_names, dtype=object),
    }

    for result in configuration_results:
        key = result["cfg"]["key"]
        scan = result["scan"]
        glob = result["global"]

        save_dict[f"{key}_sin2_theta12_grid"] = scan["sin2_theta12_grid"]
        save_dict[f"{key}_dm21_grid"] = scan["dm21_grid"]
        save_dict[f"{key}_chi2_grid"] = scan["chi2_grid"]
        save_dict[f"{key}_dchi2_grid"] = scan["dchi2_grid"]
        save_dict[f"{key}_pull_grid"] = scan["pull_grid"]
        save_dict[f"{key}_best_sin2"] = scan["best_sin2"]
        save_dict[f"{key}_best_dm21"] = scan["best_dm21"]
        save_dict[f"{key}_N_best_scan"] = scan["N_best_scan"]
        save_dict[f"{key}_N_reactor_best_scan"] = (
            scan["N_reactor_best_scan"]
        )
        save_dict[f"{key}_N_nominal"] = glob["N_nominal"]
        save_dict[f"{key}_N_best_global"] = glob["N_best_global"]
        save_dict[f"{key}_N_reactor_best_global"] = (
            glob["N_reactor_best_global"]
        )

    np.savez(CONFIG_COMPARISON_RESULTS_PATH, **save_dict)

    print(f"\nSaved cnf 1 fit arrays to:")
    print(f"  {CONFIG_COMPARISON_RESULTS_PATH}")


# ============================================================
# Figure 2-style plot: cnf 1 contours + spectrum
# ============================================================

if RUN_SOLAR_SCAN:
    from matplotlib.ticker import AutoMinorLocator

    FIG2_PATH = "figure2_cnf1_two_juno_spectrum_lines.png"
    FIG2_PDF_PATH = "figure2_cnf1_two_juno_spectrum_lines.pdf"

    # Optional exact external JUNO likelihood grid.
    # Expected keys:
    #   sin2_theta12_grid
    #   dm21_grid
    #   dchi2_grid
    #
    # If unavailable, a correlated-Gaussian JUNO reference is
    # generated from the published best fit and 1D uncertainties.
    JUNO_REFERENCE_NPZ = Path("data/juno_reference_solar_scan.npz")

    paper_colors = {
        "cnf1": "#ff1493",   # magenta-like
    }

    plt.rcParams.update({
        "font.size": 10,
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.linewidth": 0.9,
    })

    def get_config_result(key: str) -> dict:
        for result in configuration_results:
            if result["cfg"]["key"] == key:
                return result
        raise KeyError(f"Could not find configuration '{key}'.")

    def style_axis(ax) -> None:
        ax.tick_params(
            direction="in",
            top=True,
            right=True,
            which="both",
            length=4,
        )
        ax.tick_params(which="minor", length=2)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())

    def load_juno_reference() -> dict:
        """
        Load an exact external JUNO Delta-chi-square grid when
        available. Otherwise return a correlated-Gaussian visual
        reference.
        """

        if JUNO_REFERENCE_NPZ.exists():
            data = np.load(JUNO_REFERENCE_NPZ)

            juno_ref = {
                "sin2_theta12_grid": data["sin2_theta12_grid"],
                "dm21_grid": data["dm21_grid"],
                "dchi2_grid": data["dchi2_grid"],
            }

            print("\nLoaded JUNO reference contours from:")
            print(f"  {JUNO_REFERENCE_NPZ}")

            return juno_ref

        print("\nWarning: exact JUNO likelihood grid not found.")
        print("Using a correlated-Gaussian JUNO reference.")

        sin2_grid = np.linspace(
            SIN2_THETA12_RANGE[0],
            SIN2_THETA12_RANGE[1],
            301,
        )

        dm21_grid = np.linspace(
            DM21_RANGE[0],
            DM21_RANGE[1],
            301,
        )

        theta_mesh, dm21_mesh = np.meshgrid(
            sin2_grid,
            dm21_grid,
        )

        theta_std = (
            theta_mesh - JUNO_BEST_SIN2_THETA12
        ) / JUNO_SIGMA_SIN2_THETA12

        dm21_std = (
            dm21_mesh - JUNO_BEST_DM21
        ) / JUNO_SIGMA_DM21

        dchi2_grid = (
            theta_std**2
            - 2.0 * JUNO_CORRELATION * theta_std * dm21_std
            + dm21_std**2
        ) / (1.0 - JUNO_CORRELATION**2)

        return {
            "sin2_theta12_grid": sin2_grid,
            "dm21_grid": dm21_grid,
            "dchi2_grid": dchi2_grid,
        }

    def plot_contours_and_profiles(
        ax_main,
        ax_top,
        ax_right,
        sin2_grid: np.ndarray,
        dm21_grid: np.ndarray,
        dchi2_grid: np.ndarray,
        color: str,
        label: str,
        linestyle: str = "-",
        linewidth: float = 1.45,
        zorder: int = 3,
        plot_profiles: bool = True,
    ) -> None:
        """
        Plot the 2D confidence contours.

        When plot_profiles is True, also draw the two marginalized
        one-dimensional Delta-chi-square curves on the top and right axes.
        """

        Z = np.asarray(dchi2_grid, dtype=float)
        Z = Z - np.nanmin(Z)

        X, Y = np.meshgrid(
            sin2_grid,
            dm21_grid * 1.0e5,
        )

        # 1 sigma, 2 sigma, 3 sigma for two fitted parameters.
        levels_2d = [2.30, 6.18, 11.83]

        good_levels = [
            lev for lev in levels_2d
            if np.nanmin(Z) < lev < np.nanmax(Z)
        ]

        if good_levels:
            ax_main.contour(
                X,
                Y,
                Z,
                levels=good_levels,
                colors=color,
                linestyles=linestyle,
                linewidths=linewidth,
                zorder=zorder,
            )

        if plot_profiles:
            profile_sin2 = np.nanmin(Z, axis=0)
            profile_dm21 = np.nanmin(Z, axis=1)

            profile_sin2 = profile_sin2 - np.nanmin(profile_sin2)
            profile_dm21 = profile_dm21 - np.nanmin(profile_dm21)

            ax_top.plot(
                sin2_grid,
                profile_sin2,
                color=color,
                linestyle=linestyle,
                lw=linewidth,
                label=label,
                zorder=zorder,
            )

            ax_right.plot(
                profile_dm21,
                dm21_grid * 1.0e5,
                color=color,
                linestyle=linestyle,
                lw=linewidth,
                zorder=zorder,
            )


    # --------------------------------------------------------
    # Figure layout
    # --------------------------------------------------------

    fig = plt.figure(figsize=(7.6, 5.5))

    outer = fig.add_gridspec(
        1,
        2,
        left=0.08,
        right=0.98,
        bottom=0.12,
        top=0.93,
        width_ratios=[1.48, 1.0],
        wspace=0.28,
    )

    left = outer[0, 0].subgridspec(
        2,
        2,
        height_ratios=[0.82, 1.68],
        width_ratios=[1.85, 0.92],
        hspace=0.05,
        wspace=0.05,
    )

    right = outer[0, 1].subgridspec(
        1,
        1,
    )

    ax_top = fig.add_subplot(left[0, 0])
    ax_main = fig.add_subplot(left[1, 0], sharex=ax_top)
    ax_prof = fig.add_subplot(left[1, 1], sharey=ax_main)

    ax_spec1 = fig.add_subplot(right[0, 0])

    # --------------------------------------------------------
    # Left panel: solar-parameter contours
    # --------------------------------------------------------

    for result in configuration_results:
        cfg = result["cfg"]
        scan = result["scan"]

        key = cfg["key"]
        label = cfg["label"]
        color = paper_colors.get(key, CONFIG_COLORS[key])

        plot_contours_and_profiles(
            ax_main=ax_main,
            ax_top=ax_top,
            ax_right=ax_prof,
            sin2_grid=scan["sin2_theta12_grid"],
            dm21_grid=scan["dm21_grid"],
            dchi2_grid=scan["dchi2_grid"],
            color=color,
            label=label,
            linestyle="-",
            linewidth=1.55,
            zorder=3,
        )

    juno_ref = load_juno_reference()

    if juno_ref is not None:
        plot_contours_and_profiles(
            ax_main=ax_main,
            ax_top=ax_top,
            ax_right=ax_prof,
            sin2_grid=juno_ref["sin2_theta12_grid"],
            dm21_grid=juno_ref["dm21_grid"],
            dchi2_grid=juno_ref["dchi2_grid"],
            color="black",
            label="JUNO",
            linestyle="--",
            linewidth=1.55,
            zorder=5,
            plot_profiles=False,
        )

    ax_main.set_xlabel(r"$\sin^2\theta_{12}$")
    ax_main.set_ylabel(r"$\Delta m^2_{21}\,[10^{-5}\,\mathrm{eV}^2]$")

    ax_top.set_ylabel(r"$\Delta \chi^2$")
    ax_prof.set_xlabel(r"$\Delta \chi^2$")

    ax_top.set_yticks([0, 2, 4, 6, 8])
    ax_prof.set_xticks([0, 2, 4, 6, 8])

    plt.setp(ax_top.get_xticklabels(), visible=False)
    plt.setp(ax_prof.get_yticklabels(), visible=False)

    contour_legend_handles = [
        Line2D(
            [0],
            [0],
            color=paper_colors["cnf1"],
            linestyle="-",
            lw=1.55,
            label="cnf 1",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="--",
            lw=1.55,
            label="JUNO",
        ),
    ]

    ax_top.legend(
        handles=contour_legend_handles,
        loc="upper right",
        frameon=True,
        fontsize=9,
        handlelength=2.2,
    )

    for ax in [ax_top, ax_main, ax_prof]:
        style_axis(ax)

    # --------------------------------------------------------
    # Right panel: best-fit spectra
    # --------------------------------------------------------

    spectra_info = [
        ("cnf1", ax_spec1),
    ]

    for key, ax in spectra_info:
        result = get_config_result(key)

        cfg = result["cfg"]
        glob = result["global"]

        label = cfg["label"]
        color = paper_colors.get(key, CONFIG_COLORS[key])

        set_analysis_configuration(cfg)

        # Use the continuous global best fit for the spectrum panel.
        x_best_with_pulls = glob["x_best_global"]
        p_best = unpack_fit_vector(x_best_with_pulls)
        N_best_with_pulls_and_backgrounds = glob["N_best_global"]

        # Figure-2 lower histogram: best-fit oscillation parameters with
        # every nuisance pull set to zero.
        x_best_without_pulls = make_x_from_osc_and_pulls(
            sin2_theta12=p_best["sin2_theta12"],
            dm21=p_best["dm21"],
            theta=np.zeros(n_pulls, dtype=float),
        )
        N_reactor_without_pulls = predict_components_from_fit_vector(
            x_best_without_pulls
        )["reactor"]

        ax.step(
            x_model[fit_mask],
            N_reactor_without_pulls[fit_mask],
            where="mid",
            color=color,
            lw=1.2,
            label=label,
        )

        # Figure-2 upper histogram: reactor plus backgrounds with the
        # profiled pull shifts included.
        ax.step(
            x_model[fit_mask],
            N_best_with_pulls_and_backgrounds[fit_mask],
            where="mid",
            color=color,
            lw=1.2,
        )

        # Smooth official JUNO references. The noisy measured data remain
        # in the likelihood but are deliberately not drawn as a histogram.
        ax.step(
            x_model[fit_mask],
            N_plot_reactor[fit_mask],
            where="mid",
            color="black",
            lw=1.0,
            label="JUNO",
        )
        ax.step(
            x_model[fit_mask],
            N_plot_total[fit_mask],
            where="mid",
            color="black",
            lw=1.0,
        )

        ax.set_xlim(FIT_ENERGY_MIN, FIT_ENERGY_MAX)
        ax.set_ylabel("events per 0.1 MeV", fontsize=9)

        ax.legend(
            loc="upper right",
            frameon=True,
            fontsize=8,
            handlelength=2.0,
        )

        style_axis(ax)

    ax_spec1.set_xlabel(r"$E_{\rm pr}$ [MeV]")

    # --------------------------------------------------------
    # Caption inside the saved figure
    # --------------------------------------------------------

    fig.savefig(FIG2_PATH, dpi=300)
    fig.savefig(FIG2_PDF_PATH)

    print("\nSaved cnf 1 Figure 2-style plot to:")
    print(f"  {FIG2_PATH}")
    print(f"  {FIG2_PDF_PATH}")


# ============================================================
# Smooth-JUNO fit diagnostics
# ============================================================

if RUN_SOLAR_SCAN:
    print()
    print("=" * 88)
    print("SMOOTH-JUNO FIT RESIDUALS")
    print("=" * 88)

    for result in configuration_results:
        key = result["cfg"]["key"]
        scan = result["scan"]

        reactor_residual = (
            scan["N_reactor_best_scan"][fit_mask]
            - N_obs_reactor[fit_mask]
        )

        total_residual = (
            scan["N_best_scan"][fit_mask]
            - N_plot_total[fit_mask]
        )

        data_residual = (
            scan["N_best_scan"][fit_mask]
            - N_obs_total[fit_mask]
        )

        print(f"\n{key}")
        print(
            "  reactor-only: "
            f"RMS = {np.sqrt(np.mean(reactor_residual**2)):.6f}, "
            f"max |residual| = {np.max(np.abs(reactor_residual)):.6f}"
        )
        print(
            "  smooth total: "
            f"RMS = {np.sqrt(np.mean(total_residual**2)):.6f}, "
            f"max |residual| = {np.max(np.abs(total_residual)):.6f}"
        )
        print(
            "  measured data:"
            f" RMS = {np.sqrt(np.mean(data_residual**2)):.6f}, "
            f"max |residual| = {np.max(np.abs(data_residual)):.6f}"
        )

    print("=" * 88)

print()
print("Fitting system: full nonlinear nuisance profiling with L-BFGS-B")
print(f"Likelihood target: {FIT_TARGET_NAME}")
print(f"Pull penalty weight: {PULL_PENALTY_WEIGHT:.3f}")


# ============================================================
# Final timing
# ============================================================

total_elapsed = perf_counter() - total_t0

print("\nTotal script time before opening plot window:")
print(f"  {total_elapsed:.2f} s")
print(f"  {total_elapsed / 60:.2f} min")
print("\nNote: time spent while the plot window is open is not included.")

plt.show()