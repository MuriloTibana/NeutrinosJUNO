#!/usr/bin/env python3
from __future__ import annotations

# ============================================================
# JUNO-like reactor spectrum with digitized backgrounds
# and full systematic pulls
#
# Includes:
#   1. Reactor normalization pull
#   2. Background normalization pulls
#   3. LiHe shape pull
#   4. 25 reactor flux pulls from the Daya Bay covariance
#   5. Energy scale pull
#   6. Energy bias pull
#   7. Energy resolution pull
#
# Digitized backgrounds are used exactly as loaded.
# No digitized-background bin-width rescaling is applied.
# ============================================================

from math import isfinite
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.special import erf
from scipy.interpolate import PchipInterpolator
from scipy.signal import savgol_filter
from scipy.optimize import minimize

from src.readDayaBay import (
    read_total_flux,
    read_covariance_matrix,
    recast_covariance_matrix,
)
from src.deltaSplines import create_delta_basis
from src.continuous_flux import build_continuous_flux_model
from src.phiHuber import phi_huber_weighted
from src.crossSectionIBD import sigma_ibd


# ============================================================
# Paths and options
# ============================================================

JUNO_PATH = "data/spect-mine.txt"
DYB_PATH = "data/DYB_unfolded_spectra_tot_U235_Pu239.txt"
DIGITIZED_BKG_PATH = "digitized_backgrounds.csv"
NONLINEARITY_PATH = "data/positron_nonlinearity.csv"

RUN_FIT = True
FIT_MAXITER = 120

DIGITIZED_BKG_BIN_WIDTH = 0.02


# ============================================================
# Detector and energy settings
# ============================================================

prompt_alpha = 1.0
prompt_beta = 0.0

res_a = 0.033
res_b = 0.01

E_nu = np.linspace(1.81, 10.0, 2000)

bin_width = 0.1
Epr_edges = prompt_alpha * np.arange(0.0, 10.0 + bin_width, bin_width) + prompt_beta
Epr_centers = 0.5 * (Epr_edges[:-1] + Epr_edges[1:])
x_model = Epr_centers - prompt_beta


# ============================================================
# Oscillation parameters
# ============================================================

sin2_theta12 = 0.303
sin2_theta13 = 0.02203

dm21 = 7.41e-5
dm31 = 2.437e-3


# ============================================================
# Constants
# ============================================================

kg_to_MeV = 5.61e29

m_p = 1.6726219e-27 * kg_to_MeV
m_n = 1.6749275e-27 * kg_to_MeV
m_e = 9.1093837e-31 * kg_to_MeV

Delta = m_n - m_p


# ============================================================
# Systematic uncertainties
# ============================================================

# Reactor normalization uncertainty:
# Screenshot says sigma_norm = 1.8%.
SIGMA_REACTOR_NORM = 0.018

# Background normalization uncertainties:
# LiHe, Geo, world-reactor, BiPo, other.
SIGMA_BG_NORM = {
    "LiHe": 0.33,
    "Geo": 0.56,
    "World": 0.10,
    "BiPo": 0.56,
    "Other": 1.00,
}

# LiHe shape uncertainty:
# 20% at 1 MeV, linearly proportional to prompt energy.
SIGMA_LIHE_SHAPE_AT_1MEV = 0.20

# Energy scale systematics:
# sigma_scl = sigma_bias = 0.1%.
SIGMA_ENERGY_SCALE = 0.001
SIGMA_ENERGY_BIAS = 0.001

# Energy resolution uncertainty:
# sigma_res = 5%.
SIGMA_ENERGY_RES = 0.05


# ============================================================
# Load JUNO-like reference file
# ============================================================

df_JUNO = pd.read_csv(JUNO_PATH, sep=r"\s+", header=None)

df_JUNO.columns = [
    "energy",
    "reactor_signal",
    "reactor_background",
    "data",
    "unoscillated_signal",
]

E_juno = df_JUNO["energy"].to_numpy(dtype=float)

juno_react = df_JUNO["reactor_signal"].to_numpy(dtype=float)
juno_noosc = df_JUNO["unoscillated_signal"].to_numpy(dtype=float)
juno_react_bk = df_JUNO["reactor_background"].to_numpy(dtype=float)

C_noosc = float(np.max(juno_noosc))
C_react = float(np.max(juno_react))

juno_react_interp = np.interp(x_model, E_juno, juno_react, left=0.0, right=0.0)
juno_noosc_interp = np.interp(x_model, E_juno, juno_noosc, left=0.0, right=0.0)
juno_react_bk_interp = np.interp(x_model, E_juno, juno_react_bk, left=0.0, right=0.0)


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
    {"name": "DayaBay-effective", "P_GWth": 17.4, "L_km": 215.0},
]

reactor_data = pd.DataFrame(reactors)

km_to_cm = 1.0e5
reactor_data["L_cm"] = reactor_data["L_km"] * km_to_cm

reactor_data["w"] = reactor_data["P_GWth"] / (
    4.0 * np.pi * reactor_data["L_cm"] ** 2
)


# ============================================================
# Huber coefficients and fission fractions
# ============================================================

alpha_huber = {
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
# Build continuous flux model from Daya Bay unfolded spectrum
# ============================================================

df_total = read_total_flux(DYB_PATH, "Total")

C_ij = read_covariance_matrix(DYB_PATH)
Psi_ik = recast_covariance_matrix(C_ij)

Phi0 = df_total["Flux"].to_numpy(dtype=float)
E_high = df_total["E_high"].to_numpy(dtype=float)
E_low = df_total["E_low"].to_numpy(dtype=float)
E_center = df_total["E_center"].to_numpy(dtype=float)

delta, splines, I = create_delta_basis(E_center)

phi_cont, extras = build_continuous_flux_model(
    E_center=E_center,
    E_low=E_low,
    E_high=E_high,
    Phi0=Phi0,
    Psi_ik=Psi_ik,
    delta=delta,
    phi_huber_weighted=phi_huber_weighted,
    sigma_ibd=sigma_ibd,
    frac=frac,
    alpha=alpha_huber,
    Delta=Delta,
    m_e=m_e,
    N_int=500,
)

nbin = int(extras["nbin"])

print(f"\nNumber of reactor flux pulls: {nbin}")

xi_flux_zero = np.zeros(nbin, dtype=float)

phi_E_nominal = np.asarray(phi_cont(E_nu, xi_flux_zero), dtype=float).ravel()
phi_E_nominal = np.clip(phi_E_nominal, 0.0, None)

sig_ibd = sigma_ibd(E_nu, Delta, m_e)


# ============================================================
# Oscillation probability
# ============================================================

def Pee_3nu_vac(E_nu, L_km, sin2_theta12, sin2_theta13, dm21, dm31):
    E_nu = np.asarray(E_nu, dtype=float)

    s12_2 = sin2_theta12
    c12_2 = 1.0 - s12_2

    s13_2 = sin2_theta13
    c13_2 = 1.0 - s13_2

    sin2_2theta12 = 4.0 * s12_2 * c12_2
    sin2_2theta13 = 4.0 * s13_2 * c13_2

    dm32 = dm31 - dm21

    phase21 = 1.267e3 * dm21 * L_km / E_nu
    phase31 = 1.267e3 * dm31 * L_km / E_nu
    phase32 = 1.267e3 * dm32 * L_km / E_nu

    Pee = (
        1.0
        - (c13_2 ** 2) * sin2_2theta12 * np.sin(phase21) ** 2
        - sin2_2theta13 * (
            c12_2 * np.sin(phase31) ** 2
            + s12_2 * np.sin(phase32) ** 2
        )
    )

    return Pee


# ============================================================
# Nonlinearity model
# ============================================================

def load_or_make_nonlinearity_points():
    path = Path(NONLINEARITY_PATH)

    if path.exists():
        df = pd.read_csv(path)

        if "E_pr" not in df.columns or "F_nl" not in df.columns:
            raise ValueError(
                f"{NONLINEARITY_PATH} must contain columns 'E_pr' and 'F_nl'."
            )

        E_pts = df["E_pr"].to_numpy(dtype=float)
        F_pts = df["F_nl"].to_numpy(dtype=float)

        print(f"\nLoaded positron nonlinearity from {path.resolve()}")

    else:
        print("\nWarning: nonlinearity CSV not found.")
        print("Using placeholder points instead.")

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
    E_pts = unique_E
    F_pts = F_pts[unique_idx]

    return E_pts, F_pts


E_pts, F_pts = load_or_make_nonlinearity_points()

if len(F_pts) >= 7:
    window_length = 7
    F_pts_sm = savgol_filter(
        F_pts,
        window_length=window_length,
        polyorder=2,
        mode="interp",
    )
else:
    F_pts_sm = F_pts.copy()

_Fnl = PchipInterpolator(E_pts, F_pts_sm, extrapolate=False)


def F_nl(Epr):
    Epr = np.asarray(Epr, dtype=float)
    Ecl = np.clip(Epr, E_pts[0], E_pts[-1])
    return _Fnl(Ecl)


def sigma_E(E, a=res_a, b=res_b):
    E = np.asarray(E, dtype=float)
    E = np.clip(E, 1e-6, None)

    return np.sqrt(a * a * E + b * b * E * E)


def gaussian_bin_prob(mu, sig, lo, hi):
    mu = np.asarray(mu, dtype=float)
    sig = np.asarray(sig, dtype=float)
    sig = np.clip(sig, 1e-12, None)

    z_hi = (hi - mu) / (np.sqrt(2.0) * sig)
    z_lo = (lo - mu) / (np.sqrt(2.0) * sig)

    return 0.5 * (erf(z_hi) - erf(z_lo))


def R_i_nl(
    E_nu,
    Ei_lo,
    Ei_hi,
    Delta,
    m_e,
    alpha=prompt_alpha,
    beta=prompt_beta,
    a=res_a,
    b=res_b,
    xi_scl=0.0,
    xi_bias=0.0,
    xi_res=0.0,
):
    """
    Nonlinear detector response with energy-scale and resolution pulls.

    Evis = E_nu - Delta + m_e

    Epr0 = alpha Evis + beta

    Energy scale systematic:

        Epr_tilde = Epr0 [
            (1 + sigma_scl xi_scl) F_nl(Epr0)
            + sigma_bias xi_bias
        ]

    Resolution systematic:

        sigma_tilde = (1 + sigma_res xi_res) sigma
    """

    E_nu = np.asarray(E_nu, dtype=float)

    Evis = E_nu - Delta + m_e
    Epr0 = alpha * Evis + beta

    scale_factor = 1.0 + SIGMA_ENERGY_SCALE * xi_scl
    bias_term = SIGMA_ENERGY_BIAS * xi_bias

    mu = Epr0 * (scale_factor * F_nl(Epr0) + bias_term)

    sigE_nominal = sigma_E(mu, a=a, b=b)
    sigE = (1.0 + SIGMA_ENERGY_RES * xi_res) * sigE_nominal
    sigE = np.clip(sigE, 1e-12, None)

    Ri = np.zeros_like(E_nu, dtype=float)
    mask = Evis > 0.0

    Ri[mask] = gaussian_bin_prob(mu[mask], sigE[mask], Ei_lo, Ei_hi)

    return Ri


# ============================================================
# Flux with Daya Bay pull modes
# ============================================================

def evaluate_flux_with_pulls(xi_flux=None):
    if xi_flux is None:
        xi_flux = np.zeros(nbin, dtype=float)

    xi_flux = np.asarray(xi_flux, dtype=float)

    if xi_flux.size != nbin:
        raise ValueError(f"Expected {nbin} flux pulls, got {xi_flux.size}.")

    phi = np.asarray(phi_cont(E_nu, xi_flux), dtype=float).ravel()
    phi = np.clip(phi, 0.0, None)

    return phi


# ============================================================
# Reactor spectrum computation
# ============================================================

def compute_reactor_spectrum_raw(
    use_osc=False,
    xi_flux=None,
    xi_scl=0.0,
    xi_bias=0.0,
    xi_res=0.0,
):
    """
    Compute raw reactor spectrum before final display normalization.

    The final scale is applied later using fixed nominal scale factors.
    This prevents systematic pulls from being normalized away.
    """

    Ni = np.zeros_like(Epr_centers, dtype=float)

    phi_E = evaluate_flux_with_pulls(xi_flux)

    for i in range(len(Epr_centers)):
        Ei_lo = Epr_edges[i]
        Ei_hi = Epr_edges[i + 1]

        Ri = R_i_nl(
            E_nu,
            Ei_lo,
            Ei_hi,
            Delta=Delta,
            m_e=m_e,
            alpha=prompt_alpha,
            beta=prompt_beta,
            a=res_a,
            b=res_b,
            xi_scl=xi_scl,
            xi_bias=xi_bias,
            xi_res=xi_res,
        )

        total_i = 0.0

        for _, rx in reactor_data.iterrows():
            L_km = rx["L_km"]
            w = rx["w"]

            if use_osc:
                Pee = Pee_3nu_vac(
                    E_nu=E_nu,
                    L_km=L_km,
                    sin2_theta12=sin2_theta12,
                    sin2_theta13=sin2_theta13,
                    dm21=dm21,
                    dm31=dm31,
                )
            else:
                Pee = np.ones_like(E_nu)

            integrand = phi_E * sig_ibd * Pee * Ri
            total_i += w * np.trapezoid(integrand, E_nu)

        Ni[i] = total_i

    return Ni


def safe_peak_scale(target_peak, spectrum):
    peak = float(np.max(spectrum))

    if peak <= 0.0 or not isfinite(peak):
        raise ValueError("Cannot normalize spectrum. Peak is zero or invalid.")

    return target_peak / peak


print("\nComputing nominal reactor spectra...")

Ni_noosc_raw_nominal = compute_reactor_spectrum_raw(use_osc=False)
Ni_osc_raw_nominal = compute_reactor_spectrum_raw(use_osc=True)

SCALE_NOOSC = safe_peak_scale(C_noosc, Ni_noosc_raw_nominal)
SCALE_OSC = safe_peak_scale(C_react, Ni_osc_raw_nominal)

Ni_noosc_nl = SCALE_NOOSC * Ni_noosc_raw_nominal
Ni_osc_nl = SCALE_OSC * Ni_osc_raw_nominal


# ============================================================
# Digitized backgrounds
# ============================================================

def normalize_column_name(name):
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def find_digitized_column(df, possible_names):
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
    df,
    x_source,
    x_target,
    possible_names,
    label,
    allow_missing=True,
):
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


def load_digitized_backgrounds():
    csv_path = Path(DIGITIZED_BKG_PATH)

    if not csv_path.exists():
        alt_path = Path("data") / DIGITIZED_BKG_PATH

        if alt_path.exists():
            csv_path = alt_path
        else:
            raise FileNotFoundError(
                f"Could not find digitized background CSV at "
                f"'{DIGITIZED_BKG_PATH}' or '{alt_path}'."
            )

    df = pd.read_csv(csv_path)
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(axis=0, how="all")
    df = df.sort_values(df.columns[0]).reset_index(drop=True)

    x_source = df.iloc[:, 0].to_numpy(dtype=float)

    print("\nUsing digitized backgrounds only")
    print(f"Loaded: {csv_path.resolve()}")
    print(f"Digitized bin width: {DIGITIZED_BKG_BIN_WIDTH:.3f} MeV")
    print("Digitized backgrounds are used exactly as loaded.")
    print("No bin-width rescaling is applied.")

    B_Geo_raw = interpolate_digitized_component(
        df,
        x_source,
        x_model,
        possible_names=["Geoneutrinos", "Geo", "GeoNeutrinos"],
        label="Geo",
    )

    B_Acc_raw = interpolate_digitized_component(
        df,
        x_source,
        x_model,
        possible_names=["Accidentals", "Accidental", "Acc"],
        label="Acc",
    )

    B_LiHe_raw = interpolate_digitized_component(
        df,
        x_source,
        x_model,
        possible_names=["Li9_He8", "9Li_8He", "LiHe", "9Li8He"],
        label="LiHe",
    )

    B_C13an_raw = interpolate_digitized_component(
        df,
        x_source,
        x_model,
        possible_names=["C13_alpha_n_O16", "13C_alpha_n_16O", "C13an", "13C"],
        label="C13an",
    )

    B_FastN_raw = interpolate_digitized_component(
        df,
        x_source,
        x_model,
        possible_names=["FastNeutrons", "Fast_neutrons", "FastN"],
        label="FastN",
    )

    B_World_raw = interpolate_digitized_component(
        df,
        x_source,
        x_model,
        possible_names=[
            "WorldReactor_antinu",
            "World_reactor_nubar",
            "WorldReactor",
            "World",
        ],
        label="World",
    )

    B_AtmNC_raw = interpolate_digitized_component(
        df,
        x_source,
        x_model,
        possible_names=["AtmosphericNC", "Atmospheric_NC", "AtmNC"],
        label="AtmNC",
    )

    B_BiPo_raw = interpolate_digitized_component(
        df,
        x_source,
        x_model,
        possible_names=["BiPo", "Bi214Po214", "214Bi214Po"],
        label="BiPo",
        allow_missing=True,
    )

    B_DoubleN_raw = interpolate_digitized_component(
        df,
        x_source,
        x_model,
        possible_names=["DoubleNeutrons", "Double_neutrons", "DoubleN"],
        label="DoubleN",
        allow_missing=True,
    )

    B_total_raw = (
        B_Geo_raw
        + B_World_raw
        + B_Acc_raw
        + B_LiHe_raw
        + B_BiPo_raw
        + B_AtmNC_raw
        + B_FastN_raw
        + B_DoubleN_raw
        + B_C13an_raw
    )

    background_dict = {
        "B_Geo_U": B_Geo_raw.copy(),
        "B_Geo_Th": np.zeros_like(x_model, dtype=float),
        "B_Geo": B_Geo_raw,
        "B_World": B_World_raw,
        "B_Acc": B_Acc_raw,
        "B_LiHe": B_LiHe_raw,
        "B_BiPo": B_BiPo_raw,
        "B_AtmNC": B_AtmNC_raw,
        "B_FastN": B_FastN_raw,
        "B_DoubleN": B_DoubleN_raw,
        "B_C13an": B_C13an_raw,
        "B_total": B_total_raw,
    }

    return background_dict


background_dict = load_digitized_backgrounds()

B_Geo_U = background_dict["B_Geo_U"]
B_Geo_Th = background_dict["B_Geo_Th"]
B_Geo = background_dict["B_Geo"]

B_World = background_dict["B_World"]
B_Acc = background_dict["B_Acc"]
B_LiHe = background_dict["B_LiHe"]
B_BiPo = background_dict["B_BiPo"]
B_AtmNC = background_dict["B_AtmNC"]
B_FastN = background_dict["B_FastN"]
B_DoubleN = background_dict["B_DoubleN"]
B_C13an = background_dict["B_C13an"]

B_total = background_dict["B_total"]

B_Other = B_C13an + B_FastN + B_DoubleN + B_AtmNC + B_Acc

B_components = {
    "LiHe": B_LiHe,
    "Geo": B_Geo,
    "World": B_World,
    "BiPo": B_BiPo,
    "Other": B_Other,
}

B_total_grouped = (
    B_components["LiHe"]
    + B_components["Geo"]
    + B_components["World"]
    + B_components["BiPo"]
    + B_components["Other"]
)

S_reactor_nominal = Ni_osc_nl.copy()
N_total_nominal = S_reactor_nominal + B_total_grouped

Ni_noosc_with_bkg = Ni_noosc_nl + B_total_grouped
Ni_osc_with_bkg = Ni_osc_nl + B_total_grouped


# ============================================================
# Background sanity table
# ============================================================

summary = pd.DataFrame(
    {
        "component": [
            "Geo_U",
            "Geo_Th",
            "Geo_total",
            "World",
            "Acc",
            "LiHe",
            "BiPo",
            "C13an",
            "FastN",
            "DoubleN",
            "AtmNC",
            "Other",
            "Total grouped",
            "Total raw",
        ],
        "sum_over_bins": [
            float(np.sum(B_Geo_U)),
            float(np.sum(B_Geo_Th)),
            float(np.sum(B_Geo)),
            float(np.sum(B_World)),
            float(np.sum(B_Acc)),
            float(np.sum(B_LiHe)),
            float(np.sum(B_BiPo)),
            float(np.sum(B_C13an)),
            float(np.sum(B_FastN)),
            float(np.sum(B_DoubleN)),
            float(np.sum(B_AtmNC)),
            float(np.sum(B_Other)),
            float(np.sum(B_total_grouped)),
            float(np.sum(B_total)),
        ],
    }
)

print("\nDigitized background summary:")
print(summary.to_string(index=False))

print("\nSystematic uncertainties:")
print(f"  Reactor norm:      {100.0 * SIGMA_REACTOR_NORM:.2f}%")
print(f"  LiHe norm:         {100.0 * SIGMA_BG_NORM['LiHe']:.1f}%")
print(f"  Geo norm:          {100.0 * SIGMA_BG_NORM['Geo']:.1f}%")
print(f"  World norm:        {100.0 * SIGMA_BG_NORM['World']:.1f}%")
print(f"  BiPo norm:         {100.0 * SIGMA_BG_NORM['BiPo']:.1f}%")
print(f"  Other norm:        {100.0 * SIGMA_BG_NORM['Other']:.1f}%")
print(f"  LiHe shape at 1 MeV: {100.0 * SIGMA_LIHE_SHAPE_AT_1MEV:.1f}%")
print(f"  Energy scale:      {100.0 * SIGMA_ENERGY_SCALE:.3f}%")
print(f"  Energy bias:       {100.0 * SIGMA_ENERGY_BIAS:.3f}%")
print(f"  Energy resolution: {100.0 * SIGMA_ENERGY_RES:.1f}%")


# ============================================================
# Prediction with all pulls
# ============================================================

lihe_shape_fraction = SIGMA_LIHE_SHAPE_AT_1MEV * x_model


def positive_scale(sigma, xi):
    return max(0.0, 1.0 + sigma * xi)


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
    ]
)

pull_index = {name: i for i, name in enumerate(pull_names)}


def unpack_all_pulls(theta):
    theta = np.asarray(theta, dtype=float)

    expected = len(pull_names)

    if theta.size != expected:
        raise ValueError(f"Expected {expected} pulls, got {theta.size}.")

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

    return {
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
    }


def predict_with_all_pulls(theta):
    pulls = unpack_all_pulls(theta)

    S_reactor_raw = compute_reactor_spectrum_raw(
        use_osc=True,
        xi_flux=pulls["xi_flux"],
        xi_scl=pulls["xi_scl"],
        xi_bias=pulls["xi_bias"],
        xi_res=pulls["xi_res"],
    )

    S_reactor_pulled = SCALE_OSC * S_reactor_raw

    S_sys = (
        positive_scale(SIGMA_REACTOR_NORM, pulls["xi_reactor"])
        * S_reactor_pulled
    )

    B_LiHe_sys = (
        positive_scale(SIGMA_BG_NORM["LiHe"], pulls["xi_LiHe"])
        * B_components["LiHe"]
    )

    B_LiHe_sys = B_LiHe_sys * (
        1.0 + pulls["xi_LiHe_shape"] * lihe_shape_fraction
    )
    B_LiHe_sys = np.clip(B_LiHe_sys, 0.0, None)

    B_Geo_sys = (
        positive_scale(SIGMA_BG_NORM["Geo"], pulls["xi_Geo"])
        * B_components["Geo"]
    )

    B_World_sys = (
        positive_scale(SIGMA_BG_NORM["World"], pulls["xi_World"])
        * B_components["World"]
    )

    B_BiPo_sys = (
        positive_scale(SIGMA_BG_NORM["BiPo"], pulls["xi_BiPo"])
        * B_components["BiPo"]
    )

    B_Other_sys = (
        positive_scale(SIGMA_BG_NORM["Other"], pulls["xi_Other"])
        * B_components["Other"]
    )

    N_pred = (
        S_sys
        + B_LiHe_sys
        + B_Geo_sys
        + B_World_sys
        + B_BiPo_sys
        + B_Other_sys
    )

    return np.clip(N_pred, 1e-12, None)


def make_theta(**kwargs):
    theta = np.zeros(len(pull_names), dtype=float)

    for name, value in kwargs.items():
        if name not in pull_index:
            raise KeyError(f"Unknown pull name: {name}")

        theta[pull_index[name]] = value

    return theta


# ============================================================
# Chi-square fit
# ============================================================

N_obs = juno_react_bk_interp.copy()


def chi2_with_all_pulls(theta):
    N_pred = predict_with_all_pulls(theta)

    stat_var = np.maximum(N_obs, 1.0)

    chi2_stat = np.sum((N_obs - N_pred) ** 2 / stat_var)

    # All pulls are Gaussian with unit variance.
    chi2_pull = np.sum(np.asarray(theta, dtype=float) ** 2)

    return chi2_stat + chi2_pull


theta_best = np.zeros(len(pull_names), dtype=float)
N_best_all = N_total_nominal.copy()
result = None

if RUN_FIT:
    print("\nRunning fit with all systematic pulls...")
    print(f"Total number of pulls: {len(pull_names)}")

    theta0 = np.zeros(len(pull_names), dtype=float)

    result = minimize(
        chi2_with_all_pulls,
        x0=theta0,
        method="L-BFGS-B",
        bounds=[(-5.0, 5.0)] * len(theta0),
        options={"maxiter": FIT_MAXITER},
    )

    theta_best = result.x
    N_best_all = predict_with_all_pulls(theta_best)

    print("\nBest-fit pulls including all systematics:")
    print(f"  success = {result.success}")
    print(f"  message = {result.message}")
    print(f"  chi2_min = {result.fun:.4f}")

    for name, val in zip(pull_names, theta_best):
        print(f"  {name:20s}: {val:+.4f}")

    pulls_best = unpack_all_pulls(theta_best)

    print("\nBest-fit fractional shifts:")
    print(
        f"  Reactor norm:      "
        f"{100.0 * SIGMA_REACTOR_NORM * pulls_best['xi_reactor']:+.4f}%"
    )
    print(
        f"  LiHe norm:         "
        f"{100.0 * SIGMA_BG_NORM['LiHe'] * pulls_best['xi_LiHe']:+.4f}%"
    )
    print(
        f"  Geo norm:          "
        f"{100.0 * SIGMA_BG_NORM['Geo'] * pulls_best['xi_Geo']:+.4f}%"
    )
    print(
        f"  World norm:        "
        f"{100.0 * SIGMA_BG_NORM['World'] * pulls_best['xi_World']:+.4f}%"
    )
    print(
        f"  BiPo norm:         "
        f"{100.0 * SIGMA_BG_NORM['BiPo'] * pulls_best['xi_BiPo']:+.4f}%"
    )
    print(
        f"  Other norm:        "
        f"{100.0 * SIGMA_BG_NORM['Other'] * pulls_best['xi_Other']:+.4f}%"
    )
    print(
        f"  Energy scale:      "
        f"{100.0 * SIGMA_ENERGY_SCALE * pulls_best['xi_scl']:+.4f}%"
    )
    print(
        f"  Energy bias:       "
        f"{100.0 * SIGMA_ENERGY_BIAS * pulls_best['xi_bias']:+.4f}%"
    )
    print(
        f"  Energy resolution: "
        f"{100.0 * SIGMA_ENERGY_RES * pulls_best['xi_res']:+.4f}%"
    )


# ============================================================
# Diagnostic spectra for +1 sigma pulls
# ============================================================

theta_energy_scale = make_theta(**{"Energy scale": +1.0})
theta_energy_bias = make_theta(**{"Energy bias": +1.0})
theta_energy_res = make_theta(**{"Energy resolution": +1.0})
theta_lihe_shape = make_theta(**{"LiHe shape": +1.0})

theta_flux1 = np.zeros(len(pull_names), dtype=float)
theta_flux1[pull_index["Flux mode 1"]] = +1.0

N_energy_scale = predict_with_all_pulls(theta_energy_scale)
N_energy_bias = predict_with_all_pulls(theta_energy_bias)
N_energy_res = predict_with_all_pulls(theta_energy_res)
N_lihe_shape = predict_with_all_pulls(theta_lihe_shape)
N_flux1 = predict_with_all_pulls(theta_flux1)


# ============================================================
# Plot 0: Nonlinearity curve
# ============================================================

plt.figure(figsize=(7.4, 4.8))

Etest = np.linspace(E_pts[0], E_pts[-1], 500)

plt.plot(E_pts, F_pts, "o", label="Input points")
plt.plot(Etest, F_nl(Etest), lw=2.2, label="Smoothed interpolation")

plt.xlabel(r"$E_{\rm pr}$ [MeV]", fontsize=15)
plt.ylabel(r"$F_{\rm nl}(E_{\rm pr})$", fontsize=15)
plt.title("Positron Nonlinearity Function", fontsize=16)
plt.grid(True)
plt.legend()
plt.tight_layout()


# ============================================================
# Plot 1: Digitized background components
# ============================================================

plt.figure(figsize=(8.5, 5.8))

plt.plot(x_model, B_Geo, lw=2.0, label="Geo")
plt.plot(x_model, B_Acc, lw=2.0, label="Accidentals")
plt.plot(x_model, B_LiHe, lw=2.0, label="Li9/He8")
plt.plot(x_model, B_C13an, lw=2.0, label="C13(alpha,n)O16")
plt.plot(x_model, B_FastN, lw=2.0, label="Fast neutrons")
plt.plot(x_model, B_World, lw=2.0, label="World reactor")
plt.plot(x_model, B_AtmNC, lw=2.0, label="Atmospheric NC")

if np.any(B_BiPo > 0):
    plt.plot(x_model, B_BiPo, lw=2.0, label="BiPo")

if np.any(B_DoubleN > 0):
    plt.plot(x_model, B_DoubleN, lw=2.0, label="Double neutrons")

plt.plot(
    x_model,
    B_total_grouped,
    lw=3.0,
    color="black",
    label="Total grouped background",
)

plt.xlabel(r"$E_{\rm pr}$ [MeV]", fontsize=16)
plt.ylabel(f"events per {DIGITIZED_BKG_BIN_WIDTH:.2f} MeV", fontsize=16)
plt.title("Digitized Background Components", fontsize=16)
plt.xlim(0.8, 10.0)
plt.ylim(bottom=0.0)
plt.legend(fontsize=8.5, ncol=2)
plt.tight_layout()


# ============================================================
# Plot 2: Model vs JUNO
# ============================================================

plt.figure(figsize=(8.5, 5.8))

plt.plot(x_model, Ni_noosc_nl, "--", lw=2.4, label="Model: no oscillations")
plt.plot(x_model, Ni_osc_nl, "--", lw=2.4, label="Model: reactor")
plt.plot(x_model, N_total_nominal, "--", lw=2.4, label="Model: reactor + BK")

plt.plot(E_juno, juno_noosc, "-", lw=2.0, label="JUNO: no oscillations")
plt.plot(E_juno, juno_react, "-", lw=2.0, label="JUNO: reactor")
plt.plot(E_juno, juno_react_bk, "-", lw=2.0, label="JUNO: reactor + BK")

plt.xlabel(r"$E_{\rm pr}$ [MeV]", fontsize=18)
plt.ylabel("events per 0.1 MeV", fontsize=18)
plt.title("Model vs JUNO Prediction", fontsize=16)
plt.xlim(0.8, 10.0)
plt.ylim(
    0.0,
    1.08 * max(
        np.max(Ni_noosc_nl),
        np.max(N_total_nominal),
        np.max(juno_noosc),
        np.max(juno_react_bk),
    ),
)
plt.legend(fontsize=9, frameon=True)
plt.tight_layout()


# ============================================================
# Plot 3: Reactor normalization systematic
# ============================================================

N_reactor_minus_1 = (
    positive_scale(SIGMA_REACTOR_NORM, -1.0) * S_reactor_nominal
    + B_total_grouped
)

N_reactor_plus_1 = (
    positive_scale(SIGMA_REACTOR_NORM, +1.0) * S_reactor_nominal
    + B_total_grouped
)

N_reactor_minus_2 = (
    positive_scale(SIGMA_REACTOR_NORM, -2.0) * S_reactor_nominal
    + B_total_grouped
)

N_reactor_plus_2 = (
    positive_scale(SIGMA_REACTOR_NORM, +2.0) * S_reactor_nominal
    + B_total_grouped
)

plt.figure(figsize=(8.5, 5.8))

plt.fill_between(
    x_model,
    N_reactor_minus_2,
    N_reactor_plus_2,
    alpha=0.16,
    label=f"Reactor norm. +/- 2 sigma ({200.0 * SIGMA_REACTOR_NORM:.2f}%)",
)

plt.fill_between(
    x_model,
    N_reactor_minus_1,
    N_reactor_plus_1,
    alpha=0.30,
    label=f"Reactor norm. +/- 1 sigma ({100.0 * SIGMA_REACTOR_NORM:.2f}%)",
)

plt.plot(
    x_model,
    N_total_nominal,
    lw=2.5,
    label="Nominal model: reactor + BK",
)

plt.plot(
    E_juno,
    juno_react_bk,
    "k--",
    lw=2.0,
    alpha=0.8,
    label="JUNO: reactor + BK",
)

plt.xlabel(r"$E_{\rm pr}$ [MeV]", fontsize=18)
plt.ylabel("events per 0.1 MeV", fontsize=18)
plt.title("Systematic: Reactor Normalization", fontsize=16)
plt.xlim(0.8, 10.0)
plt.legend(fontsize=9, frameon=True)
plt.tight_layout()


# ============================================================
# Plot 4: Background normalization systematics
# ============================================================

plt.figure(figsize=(8.5, 5.8))

plt.plot(
    x_model,
    N_total_nominal,
    color="black",
    lw=2.6,
    label="Nominal model: reactor + BK",
)

for name in ["LiHe norm", "Geo norm", "World norm", "BiPo norm", "Other norm"]:
    theta = make_theta(**{name: +1.0})
    N_shifted = predict_with_all_pulls(theta)

    plt.plot(
        x_model,
        N_shifted,
        lw=1.8,
        label=f"{name} +1 sigma",
    )

plt.plot(
    E_juno,
    juno_react_bk,
    "k--",
    lw=2.0,
    alpha=0.8,
    label="JUNO: reactor + BK",
)

plt.xlabel(r"$E_{\rm pr}$ [MeV]", fontsize=18)
plt.ylabel("events per 0.1 MeV", fontsize=18)
plt.title("Systematics: Background Normalizations", fontsize=16)
plt.xlim(0.8, 10.0)
plt.legend(fontsize=8.5, frameon=True)
plt.tight_layout()


# ============================================================
# Plot 5: Shape, flux, and detector systematics
# ============================================================

plt.figure(figsize=(8.5, 5.8))

plt.plot(
    x_model,
    N_total_nominal,
    color="black",
    lw=2.5,
    label="Nominal model",
)

plt.plot(x_model, N_lihe_shape, lw=1.8, label="LiHe shape +1 sigma")
plt.plot(x_model, N_flux1, lw=1.8, label="Flux mode 1 +1 sigma")
plt.plot(x_model, N_energy_scale, lw=1.8, label="Energy scale +1 sigma")
plt.plot(x_model, N_energy_bias, lw=1.8, label="Energy bias +1 sigma")
plt.plot(x_model, N_energy_res, lw=1.8, label="Energy resolution +1 sigma")

plt.plot(
    E_juno,
    juno_react_bk,
    "k--",
    lw=2.0,
    alpha=0.8,
    label="JUNO: reactor + BK",
)

plt.xlabel(r"$E_{\rm pr}$ [MeV]", fontsize=18)
plt.ylabel("events per 0.1 MeV", fontsize=18)
plt.title("Shape, Flux, and Detector Systematics", fontsize=16)
plt.xlim(0.8, 10.0)
plt.legend(fontsize=8.5, frameon=True)
plt.tight_layout()


# ============================================================
# Plot 6: Fit result
# ============================================================

plt.figure(figsize=(8.5, 5.8))

plt.plot(
    x_model,
    N_total_nominal,
    lw=2.3,
    ls="--",
    label="Nominal model",
)

if RUN_FIT:
    plt.plot(
        x_model,
        N_best_all,
        lw=2.6,
        label="Best fit with all pulls",
    )

plt.plot(
    E_juno,
    juno_react_bk,
    color="gray",
    lw=2.0,
    alpha=0.85,
    label="JUNO: reactor + BK",
)

plt.xlabel(r"$E_{\rm pr}$ [MeV]", fontsize=18)
plt.ylabel("events per 0.1 MeV", fontsize=18)
plt.title("Fit with All Systematic Pulls", fontsize=16)
plt.xlim(0.8, 10.0)
plt.legend(fontsize=9.5, frameon=True)
plt.tight_layout()


# ============================================================
# Plot 7: Pull diagnostics
# ============================================================

if RUN_FIT:
    main_pull_names = [
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
    ]

    main_pull_values = np.array([
        theta_best[pull_index[name]]
        for name in main_pull_names
    ])

    plt.figure(figsize=(8.5, 5.2))

    plt.bar(main_pull_names, main_pull_values)

    plt.axhline(0.0, color="black", ls="--", lw=1.2)
    plt.axhline(+1.0, color="gray", ls=":", lw=1.2)
    plt.axhline(-1.0, color="gray", ls=":", lw=1.2)

    plt.ylabel(r"Best-fit pull value $\xi_j$", fontsize=15)
    plt.title("Best-Fit Main Systematic Pulls", fontsize=16)
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    flux_pull_values = np.array([
        theta_best[pull_index[f"Flux mode {k + 1}"]]
        for k in range(nbin)
    ])

    plt.figure(figsize=(8.5, 5.2))

    plt.bar(np.arange(1, nbin + 1), flux_pull_values)

    plt.axhline(0.0, color="black", ls="--", lw=1.2)
    plt.axhline(+1.0, color="gray", ls=":", lw=1.2)
    plt.axhline(-1.0, color="gray", ls=":", lw=1.2)

    plt.xlabel("Flux mode index", fontsize=15)
    plt.ylabel(r"Best-fit pull value $\xi_k^{\rm flux}$", fontsize=15)
    plt.title("Best-Fit Reactor Flux Pulls", fontsize=16)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()


# ============================================================
# Show all plots
# ============================================================

plt.show()