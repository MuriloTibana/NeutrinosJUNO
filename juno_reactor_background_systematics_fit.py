#!/usr/bin/env python3
from __future__ import annotations

# ============================================================
# JUNO-like reactor spectrum with digitized backgrounds
#
# This script:
#   1. Builds a Daya Bay-based continuous reactor antineutrino flux
#   2. Computes no-oscillation and oscillated JUNO-like prompt spectra
#   3. Loads digitized background components from digitized_backgrounds.csv
#   4. Uses digitized backgrounds exactly as they appear in the CSV
#   5. Adds backgrounds to the oscillated reactor spectrum
#   6. Applies reactor/background normalization pulls
#   7. Fits those pulls to the JUNO reactor + background reference spectrum
#   8. Shows diagnostic plots
#
# Backgrounds are always digitized.
# No calculated-background option.
# No digitized-background rescaling option.
# ============================================================

from math import erf
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

# The digitized background graph has y-axis units:
#     events per 0.02 MeV
#
# The digitized background values are used exactly as they appear.
# No conversion to events per 0.1 MeV is applied.
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

C_noosc = float(np.max(df_JUNO["unoscillated_signal"]))
C_react = float(np.max(df_JUNO["reactor_signal"]))


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
xi0 = np.zeros(nbin, dtype=float)

phi_E = np.asarray(phi_cont(E_nu, xi0), dtype=float).ravel()
phi_E = np.clip(phi_E, 0.0, None)

sig = sigma_ibd(E_nu, Delta, m_e)


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
# Nonlinear detector response
# ============================================================

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


def F_nl(Epr):
    Epr = np.asarray(Epr, dtype=float)
    Ecl = np.clip(Epr, E_pts[0], E_pts[-1])
    return _Fnl(Ecl)


def sigma_E(E, a=res_a, b=res_b):
    E = np.asarray(E, dtype=float)
    E = np.clip(E, 1e-6, None)
    return np.sqrt(a * a * E + b * b * E * E)


def gaussian_bin_prob(mu, sig, lo, hi):
    sig = np.asarray(sig, dtype=float)
    sig = np.clip(sig, 1e-12, None)

    z_hi = (hi - mu) / (np.sqrt(2.0) * sig)
    z_lo = (lo - mu) / (np.sqrt(2.0) * sig)

    erf_vec = np.vectorize(erf)

    return 0.5 * (erf_vec(z_hi) - erf_vec(z_lo))


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
):
    E_nu = np.asarray(E_nu, dtype=float)

    Evis = E_nu - Delta + m_e
    Epr0 = alpha * Evis + beta

    mu = Epr0 * F_nl(Epr0)
    sigE = sigma_E(mu, a=a, b=b)

    Ri = np.zeros_like(E_nu, dtype=float)
    mask = Evis > 0.0

    Ri[mask] = gaussian_bin_prob(mu[mask], sigE[mask], Ei_lo, Ei_hi)

    return Ri


# ============================================================
# Compute reactor spectra
# ============================================================

def compute_spectrum_nl(use_osc=False, target_norm=None):
    Ni = np.zeros_like(Epr_centers, dtype=float)

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

            integrand = phi_E * sig * Pee * Ri
            total_i += w * np.trapezoid(integrand, E_nu)

        Ni[i] = total_i

    if target_norm is not None:
        peak = np.max(Ni)
        if peak > 0:
            Ni = target_norm * Ni / peak

    return Ni


Ni_noosc_nl = compute_spectrum_nl(use_osc=False, target_norm=C_noosc)
Ni_osc_nl = compute_spectrum_nl(use_osc=True, target_norm=C_react)


# ============================================================
# Digitized backgrounds only
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

    plot_components = {
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
        "ylabel": f"events per {DIGITIZED_BKG_BIN_WIDTH:.2f} MeV",
    }

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
    }

    return background_dict, plot_components


background_dict, background_plot_dict = load_digitized_backgrounds()

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

B_total = (
    B_Geo
    + B_World
    + B_Acc
    + B_LiHe
    + B_BiPo
    + B_AtmNC
    + B_FastN
    + B_DoubleN
    + B_C13an
)

Ni_noosc_with_bkg = Ni_noosc_nl + B_total
Ni_osc_with_bkg = Ni_osc_nl + B_total


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
            "Total",
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
            float(np.sum(B_total)),
        ],
    }
)

print("\nDigitized background summary:")
print(summary.to_string(index=False))


# ============================================================
# Reactor normalization systematic
# ============================================================

TABLE2_REACTOR_UNCERTAINTIES_PERCENT = {
    "Target protons": 1.0,
    "Reference spectrum": 1.2,
    "Thermal power": 0.5,
    "Fission fraction": 0.6,
    "Spent nuclear fuel": 0.3,
    "Non-equilibrium": 0.2,
    "Different fission fraction": 0.1,
}

sigma_reactor_norm_percent = np.sqrt(
    np.sum([
        value ** 2
        for value in TABLE2_REACTOR_UNCERTAINTIES_PERCENT.values()
    ])
)

SIGMA_REACTOR_NORM = sigma_reactor_norm_percent / 100.0

print("\nReactor normalization uncertainty:")
for key, value in TABLE2_REACTOR_UNCERTAINTIES_PERCENT.items():
    print(f"  {key:28s}: {value:.2f}%")

print(f"  {'Total quadrature':28s}: {sigma_reactor_norm_percent:.3f}%")
print(f"  Fractional sigma: {SIGMA_REACTOR_NORM:.5f}")


# ============================================================
# Background normalization uncertainties
# ============================================================

SIGMA_BG_NORM = {
    "LiHe": 0.33,
    "Geo": 0.56,
    "World": 0.10,
    "BiPo": 0.56,
    "Other": 1.00,
}

B_Other = B_C13an + B_FastN + B_DoubleN + B_AtmNC + B_Acc

S_reactor_nominal = Ni_osc_nl.copy()

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

N_total_nominal = S_reactor_nominal + B_total_grouped

print("\nBackground normalization pull uncertainties:")
for name, sigma in SIGMA_BG_NORM.items():
    print(f"  {name:8s}: {100.0 * sigma:.1f}%")

print("\nBackground grouping check:")
print(f"  grouped total  = {np.sum(B_total_grouped):.6f}")
print(f"  original total = {np.sum(B_total):.6f}")
print(f"  difference     = {np.sum(B_total_grouped - B_total):.6e}")


# ============================================================
# Prediction with normalization pulls
# ============================================================

def predict_with_norm_pulls(theta):
    xi_reactor = theta[0]
    xi_LiHe = theta[1]
    xi_Geo = theta[2]
    xi_World = theta[3]
    xi_BiPo = theta[4]
    xi_Other = theta[5]

    S_sys = (1.0 + SIGMA_REACTOR_NORM * xi_reactor) * S_reactor_nominal

    B_LiHe_sys = (1.0 + SIGMA_BG_NORM["LiHe"] * xi_LiHe) * B_components["LiHe"]
    B_Geo_sys = (1.0 + SIGMA_BG_NORM["Geo"] * xi_Geo) * B_components["Geo"]
    B_World_sys = (1.0 + SIGMA_BG_NORM["World"] * xi_World) * B_components["World"]
    B_BiPo_sys = (1.0 + SIGMA_BG_NORM["BiPo"] * xi_BiPo) * B_components["BiPo"]
    B_Other_sys = (1.0 + SIGMA_BG_NORM["Other"] * xi_Other) * B_components["Other"]

    N_pred = (
        S_sys
        + B_LiHe_sys
        + B_Geo_sys
        + B_World_sys
        + B_BiPo_sys
        + B_Other_sys
    )

    return np.clip(N_pred, 1e-12, None)


def total_prediction_with_reactor_norm_pull(S_reactor, B_total_input, xi_norm):
    return (1.0 + SIGMA_REACTOR_NORM * xi_norm) * S_reactor + B_total_input


# ============================================================
# JUNO curves and fit target
# ============================================================

E_juno = df_JUNO["energy"].to_numpy()

juno_react = df_JUNO["reactor_signal"].to_numpy()
juno_noosc = df_JUNO["unoscillated_signal"].to_numpy()
juno_react_bk = df_JUNO["reactor_background"].to_numpy()

juno_react_interp = np.interp(x_model, E_juno, juno_react)
juno_noosc_interp = np.interp(x_model, E_juno, juno_noosc)
juno_react_bk_interp = np.interp(x_model, E_juno, juno_react_bk)

res_react = Ni_osc_nl - juno_react_interp
res_noosc = Ni_noosc_nl - juno_noosc_interp
res_react_bk = N_total_nominal - juno_react_bk_interp

N_obs = juno_react_bk_interp.copy()


def chi2_with_norm_pulls(theta):
    N_pred = predict_with_norm_pulls(theta)

    stat_var = np.maximum(N_pred, 1.0)

    chi2_stat = np.sum((N_obs - N_pred) ** 2 / stat_var)
    chi2_pull = np.sum(theta ** 2)

    return chi2_stat + chi2_pull


pull_names = [
    "Reactor norm",
    "LiHe norm",
    "Geo norm",
    "World norm",
    "BiPo norm",
    "Other norm",
]

theta0 = np.zeros(6, dtype=float)

result = minimize(
    chi2_with_norm_pulls,
    x0=theta0,
    method="L-BFGS-B",
    bounds=[(-5.0, 5.0)] * len(theta0),
)

theta_best = result.x
N_best_all_norm = predict_with_norm_pulls(theta_best)

print("\nBest-fit normalization pulls:")
print(f"  success = {result.success}")
print(f"  chi2_min = {result.fun:.4f}")

for name, val in zip(pull_names, theta_best):
    print(f"  {name:14s}: {val:+.4f}")

print("\nBest-fit fractional shifts:")
print(f"  Reactor: {100.0 * SIGMA_REACTOR_NORM * theta_best[0]:+.4f}%")
print(f"  LiHe:    {100.0 * SIGMA_BG_NORM['LiHe']  * theta_best[1]:+.4f}%")
print(f"  Geo:     {100.0 * SIGMA_BG_NORM['Geo']   * theta_best[2]:+.4f}%")
print(f"  World:   {100.0 * SIGMA_BG_NORM['World'] * theta_best[3]:+.4f}%")
print(f"  BiPo:    {100.0 * SIGMA_BG_NORM['BiPo']  * theta_best[4]:+.4f}%")
print(f"  Other:   {100.0 * SIGMA_BG_NORM['Other'] * theta_best[5]:+.4f}%")


# ============================================================
# Plot 0: digitized background components
# ============================================================

plt.figure(figsize=(8.5, 5.8))

plt.plot(x_model, background_plot_dict["B_Geo"], lw=2.0, label="Geo")
plt.plot(x_model, background_plot_dict["B_Acc"], lw=2.0, label="Accidentals")
plt.plot(x_model, background_plot_dict["B_LiHe"], lw=2.0, label="Li9/He8")
plt.plot(x_model, background_plot_dict["B_C13an"], lw=2.0, label="C13(alpha,n)O16")
plt.plot(x_model, background_plot_dict["B_FastN"], lw=2.0, label="Fast neutrons")
plt.plot(x_model, background_plot_dict["B_World"], lw=2.0, label="World reactor")
plt.plot(x_model, background_plot_dict["B_AtmNC"], lw=2.0, label="Atmospheric NC")

if background_plot_dict["B_BiPo"] is not None and np.any(background_plot_dict["B_BiPo"] > 0):
    plt.plot(x_model, background_plot_dict["B_BiPo"], lw=2.0, label="BiPo")

if background_plot_dict["B_DoubleN"] is not None and np.any(background_plot_dict["B_DoubleN"] > 0):
    plt.plot(x_model, background_plot_dict["B_DoubleN"], lw=2.0, label="Double neutrons")

plt.plot(
    x_model,
    background_plot_dict["B_total"],
    lw=3.0,
    color="black",
    label="Total digitized background",
)

plt.xlabel("Epr [MeV]", fontsize=16)
plt.ylabel(background_plot_dict["ylabel"], fontsize=16)
plt.title("Digitized Background Components", fontsize=16)
plt.xlim(0.8, 10.0)
plt.ylim(bottom=0.0)
plt.legend(fontsize=8.5, ncol=2)
plt.tight_layout()


# ============================================================
# Plot 1: model vs JUNO prediction
# ============================================================

plt.figure(figsize=(8.5, 5.8))

plt.plot(x_model, Ni_noosc_nl, "--", lw=2.4, label="Model: no oscillations")
plt.plot(x_model, Ni_osc_nl, "--", lw=2.4, label="Model: reactor")
plt.plot(x_model, N_total_nominal, "--", lw=2.4, label="Model: reactor + digitized BK")

plt.plot(E_juno, juno_noosc, "-", lw=2.0, label="JUNO prediction: no oscillations")
plt.plot(E_juno, juno_react, "-", lw=2.0, label="JUNO prediction: reactor")
plt.plot(E_juno, juno_react_bk, "-", lw=2.0, label="JUNO prediction: reactor + BK")

plt.xlabel("Epr [MeV]", fontsize=18)
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
# Plot 2: reactor normalization systematic
# ============================================================

N_total_minus_1sigma_R = total_prediction_with_reactor_norm_pull(
    S_reactor_nominal,
    B_total_grouped,
    -1.0,
)

N_total_plus_1sigma_R = total_prediction_with_reactor_norm_pull(
    S_reactor_nominal,
    B_total_grouped,
    +1.0,
)

N_total_minus_2sigma_R = total_prediction_with_reactor_norm_pull(
    S_reactor_nominal,
    B_total_grouped,
    -2.0,
)

N_total_plus_2sigma_R = total_prediction_with_reactor_norm_pull(
    S_reactor_nominal,
    B_total_grouped,
    +2.0,
)

plt.figure(figsize=(8.5, 5.8))

plt.fill_between(
    x_model,
    N_total_minus_2sigma_R,
    N_total_plus_2sigma_R,
    alpha=0.16,
    label=f"Reactor norm. +/- 2 sigma ({2.0 * sigma_reactor_norm_percent:.2f}%)",
)

plt.fill_between(
    x_model,
    N_total_minus_1sigma_R,
    N_total_plus_1sigma_R,
    alpha=0.30,
    label=f"Reactor norm. +/- 1 sigma ({sigma_reactor_norm_percent:.2f}%)",
)

plt.plot(
    x_model,
    N_total_nominal,
    lw=2.5,
    label="Nominal model: reactor + digitized BK",
)

plt.plot(
    E_juno,
    juno_react_bk,
    "k--",
    lw=2.0,
    alpha=0.8,
    label="JUNO prediction: reactor + BK",
)

plt.xlabel("Epr [MeV]", fontsize=18)
plt.ylabel("events per 0.1 MeV", fontsize=18)
plt.title("Systematic: Reactor Normalization", fontsize=16)
plt.xlim(0.8, 10.0)
plt.legend(fontsize=9, frameon=True)
plt.tight_layout()


# ============================================================
# Plot 3: background normalization systematics
# ============================================================

plt.figure(figsize=(8.5, 5.8))

plt.plot(
    x_model,
    N_total_nominal,
    color="black",
    lw=2.6,
    label="Nominal model: reactor + digitized BK",
)

for j, name in enumerate(["LiHe", "Geo", "World", "BiPo", "Other"], start=1):
    theta = np.zeros(6, dtype=float)
    theta[j] = +1.0

    N_shifted = predict_with_norm_pulls(theta)

    plt.plot(
        x_model,
        N_shifted,
        lw=1.8,
        label=f"{name} norm. +1 sigma",
    )

plt.plot(
    E_juno,
    juno_react_bk,
    "k--",
    lw=2.0,
    alpha=0.8,
    label="JUNO prediction: reactor + BK",
)

plt.xlabel("Epr [MeV]", fontsize=18)
plt.ylabel("events per 0.1 MeV", fontsize=18)
plt.title("Systematics: Background Normalizations", fontsize=16)
plt.xlim(0.8, 10.0)
plt.legend(fontsize=8.5, frameon=True)
plt.tight_layout()


# ============================================================
# Plot 4: fitting result
# ============================================================

plt.figure(figsize=(8.5, 5.8))

plt.plot(
    x_model,
    N_total_nominal,
    lw=2.3,
    ls="--",
    label="Nominal model",
)

plt.plot(
    x_model,
    N_best_all_norm,
    lw=2.6,
    label="Best fit with norm. pulls",
)

plt.plot(
    E_juno,
    juno_react_bk,
    color="gray",
    lw=2.0,
    alpha=0.85,
    label="JUNO prediction: reactor + BK",
)

plt.xlabel("Epr [MeV]", fontsize=18)
plt.ylabel("events per 0.1 MeV", fontsize=18)
plt.title("Fit with Reactor and Background Normalization Pulls", fontsize=16)
plt.xlim(0.8, 10.0)
plt.legend(fontsize=9.5, frameon=True)
plt.tight_layout()


# ============================================================
# Plot 5: fitting diagnostics
# ============================================================

plt.figure(figsize=(8.2, 5.2))

plt.bar(pull_names, theta_best)

plt.axhline(0.0, color="black", ls="--", lw=1.2)
plt.axhline(+1.0, color="gray", ls=":", lw=1.2)
plt.axhline(-1.0, color="gray", ls=":", lw=1.2)

plt.ylabel("Best-fit pull value xi_j", fontsize=15)
plt.title("Best-Fit Normalization Pulls", fontsize=16)
plt.xticks(rotation=30, ha="right")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()


# ============================================================
# Show all plots
# ============================================================

plt.show()