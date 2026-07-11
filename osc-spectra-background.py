from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Project paths
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

# This lets the script work if it is either in the project root
# or inside a subfolder like background/
if (SCRIPT_DIR / "src").exists():
    ROOT = SCRIPT_DIR
else:
    ROOT = SCRIPT_DIR.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"
IMG_DIR = ROOT / "img"
IMG_DIR.mkdir(exist_ok=True)


DYB_PATH = DATA_DIR / "DYB_unfolded_spectra_tot_U235_Pu239.txt"
JUNO_PATH = DATA_DIR / "spect-fit.txt"
NONLINEARITY_PATH = DATA_DIR / "positron_nonlinearity.csv"

RAW_BG_PATH = DATA_DIR / "fig3_backgrounds_digitized_raw.csv"

OUT_FIG = IMG_DIR / "oscillated_spectrum_with_aligned_backgrounds.png"


# ============================================================
# Imports from src
# ============================================================

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


# ============================================================
# Constants
# ============================================================

kg_to_MeV = 5.61e29

m_p = 1.6726219e-27 * kg_to_MeV
m_n = 1.6749275e-27 * kg_to_MeV
m_e = 9.1093837e-31 * kg_to_MeV

Delta = m_n - m_p

sin2_theta12 = 0.308
sin2_theta13 = 0.02215

dm21 = 7.49e-5
dm31 = 2.513e-3


# ============================================================
# Background Table 1 normalization
# ============================================================

LIVE_DAYS = 59.1

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


# ============================================================
# Reactors
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

reactor_data["w"] = reactor_data["P_GWth"] / (
    4.0 * np.pi * reactor_data["L_km"] ** 2
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
# Model prompt-energy bins
# ============================================================

bin_width = 0.1

Epr_edges = np.arange(0.0, 10.0 + bin_width, bin_width)
Epr_centers = 0.5 * (Epr_edges[:-1] + Epr_edges[1:])


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


# ============================================================
# Oscillation and reactor contribution
# ============================================================

E_nu = np.linspace(1.81, 10.0, 2000)

Pee_weighted = np.zeros_like(E_nu)

for _, reactor in reactor_data.iterrows():
    L_km = reactor["L_km"]
    w = reactor["w"]

    Pee_r = neutrino_oscillation(
        E_nu,
        L_km,
        sin2_theta12,
        sin2_theta13,
        dm21,
        dm31,
    )

    Pee_weighted += w * Pee_r


# ============================================================
# Evaluate flux at central value, no pulls
# ============================================================

xi0 = np.zeros(nbin, dtype=float)

phi_E = np.asarray(phi_cont(E_nu, xi0), dtype=float).ravel()
phi_E = np.clip(phi_E, 0.0, None)

sigma = sigma_ibd(E_nu, Delta, m_e)

integrand_common = phi_E * sigma * Pee_weighted


# ============================================================
# Detector energy response
# ============================================================

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
# Load JUNO spectrum and normalize model reactor spectrum
# ============================================================

df_JUNO = pd.read_csv(JUNO_PATH, sep=r"\s+", header=None)

df_JUNO.columns = [
    "energy",
    "reactor_signal",
    "reactor_background",
    "data",
    "unoscillated_signal",
]

juno_energy = df_JUNO["energy"].to_numpy(dtype=float)
juno_reactor_signal = df_JUNO["reactor_signal"].to_numpy(dtype=float)
juno_reactor_plus_background = df_JUNO["reactor_background"].to_numpy(dtype=float)

# Peak-match the model reactor signal to the JUNO oscillated reactor signal.
C_norm = np.max(juno_reactor_signal)

Ni_nl = np.clip(Ni_nl, 0.0, None)

model_reactor = C_norm * Ni_nl / np.max(Ni_nl)


# ============================================================
# Read raw digitized Fig. 3 backgrounds
# ============================================================

df_raw_bg = pd.read_csv(RAW_BG_PATH)

df_raw_bg.columns = [str(c).strip() for c in df_raw_bg.columns]

for col in df_raw_bg.columns:
    df_raw_bg[col] = pd.to_numeric(df_raw_bg[col], errors="coerce")

df_raw_bg = df_raw_bg.dropna(subset=["E_prompt"])

E_bg_raw = df_raw_bg["E_prompt"].to_numpy(dtype=float)


# ============================================================
# Background helper functions
# ============================================================

def interpolate_to_bins(E_raw, y_raw, E_bins):
    """
    Interpolate raw digitized curve onto model bin centers.
    """

    E_raw = np.asarray(E_raw, dtype=float)
    y_raw = np.asarray(y_raw, dtype=float)

    good = np.isfinite(E_raw) & np.isfinite(y_raw)

    E = E_raw[good]
    y = y_raw[good]

    order = np.argsort(E)

    E = E[order]
    y = y[order]

    y_interp = np.interp(
        E_bins,
        E,
        y,
        left=0.0,
        right=0.0,
    )

    return np.clip(y_interp, 0.0, None)


def normalize_to_table1(shape_events_per_0p1, component_name):
    """
    Normalize a background shape so that its total number of events is:

        Table 1 rate x live days.

    Fig. 3 y-axis is events per 0.1 MeV.
    Since bin_width = 0.1 MeV, each value is already events per model bin.
    """

    shape_events_per_0p1 = np.asarray(shape_events_per_0p1, dtype=float)
    shape_events_per_0p1 = np.clip(shape_events_per_0p1, 0.0, None)

    counts_per_bin = shape_events_per_0p1 * (bin_width / 0.1)

    current_total = np.sum(counts_per_bin)

    if current_total <= 0.0:
        raise ValueError(f"{component_name} has zero total before normalization.")

    target_total = TABLE1_TOTAL_EVENTS[component_name]

    return counts_per_bin * target_total / current_total


def first_nonzero_energy(E, y, threshold_fraction=1.0e-4):
    """
    Return the first energy where y becomes non-negligible.

    threshold_fraction avoids tiny numerical tails from the detector response.
    """

    E = np.asarray(E, dtype=float)
    y = np.asarray(y, dtype=float)

    y_max = np.max(y)

    if y_max <= 0.0:
        return E[0]

    threshold = threshold_fraction * y_max

    idx = np.where(y > threshold)[0]

    if len(idx) == 0:
        return E[0]

    return E[idx[0]]


# ============================================================
# First interpolate backgrounds with no shift, only to find start
# ============================================================

Li_He_shape_pre = interpolate_to_bins(
    E_bg_raw,
    df_raw_bg["Li_He"].to_numpy(dtype=float),
    Epr_centers,
)

geoneutrinos_shape_pre = interpolate_to_bins(
    E_bg_raw,
    df_raw_bg["geoneutrinos"].to_numpy(dtype=float),
    Epr_centers,
)

world_reactors_shape_pre = interpolate_to_bins(
    E_bg_raw,
    df_raw_bg["world_reactors_digitized"].to_numpy(dtype=float),
    Epr_centers,
)

bi_po_shape_pre = interpolate_to_bins(
    E_bg_raw,
    df_raw_bg["bi_po"].to_numpy(dtype=float),
    Epr_centers,
)

others_shape_pre = interpolate_to_bins(
    E_bg_raw,
    df_raw_bg["others"].to_numpy(dtype=float),
    Epr_centers,
)

background_shape_pre = (
    Li_He_shape_pre
    + geoneutrinos_shape_pre
    + world_reactors_shape_pre
    + bi_po_shape_pre
    + others_shape_pre
)


# ============================================================
# Align the start of the backgrounds with the start of the spectrum
# ============================================================

spectrum_start = first_nonzero_energy(
    Epr_centers,
    model_reactor,
    threshold_fraction=1.0e-4,
)

background_start = first_nonzero_energy(
    Epr_centers,
    background_shape_pre,
    threshold_fraction=1.0e-4,
)

energy_shift = spectrum_start - background_start

E_bg_aligned = E_bg_raw + energy_shift

print("\nBackground alignment:")
print(f"  spectrum start before adding backgrounds = {spectrum_start:.4f} MeV")
print(f"  background start before shift            = {background_start:.4f} MeV")
print(f"  applied energy shift                     = {energy_shift:.4f} MeV")


# ============================================================
# Interpolate shifted backgrounds onto model bins
# ============================================================

Li_He_shape = interpolate_to_bins(
    E_bg_aligned,
    df_raw_bg["Li_He"].to_numpy(dtype=float),
    Epr_centers,
)

geoneutrinos_shape = interpolate_to_bins(
    E_bg_aligned,
    df_raw_bg["geoneutrinos"].to_numpy(dtype=float),
    Epr_centers,
)

world_reactors_shape = interpolate_to_bins(
    E_bg_aligned,
    df_raw_bg["world_reactors_digitized"].to_numpy(dtype=float),
    Epr_centers,
)

bi_po_shape = interpolate_to_bins(
    E_bg_aligned,
    df_raw_bg["bi_po"].to_numpy(dtype=float),
    Epr_centers,
)

others_shape = interpolate_to_bins(
    E_bg_aligned,
    df_raw_bg["others"].to_numpy(dtype=float),
    Epr_centers,
)


# ============================================================
# Normalize shifted, interpolated backgrounds to Table 1
# ============================================================

Li_He = normalize_to_table1(
    Li_He_shape,
    "Li_He",
)

geoneutrinos = normalize_to_table1(
    geoneutrinos_shape,
    "geoneutrinos",
)

world_reactors = normalize_to_table1(
    world_reactors_shape,
    "world_reactors",
)

bi_po = normalize_to_table1(
    bi_po_shape,
    "bi_po",
)

others = normalize_to_table1(
    others_shape,
    "others",
)

Total_background = (
    Li_He
    + geoneutrinos
    + world_reactors
    + bi_po
    + others
)

model_reactor_plus_background = model_reactor + Total_background


# ============================================================
# Print background totals
# ============================================================

print("\nTable 1 background totals:")
for name, total in TABLE1_TOTAL_EVENTS.items():
    print(f"  {name:16s}: {total:.6f}")

print("\nNormalized background totals after alignment:")
print(f"  {'Li_He':16s}: {np.sum(Li_He):.6f}")
print(f"  {'geoneutrinos':16s}: {np.sum(geoneutrinos):.6f}")
print(f"  {'world_reactors':16s}: {np.sum(world_reactors):.6f}")
print(f"  {'bi_po':16s}: {np.sum(bi_po):.6f}")
print(f"  {'others':16s}: {np.sum(others):.6f}")
print(f"  {'Total_background':16s}: {np.sum(Total_background):.6f}")


# ============================================================
# Plot
# ============================================================

plt.figure(figsize=(9.0, 5.6))

plt.plot(
    juno_energy,
    juno_reactor_signal,
    "--",
    lw=2.3,
    label="JUNO reactor signal",
)

plt.plot(
    juno_energy,
    juno_reactor_plus_background,
    "--",
    lw=2.3,
    label="JUNO reactor + background",
)

plt.plot(
    Epr_centers,
    model_reactor,
    "-",
    lw=2.0,
    label="Model reactor signal",
)

plt.plot(
    Epr_centers,
    model_reactor_plus_background,
    "-",
    lw=2.2,
    label="Model reactor + aligned backgrounds",
)

plt.plot(
    Epr_centers,
    Total_background,
    ":",
    lw=2.0,
    label="Added total background",
)

plt.xlabel(r"$E_{\rm pr}$ [MeV]")
plt.ylabel("Events per 0.1 MeV")
plt.title("Oscillated spectrum with aligned Fig. 3 backgrounds")
plt.xlim(0.6, 10.0)
plt.ylim(bottom=0.0)
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(OUT_FIG, dpi=300)

print(f"\nSaved figure to:")
print(f"  {OUT_FIG}")

plt.show()