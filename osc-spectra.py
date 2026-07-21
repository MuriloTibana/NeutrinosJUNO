import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from src.readDayaBay import read_total_flux, read_covariance_matrix, recast_covariance_matrix
from src.deltaSplines import create_delta_basis
from src.continuous_flux import build_continuous_flux_model
from src.phiHuber import phi_huber_weighted
from src.crossSectionIBD import sigma_ibd
from src.energyResponse import compute_spectrum_with_response
from src.oscillation import neutrino_oscillation

# ============================================================
# Output
# ============================================================
figure_path = "img/osc_noosc_JUNO_Model.png"

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
# Reactor list
# ============================================================
reactors = [
    {"name": "Taishan-1",        "P_GWth": 4.6,  "L_km": 52.77},
    {"name": "Taishan-2",        "P_GWth": 4.6,  "L_km": 52.64},
    {"name": "Yangjiang-1",      "P_GWth": 2.9,  "L_km": 52.74},
    {"name": "Yangjiang-2",      "P_GWth": 2.9,  "L_km": 52.82},
    {"name": "Yangjiang-3",      "P_GWth": 2.9,  "L_km": 52.41},
    {"name": "Yangjiang-4",      "P_GWth": 2.9,  "L_km": 52.49},
    {"name": "Yangjiang-5",      "P_GWth": 2.9,  "L_km": 52.11},
    {"name": "Yangjiang-6",      "P_GWth": 2.9,  "L_km": 52.19},
    {"name": "DayaBay-effective","P_GWth": 17.4, "L_km": 215.0},
]

reactor_data = pd.DataFrame(reactors)
reactor_data["w"] = reactor_data["P_GWth"] / (4 * np.pi * reactor_data["L_km"]**2)

# ============================================================
# Huber coefficients and fission fractions
# ============================================================
alpha = {
    "U235":  np.array([4.367, -4.577, 2.100, -5.294e-1, 6.186e-2, -2.777e-3]),
    "Pu239": np.array([4.757, -5.392, 2.563, -6.596e-1, 7.820e-2, -3.536e-3]),
    "Pu241": np.array([2.990, -2.882, 1.278, -3.343e-1, 3.905e-2, -1.754e-3]),
}

frac = {
    "U235": 0.564,
    "U238": 0.076,
    "Pu239": 0.304,
    "Pu241": 0.056,
}

# ============================================================
# Build continuous flux model from Daya Bay
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
# Neutrino energy grid
# ============================================================
E_nu = np.linspace(1.81, 10.0, 2000)

# ============================================================
# Compute weighted survival probabilities
# ============================================================
Pee_weighted_osc = np.zeros_like(E_nu)
Pee_weighted_noosc = np.zeros_like(E_nu)

for _, reactor in reactor_data.iterrows():
    L_km = reactor["L_km"]
    w = reactor["w"]

    Pee_osc = neutrino_oscillation(E_nu, L_km, sin2_theta12, sin2_theta13, dm21, dm31)
    Pee_noosc = np.ones_like(E_nu)

    Pee_weighted_osc += w * Pee_osc
    Pee_weighted_noosc += w * Pee_noosc

# ============================================================
# Evaluate continuous flux at xi = 0
# ============================================================
xi0 = np.zeros(nbin, dtype=float)
phi_E = np.asarray(phi_cont(E_nu, xi0), dtype=float).ravel()
phi_E = np.clip(phi_E, 0.0, None)

# ============================================================
# IBD cross section
# ============================================================
sigma = sigma_ibd(E_nu, Delta, m_e)

integrand_osc = phi_E * sigma * Pee_weighted_osc
integrand_noosc = phi_E * sigma * Pee_weighted_noosc

# ============================================================
# Detector energy response
# ============================================================
NONLINEARITY_PATH = "data/positron_nonlinearity.csv"

bin_width = 0.1
Epr_edges = np.arange(0.0, 10.0 + bin_width, bin_width)
Epr_centers = 0.5 * (Epr_edges[:-1] + Epr_edges[1:])

res_a = 0.033
res_b = 0.01
res_c = 0.0

prompt_alpha = 1.0
prompt_beta = 0.0

Epr_centers, Ni_osc = compute_spectrum_with_response(
    E_nu, integrand_osc, Epr_edges, "nonlinear",
    True, res_a, res_b, res_c, prompt_alpha, prompt_beta, NONLINEARITY_PATH)

_, Ni_noosc = compute_spectrum_with_response(
    E_nu, integrand_noosc, Epr_edges, "nonlinear",
    True, res_a, res_b, res_c, prompt_alpha, prompt_beta, NONLINEARITY_PATH
)

# ============================================================
# Load JUNO reference spectra
# ============================================================
JUNO_path = "data/spect-fit.txt"

df_JUNO = pd.read_csv(JUNO_path, sep=r"\s+", header=None)
df_JUNO.columns = ["energy", "reactor_signal", "reactor_background", "data", "unoscillated_signal"]

# Normalize each model spectrum to the corresponding JUNO curve
C_norm_osc = np.max(df_JUNO["reactor_signal"])
C_norm_noosc = np.max(df_JUNO["unoscillated_signal"])

model_osc = C_norm_osc * Ni_osc
model_noosc = C_norm_noosc * Ni_noosc

# ============================================================
# Plot
# ============================================================
Path("img").mkdir(exist_ok=True)

plt.figure(figsize=(8.2, 5.2))

# JUNO reference curves
plt.plot(df_JUNO["energy"], df_JUNO["reactor_signal"], "--", lw=2.5, label="osc. JUNO")
plt.plot(df_JUNO["energy"], df_JUNO["unoscillated_signal"], "--", lw=2.5, label="no osc. JUNO")

# Model curves
plt.plot(Epr_centers, model_osc, "-", lw=2, label="osc. Model")
plt.plot(Epr_centers, model_noosc, "-", lw=2, label="no osc. Model")

plt.xlabel(r"$E_{\rm pr}$ [MeV]")
plt.ylabel("Events per 0.1 MeV")
plt.title("Oscillated and Non-Oscillated Spectra")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(figure_path, dpi=300)
plt.show()

print(f"Saved figure to: {figure_path}")