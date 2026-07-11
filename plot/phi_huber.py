import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

import matplotlib.pyplot as plt
from src.readDayaBay import read_total_flux
from src.crossSectionIBD import sigma_ibd
from src.phiHuber import phi_huber_weighted, phi_huber
# ---------------------------------------------------
# Definition of constants

kg_to_MeV = 5.61e29

m_p = 1.6726219e-27 * kg_to_MeV
m_n = 1.6749275e-27 * kg_to_MeV
m_e = 9.1093837e-31 * kg_to_MeV

Delta = m_n - m_p

path = "data/DYB_unfolded_spectra_tot_U235_Pu239.txt"

df_total = read_total_flux(path, "Total")

alpha = {
    "U235":  np.array([4.367, -4.577, 2.100, -5.294e-1, 6.186e-2, -2.777e-3]),
    "Pu239": np.array([4.757, -5.392, 2.563, -6.596e-1, 7.820e-2, -3.536e-3]),
    "Pu241": np.array([2.990, -2.882, 1.278, -3.343e-1, 3.905e-2, -1.754e-3]),
}

f = {"U235": 0.564, "U238": 0.076, "Pu239": 0.304, "Pu241": 0.056}

phi_h = phi_huber_weighted(df_total["E_center"], f, alpha)
sig_IBD = sigma_ibd(df_total["E_center"], Delta, m_e)

huber_ibd_weighted = phi_h * sig_IBD * 1e43  # ~10^-43 cm^2/fission/MeV

# --- Normalize shapes for fair overlay ---

dyb_area = np.trapezoid(df_total["Flux"], df_total["E_center"])
hub_area = np.trapezoid(huber_ibd_weighted, df_total["E_center"])

# --- Plot 1: approximate absolute overlay ---
plt.figure()
plt.errorbar(df_total["E_center"], df_total["Flux"], yerr=df_total["Error"], fmt="o",
             label="DYB unfolded Total")
plt.plot(df_total["E_center"], huber_ibd_weighted, label="Huber ref (U235+Pu239+Pu241, IBD-weighted)")
plt.xlabel("Neutrino Energy Eν (MeV)")
plt.ylabel("10⁻⁴³ cm² / fission / MeV")
plt.title("DYB unfolded Total vs Huber reference (approx. absolute)")
plt.legend()


# --- Plot 2: phi_huber only ---
plt.figure()
plt.plot(df_total["E_center"], phi_h, label=r"$\phi_{\mathrm{Huber}}(E_\nu)$ (weighted mix)")
plt.xlabel(r"Neutrino Energy $E_\nu$ (MeV)")
plt.ylabel(r"Flux ")  
plt.title("Huber Reference Spectrum")
plt.grid(True)
plt.legend()
plt.show()
