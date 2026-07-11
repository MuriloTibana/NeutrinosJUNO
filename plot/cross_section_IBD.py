import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# Path setup 
# --------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.crossSectionIBD import sigma_ibd

# --------------------------------------------------
# Constants 
# --------------------------------------------------
kg_to_MeV = 5.61e29

m_p = 1.6726219e-27 * kg_to_MeV
m_n = 1.6749275e-27 * kg_to_MeV
m_e = 9.1093837e-31 * kg_to_MeV

Delta = m_n - m_p   

# Threshold from your "allowed = E_e > m_e" condition
E_thr = Delta + m_e
print(f"IBD threshold (your approximation): {E_thr:.6f} MeV")

# --------------------------------------------------
# Neutrino energy grid (MeV)
# --------------------------------------------------
E_nu = np.linspace(0.0, 20.0, 1000)

# --------------------------------------------------
# Compute cross section
# --------------------------------------------------
sigma = sigma_ibd(E_nu, Delta, m_e)

# --------------------------------------------------
# Plotting 
# --------------------------------------------------
plt.figure()
plt.plot(E_nu, sigma, label=r"$\sigma_{\mathrm{IBD}}(E_\nu)$")
plt.axvline(E_thr, linestyle="--", label=rf"Threshold $\Delta + m_e \approx {E_thr:.3f}$ MeV")

plt.xlabel(r"$E_\nu$ (MeV)")
plt.ylabel(r"$\sigma_{\mathrm{IBD}}$ (cm$^2$)")
plt.title("Inverse Beta Decay Cross Section")
plt.grid(True)
plt.legend()
plt.show()
