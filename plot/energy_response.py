import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.energyResponse import R_i, R_i_nl


# ============================================================
# Constants in MeV
# ============================================================

m_e = 0.51099895
m_p = 938.2720813
m_n = 939.5654133

Delta = m_n - m_p


# ============================================================
# Neutrino energy grid
# ============================================================

E_nu = np.linspace(1.8, 10.0, 1000)


# ============================================================
# Choose one prompt-energy bin
# ============================================================
# Example:
#     3.0 MeV <= E_pr < 3.1 MeV

Ei_lo = 3.0
Ei_hi = 3.1


# ============================================================
# Detector response parameters
# ============================================================

res_a = 0.033
res_b = 0.01
res_c = 0.0

prompt_alpha = 1.0
prompt_beta = 0.0

NONLINEARITY_PATH = "data/positron_nonlinearity.csv"


# ============================================================
# Compute response functions
# ============================================================

Ri_base = R_i(
    E_nu,
    Ei_lo,
    Ei_hi,
    a=res_a,
    b=res_b,
    c=res_c,
    sigma_prescription="midpoint",
)

Ri_nl = R_i_nl(
    E_nu,
    Ei_lo,
    Ei_hi,
    Delta=Delta,
    m_e=m_e,
    a=res_a,
    b=res_b,
    c=res_c,
    alpha=prompt_alpha,
    beta=prompt_beta,
    nonlinearity_path=NONLINEARITY_PATH,
)


# ============================================================
# Plot R_i and R_i_nl
# ============================================================

plt.figure(figsize=(7.5, 4.8))

plt.plot(
    E_nu,
    Ri_base,
    lw=2,
    label=r"Baseline response $R_i(E_\nu)$",
)

plt.plot(
    E_nu,
    Ri_nl,
    lw=2,
    label=r"Nonlinear response $R_i^{\rm nl}(E_\nu)$",
)

plt.xlabel(r"True neutrino energy $E_\nu$ (MeV)")
plt.ylabel(r"Response probability")
plt.title(
    rf"Detector response for ${Ei_lo:.1f} \leq E_{{\rm pr}} < {Ei_hi:.1f}$ MeV"
)

plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()