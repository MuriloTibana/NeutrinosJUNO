import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.readDayaBay import read_total_flux
import matplotlib.pyplot as plt

path = "data/DYB_unfolded_spectra_tot_U235_Pu239.txt"

df_u235  = read_total_flux(path, "U235")
df_pu239 = read_total_flux(path, "Pu239")
df_total = read_total_flux(path, "Total")

plt.figure()

plt.errorbar(df_u235["E_center"], df_u235["Flux"],
             yerr=df_u235["Error"], fmt="o", label="U235")

plt.errorbar(df_pu239["E_center"], df_pu239["Flux"],
             yerr=df_pu239["Error"], fmt="o", label="Pu239")

plt.errorbar(df_total["E_center"], df_total["Flux"],
             yerr=df_total["Error"], fmt="o", label="Total")


plt.xlabel(r"Neutrino Energy averaged bins $E_\nu$ (MeV)")
plt.ylabel(r"Flux $\Phi_i^0 (10^{-43}\text{cm}^2/\text{fission/MeV})$")
plt.title("Unfolded Reactor Antineutrino Spectra")
plt.legend()
plt.show()
