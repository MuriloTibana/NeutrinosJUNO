import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import numpy as np
from src.fit import calculateSpectrum

with np.load("results/10_10_data_poisson.npz") as results:

    sin2_theta12_grid = results["sin2_theta12_grid"].copy()
    dm21_grid = results["dm21_grid"].copy()
    chi2_grid = results["chi2_grid"].copy()
    z_norm_grid = results["z_norm_grid"].copy()
    success_grid = results["success_grid"].copy()
    SIGMA_REACTOR_RATE = results["SIGMA_REACTOR_RATE"].copy()
    Total_background = results["Total_background"].copy()

df_JUNO = pd.read_csv("data/spect-fit.txt", sep=r"\s+", header=None)
df_JUNO.columns = ["energy", "reactor_signal", "reactor_background", "data", "unoscillated_signal"]
JUNO_energy = df_JUNO["energy"].to_numpy(dtype=float)
JUNO_data = df_JUNO["data"].to_numpy(dtype=float)
JUNO_background = (df_JUNO["reactor_background"] - df_JUNO["reactor_signal"]).to_numpy(dtype=float)
JUNO_reactor = df_JUNO["reactor_background"].to_numpy(dtype=float)

# Calculating best parameters
best_flat_index = np.nanargmin(chi2_grid)
best_dm_index, best_theta_index = (np.unravel_index(best_flat_index, chi2_grid.shape))

best_sin2_theta12 = sin2_theta12_grid[best_theta_index]
best_dm21 = dm21_grid[best_dm_index]
best_chi2 = chi2_grid[best_dm_index, best_theta_index]
best_z_norm = z_norm_grid[best_dm_index, best_theta_index]

best_reactor_factor = 1.0 + SIGMA_REACTOR_RATE * best_z_norm

delta_chi2_grid = chi2_grid - np.nanmin(chi2_grid)

# Recomputing spectrum
Epr_best, spectrum_best = calculateSpectrum(best_sin2_theta12, best_dm21)

best_reactor_nominal = np.interp(JUNO_energy, Epr_best, spectrum_best + Total_background, left=0.0, right=0.0)
best_total_prediction = best_reactor_factor * best_reactor_nominal

# Plotting 
fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(JUNO_energy, JUNO_data, marker="o", linestyle="none", markersize=4, label="JUNO data")
ax.step(JUNO_energy, JUNO_reactor, "k--", where="mid", linewidth=1.8, label="Published JUNO reactor signal")
ax.step(JUNO_energy, best_reactor_nominal, where="mid", linewidth=1.5, label="Model reactor before pull")
ax.step(JUNO_energy, best_total_prediction, where="mid", linewidth=2.0, label="Model reactor + fixed background after pull")

ax.set_xlabel(r"Prompt energy $E_{\mathrm{prompt}}$ [MeV]")
ax.set_ylabel("Events per 0.1 MeV")

ax.set_title(
    rf"$\sin^2\theta_{{12}}={best_sin2_theta12:.4f}$, "
    rf"$\Delta m^2_{{21}}={best_dm21:.3e}\ \mathrm{{eV}}^2$"
    "\n"
    rf"$\chi^2_{{\min}}={best_chi2:.3f}$, "
    rf"$z_{{\mathrm{{norm}}}}={best_z_norm:+.3f}$")

ax.grid(alpha=0.25)
ax.legend()
fig.tight_layout()
plt.show()


# Meshgrid plot
plt.figure()
plt.contourf(sin2_theta12_grid, dm21_grid, delta_chi2_grid)
plt.xlabel(r"$\sin^2\theta_{12}$")
plt.ylabel(r"$\Delta m_{21}^2$")


contour_levels = [2.30, 6.18, 11.83]

fig, ax = plt.subplots(figsize=(7.2, 6.8))

cs = ax.contour(
    sin2_theta12_grid,
    dm21_grid * 1e5,
    delta_chi2_grid,
    levels=contour_levels,
    colors="deeppink",
    linewidths=1.8)

ax.clabel(cs, fmt={2.30: r"$1\sigma$", 6.18: r"$2\sigma$", 11.83: r"$3\sigma$"})
ax.plot(best_sin2_theta12, best_dm21 * 1e5, "k*", markersize=10)

ax.set_xlabel(r"$\sin^2\theta_{12}$")
ax.set_ylabel(r"$\Delta m^2_{21}\ [10^{-5}\,\mathrm{eV}^2]$")

fig.tight_layout()
plt.show()

theta_min_at_each_dm = sin2_theta12_grid[np.nanargmin(chi2_grid, axis=1)]

for dm21_value, theta_value in zip(dm21_grid, theta_min_at_each_dm):
    print(f"{dm21_value * 1e5:.4f}  {theta_value:.6f}")

ridge_slope = np.polyfit(dm21_grid * 1e5, theta_min_at_each_dm, 1)[0]
print("Ridge slope:", ridge_slope)