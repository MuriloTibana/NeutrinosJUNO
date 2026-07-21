import numpy as np
import matplotlib.pyplot as plt

from src.fit import calculateSpectrum
N_THETA12 = 5
N_DM21 = 5

total_points = N_THETA12 * N_DM21
completed_points = 0

sin2_theta12_grid = np.linspace(0.27, 0.35, N_THETA12)
dm21_grid = np.linspace(7.0e-5, 8.0e-5, N_DM21)

chi2_grid = np.full((N_DM21, N_THETA12), np.nan, dtype=float)
z_norm_grid = np.full((N_DM21, N_THETA12), np.nan, dtype=float)
success_grid = np.zeros((N_DM21, N_THETA12), dtype=bool)

fig, ax = plt.subplots(figsize=(10, 6))

for i_dm, dm21_test in enumerate(dm21_grid):

    for i_theta, sin2_theta12_test in enumerate(sin2_theta12_grid):

        Epr_centers, spectrum = calculateSpectrum(sin2_theta12_test, dm21_test)
        completed_points += 1

        ax.plot(Epr_centers, spectrum, linewidth=1.2, alpha=0.75, label=(
                rf"$\sin^2\theta_{{12}}={sin2_theta12_test:.3f}$, "
                rf"$\Delta m^2_{{21}}={dm21_test * 1e5:.3f}"
                rf"\times10^{{-5}}\ \mathrm{{eV}}^2$"))

        print(f"Completed {completed_points}/{total_points}")

ax.set_xlabel(r"Prompt energy $E_{\mathrm{prompt}}$ [MeV]")
ax.set_ylabel("Events per bin")
ax.set_title("Spectrum at Each Oscillation-Parameter Grid Point")
ax.grid(alpha=0.25)
ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0))
fig.tight_layout()
plt.show()