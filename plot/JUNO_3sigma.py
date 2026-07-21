import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Published JUNO solar-parameter result
# ============================================================

JUNO_SIN2_THETA12 = 0.3092
JUNO_SIGMA_THETA12 = 0.0087

JUNO_DM21 = 7.50
JUNO_SIGMA_DM21 = 0.12

# Approximate correlation used to reproduce the contour tilt
JUNO_CORRELATION = -0.23

# 3 sigma contour for two parameters
DELTA_CHI2 = 11.83


# ============================================================
# Construct the dashed contour
# ============================================================

mean = np.array([
    JUNO_SIN2_THETA12,
    JUNO_DM21,
])

covariance = np.array([
    [
        JUNO_SIGMA_THETA12**2,
        JUNO_CORRELATION
        * JUNO_SIGMA_THETA12
        * JUNO_SIGMA_DM21,
    ],
    [
        JUNO_CORRELATION
        * JUNO_SIGMA_THETA12
        * JUNO_SIGMA_DM21,
        JUNO_SIGMA_DM21**2,
    ],
])

# Diagonalize the covariance matrix
eigenvalues, eigenvectors = np.linalg.eigh(covariance)

# Points around a unit circle
angle = np.linspace(0.0, 2.0 * np.pi, 1000)

unit_circle = np.vstack([
    np.cos(angle),
    np.sin(angle),
])

# Transform the circle into the JUNO ellipse
juno_contour = (
    mean[:, None]
    + np.sqrt(DELTA_CHI2)
    * eigenvectors
    @ np.diag(np.sqrt(eigenvalues))
    @ unit_circle
)

juno_theta12 = juno_contour[0]
juno_dm21 = juno_contour[1]


# ============================================================
# Plot only the single dashed JUNO line
# ============================================================

fig, ax = plt.subplots(figsize=(6.5, 5.5))

ax.plot(
    juno_theta12,
    juno_dm21,
    color="black",
    linestyle="--",
    linewidth=2.0,
    label=r"JUNO $3\sigma$",
)

ax.scatter(
    JUNO_SIN2_THETA12,
    JUNO_DM21,
    color="black",
    marker="*",
    s=100,
    label="JUNO best fit",
)

ax.set_xlim(0.27, 0.35)
ax.set_ylim(7.0, 8.0)

ax.set_xlabel(r"$\sin^2\theta_{12}$")
ax.set_ylabel(
    r"$\Delta m^2_{21}\,[10^{-5}\,\mathrm{eV}^2]$"
)

ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()

plt.savefig(
    "img/JUNO_dashed_contour.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()