import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from src.reactorsContribution import reactorsContribution
from src.readDayaBay import read_total_flux, read_covariance_matrix, recast_covariance_matrix
from src.deltaSplines import create_delta_basis
from src.continuous_flux import build_continuous_flux_model
from src.phiHuber import phi_huber_weighted
from src.crossSectionIBD import sigma_ibd
from src.energyResponse import compute_spectrum_with_response
from src.oscillation import neutrino_oscillation
from src.background import normalizeToTable, interpolateToBins, draw_physical_pull

# Define Constants

reactor_normalization = True

if reactor_normalization: 
    FIG_PATH = "img/osc_sys_norm_bg.png"
else:
    FIG_PATH = "img/osc_sys_bg.png"


kg_to_MeV = 5.61e29
m_p = 1.6726219e-27 * kg_to_MeV
m_n = 1.6749275e-27 * kg_to_MeV
m_e = 9.1093837e-31 * kg_to_MeV
Delta = m_n - m_p

sin2_theta12 = 0.308
sin2_theta13 = 0.02215

dm21 = 7.49e-5      
dm31 = 2.513e-3     

reactors = [
    {"name": "Taishan-1",  "P_GWth": 4.6, "L_km": 52.77},
    {"name": "Taishan-2",  "P_GWth": 4.6, "L_km": 52.64},
    {"name": "Yangjiang-1","P_GWth": 2.9, "L_km": 52.74},
    {"name": "Yangjiang-2","P_GWth": 2.9, "L_km": 52.82},
    {"name": "Yangjiang-3","P_GWth": 2.9, "L_km": 52.41},
    {"name": "Yangjiang-4","P_GWth": 2.9, "L_km": 52.49},
    {"name": "Yangjiang-5","P_GWth": 2.9, "L_km": 52.11},
    {"name": "Yangjiang-6","P_GWth": 2.9, "L_km": 52.19},
    {"name": "DayaBay-effective", "P_GWth": 17.4, "L_km": 215.0}]
reactor_data = pd.DataFrame(reactors)
reactor_data["w"] = reactor_data["P_GWth"] / (4*np.pi*reactor_data["L_km"]**2)

alpha = {
    "U235":  np.array([4.367, -4.577, 2.100, -5.294e-1, 6.186e-2, -2.777e-3]),
    "Pu239": np.array([4.757, -5.392, 2.563, -6.596e-1, 7.820e-2, -3.536e-3]),
    "Pu241": np.array([2.990, -2.882, 1.278, -3.343e-1, 3.905e-2, -1.754e-3])}

frac = {
    "U235": 0.564,
    "U238": 0.076,
    "Pu239": 0.304,
    "Pu241": 0.056}

# Build continuous flux model from DYB unfolded, and evaluate continuous flux on the neutrino energy grid
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


# Oscillation and Reactors Contribution
E_nu = np.linspace(1.81, 10.0, 2000)
Pee_weighted = np.zeros_like(E_nu)

for _, reactor in reactor_data.iterrows():

    L_km = reactor["L_km"]
    w = reactor["w"]
    Pee_r = neutrino_oscillation(E_nu, L_km, sin2_theta12, sin2_theta13, dm21, dm31)

    Pee_weighted += w * Pee_r

Pee_weighted = Pee_weighted 

# Evaluate flux at central (no pulls)
xi0 = np.zeros(nbin, dtype=float)
phi_E = np.asarray(phi_cont(E_nu, xi0), dtype=float).ravel()
phi_E = np.clip(phi_E, 0.0, None)

# This is where oscillation enters the spectrum.
sigma = sigma_ibd(E_nu, Delta, m_e)
integrand_common = phi_E * sigma * Pee_weighted

# Include detector energy response
NONLINEARITY_PATH = "data/positron_nonlinearity.csv"

bin_width = 0.1
Epr_edges = np.arange(0.0, 10.0 + bin_width, bin_width)
Epr_centers = 0.5 * (Epr_edges[:-1] + Epr_edges[1:])

res_a = 0.033
res_b = 0.01
res_c = 0.0

prompt_alpha = 1.0
prompt_beta = 0.0

Epr_centers, Ni_nl = compute_spectrum_with_response(
    E_nu, integrand_common, Epr_edges, "nonlinear",
    True, res_a, res_b, res_c, prompt_alpha, prompt_beta, NONLINEARITY_PATH)

# Compare with JUNO
JUNO_path = "data/spect-fit.txt"

df_JUNO = pd.read_csv(JUNO_path, sep=r"\s+", header=None)
df_JUNO.columns = ["energy", "reactor_signal", "reactor_background", "data", "unoscillated_signal"]

C_norm = np.max(df_JUNO["reactor_signal"])
osc_spectra_nominal = C_norm * Ni_nl


# Normalization of the reactor neutrino event rate
rng = np.random.default_rng(seed=123)
XI_REACTOR_RATE = rng.normal(loc=0.0, scale=1.0)


REACTOR_RATE_UNCERTAINTIES = {
    "target_protons": 0.010,
    "reference_spectrum": 0.012,
    "thermal_power": 0.005,
    "fission_fraction": 0.006,
    "spent_nuclear_fuel": 0.003,
    "non_equilibrium": 0.002,
    "different_fission_fraction": 0.001,
}

SIGMA_REACTOR_RATE = np.sqrt(np.sum(np.array(list(REACTOR_RATE_UNCERTAINTIES.values()), dtype=float) ** 2))
print("Combined reactor event-rate uncertainty: "f"{100.0 * SIGMA_REACTOR_RATE:.3f}%")

reactor_rate_factor = 1.0 + SIGMA_REACTOR_RATE * XI_REACTOR_RATE
osc_spectra = reactor_rate_factor * osc_spectra_nominal

print(f"Random reactor-rate pull: xi = {XI_REACTOR_RATE:.4f}")
print(f"Reactor-rate factor: {reactor_rate_factor:.6f}")
print("Applied normalization shift: "f"{100.0 * (reactor_rate_factor - 1.0):+.3f}%")

# Background contributions
LIVE_DAYS = 59.1

TABLE1_RATES_CPD = {
    "Li_He": 4.3,
    "geoneutrinos": 2.4,
    "world_reactors": 0.88,
    "bi_po": 0.18,
    "others": 0.04 + 0.02 + 0.05 + 0.08 + 4.9e-2}

BACKGROUND_NORM_SIGMAS = {
    "Li_He": 0.33,
    "geoneutrinos": 0.56,
    "world_reactors": 0.10,
    "bi_po": 0.56,
    "others": 1.00,
}

TABLE1_TOTAL_EVENTS = {
    name: rate * LIVE_DAYS
    for name, rate in TABLE1_RATES_CPD.items()}

BG_path = "data/digitized_backgrounds.csv"
df_raw = pd.read_csv(BG_path)
df_raw.columns = [str(c).strip() for c in df_raw.columns]
for col in df_raw.columns:
    df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")

df_raw = df_raw.dropna(subset=["E_prompt"])
E_raw = df_raw["E_prompt"].to_numpy(dtype=float)

BIN_WIDTH = 0.1

E_edges = np.arange(0.0, 10.0 + BIN_WIDTH, BIN_WIDTH)
E_prompt_bins = 0.5 * (E_edges[:-1] + E_edges[1:])

Li_He_shape = interpolateToBins(E_raw, df_raw["Li_He"].to_numpy(dtype=float), E_prompt_bins)
geoneutrinos_shape = interpolateToBins(E_raw, df_raw["geoneutrinos"].to_numpy(dtype=float), E_prompt_bins)
world_reactors_shape = interpolateToBins(E_raw, df_raw["world_reactors_digitized"].to_numpy(dtype=float), E_prompt_bins)
bi_po_shape = interpolateToBins(E_raw, df_raw["bi_po"].to_numpy(dtype=float), E_prompt_bins)
others_shape = interpolateToBins(E_raw, df_raw["others"].to_numpy(dtype=float), E_prompt_bins)

Li_He = normalizeToTable(Li_He_shape, "Li_He", BIN_WIDTH, TABLE1_TOTAL_EVENTS)
geoneutrinos = normalizeToTable(geoneutrinos_shape, "geoneutrinos", BIN_WIDTH, TABLE1_TOTAL_EVENTS)
world_reactors = normalizeToTable(world_reactors_shape, "world_reactors", BIN_WIDTH, TABLE1_TOTAL_EVENTS)
bi_po = normalizeToTable(bi_po_shape, "bi_po", BIN_WIDTH, TABLE1_TOTAL_EVENTS)
others = normalizeToTable(others_shape, "others", BIN_WIDTH, TABLE1_TOTAL_EVENTS)

background_nominal = {
    "Li_He": Li_He,
    "geoneutrinos": geoneutrinos,
    "world_reactors": world_reactors,
    "bi_po": bi_po,
    "others": others}

XI_BACKGROUND = {
    name: draw_physical_pull(rng, BACKGROUND_NORM_SIGMAS[name])
    for name in background_nominal
}

background_pulled = {}
background_factors = {}

for name, nominal_spectrum in background_nominal.items():

    sigma_bg = BACKGROUND_NORM_SIGMAS[name]
    xi_bg = XI_BACKGROUND[name]

    normalization_factor = 1.0 + sigma_bg * xi_bg

    background_factors[name] = normalization_factor

    background_pulled[name] = normalization_factor * nominal_spectrum

print("\nBackground normalization pulls")
print("=" * 76)

for name in background_nominal:

    xi_bg = XI_BACKGROUND[name]
    sigma_bg = BACKGROUND_NORM_SIGMAS[name]
    factor_bg = background_factors[name]

    nominal_events = np.sum(
        background_nominal[name]
    )

    pulled_events = np.sum(
        background_pulled[name]
    )

    print(
        f"{name:20s} | "
        f"xi = {xi_bg:+8.4f} | "
        f"sigma = {100.0 * sigma_bg:6.2f}% | "
        f"factor = {factor_bg:8.4f} | "
        f"events = {nominal_events:9.3f} "
        f"-> {pulled_events:9.3f}"
    )

print("=" * 76)

Total_Background_nominal = sum(
    background_nominal.values()
)

Total_Background = sum(
    background_pulled.values()
)
osc_spectra_background_nominal = osc_spectra_nominal + Total_Background_nominal

if reactor_normalization:    
    osc_spectra_background = osc_spectra + Total_Background
    label_pulls = "Reactor + background pulls"
    label_nominal = "Reactor + nominal backgrounds"
    label_title = "Model with Reactor and Background Normalization Pulls"
else:
    osc_spectra_background = osc_spectra_nominal + Total_Background
    label_pulls = "Nominal Reactor + background pulls"
    label_nominal = "Nominal Reactor + nominal backgrounds"
    label_title = "Model with Nominal Reactor and Background Normalization Pulls"

plt.figure(figsize=(7.5, 4.8))
plt.plot(E_prompt_bins, osc_spectra_background, ":", color="darkgoldenrod", lw=3, label=label_pulls)
plt.plot(E_prompt_bins, osc_spectra_background_nominal, "-", color="darkorange", lw=2, label=label_nominal)
plt.xlabel(r"$E_{\rm pr}$ [MeV]")
plt.ylabel("Events per 0.1 MeV")
plt.title(label_title)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(FIG_PATH, dpi=300, bbox_inches="tight")

print(f"Saved figure to: {FIG_PATH}")
plt.show()