import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from src.fit import calculateSpectrum, chi2_CNP, chi2_poisson, calculateRawSpectrum
from src.background import normalizeToTable, interpolateToBins

# Parameters for fit procedure
N_THETA12 = 10
N_DM21 = 10
EPSILON = 1e-12

FIT_JUNO = "data"
CHI2_MODEL = "poisson"

# JUNO parameters
df_JUNO = pd.read_csv("data/spect-fit.txt", sep=r"\s+", header=None)
df_JUNO.columns = ["energy", "reactor_signal", "reactor_background", "data", "unoscillated_signal"]

JUNO_energy = df_JUNO["energy"].to_numpy(dtype=float)
JUNO_data = df_JUNO["data"].to_numpy(dtype=float)
JUNO_total = df_JUNO["reactor_background"].to_numpy(dtype=float)
JUNO_reactor = df_JUNO["reactor_signal"].to_numpy(dtype=float)
JUNO_background = (df_JUNO["reactor_background"] - df_JUNO["reactor_signal"]).to_numpy(dtype=float)

# Background
LIVE_DAYS = 59.1
BIN_WIDTH = 0.1

TABLE1_RATES_CPD = {
    "Li_He": 4.3,
    "geoneutrinos": 2.4,
    "world_reactors": 0.88,
    "bi_po": 0.18,
    "others": 0.04 + 0.02 + 0.05 + 0.08 + 4.9e-2}

TABLE1_TOTAL_EVENTS = {name: rate * LIVE_DAYS
                       for name, rate in TABLE1_RATES_CPD.items()}

BG_path = "data/digitized_backgrounds.csv"
df_raw = pd.read_csv(BG_path)

df_raw.columns = [str(c).strip() for c in df_raw.columns]

for col in df_raw.columns:
    df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")

df_raw = df_raw.dropna(subset=["E_prompt"])
E_raw = df_raw["E_prompt"].to_numpy(dtype=float)

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

Total_background = Li_He + geoneutrinos + world_reactors + bi_po + others

# Path for results
RESULTS_PATH = f"results/{N_THETA12}_{N_DM21}_{FIT_JUNO}_{CHI2_MODEL}.npz"

JUNO_SPECTRA = {
    "data": JUNO_data,
    "total": JUNO_total,
    "reactor": JUNO_reactor}
JUNO_fit = JUNO_SPECTRA[FIT_JUNO]

CHI2_MODELS = {
    "cnp": chi2_CNP,
    "poisson": chi2_poisson}
chi2_function = CHI2_MODELS[CHI2_MODEL]

# Uncertainties 
REACTOR_RATE_UNCERTAINTIES = {
    "target_protons": 0.010,
    "reference_spectrum": 0.012,
    "thermal_power": 0.005,
    "fission_fraction": 0.006,
    "spent_nuclear_fuel": 0.003,
    "non_equilibrium": 0.002,
    "different_fission_fraction": 0.001}

SIGMA_REACTOR_RATE = np.sqrt(np.sum(np.array(list(REACTOR_RATE_UNCERTAINTIES.values()), dtype=float) ** 2))
print("Combined reactor event-rate uncertainty: "f"{100.0 * SIGMA_REACTOR_RATE:.3f}%")

# Normalization constant
SIN2_THETA12_REFERENCE = 0.309
DM21_REFERENCE = 7.53e-5

Epr_reference, raw_reference = calculateRawSpectrum(SIN2_THETA12_REFERENCE, DM21_REFERENCE)
JUNO_reactor_on_reference_grid = np.interp(Epr_reference, JUNO_energy, JUNO_reactor, left=0.0, right=0.0)
REFERENCE_NORMALIZATION = JUNO_reactor_on_reference_grid.sum() / raw_reference.sum()

# Fitting
total_points = N_THETA12 * N_DM21
completed_points = 0

sin2_theta12_grid = np.linspace(0.27, 0.35, N_THETA12)
dm21_grid = np.linspace(7.0e-5, 8.0e-5, N_DM21)
chi2_grid = np.full((N_DM21, N_THETA12), np.nan, dtype=float)
z_norm_grid = np.full((N_DM21, N_THETA12), np.nan, dtype=float)
success_grid = np.zeros((N_DM21, N_THETA12), dtype=bool)

for i_dm, dm21_test in enumerate(dm21_grid):

    for i_theta, sin2_theta12_test in enumerate(sin2_theta12_grid):

        Epr_centers, Ni_osc = calculateRawSpectrum(sin2_theta12_test, dm21_test)
        spectrum = Ni_osc * REFERENCE_NORMALIZATION

        def chi2_for_this_point(z_norm):

            reactor_factor = 1.0 + SIGMA_REACTOR_RATE * z_norm

            prediction_model = reactor_factor * (spectrum + Total_background) 
            prediction_fit = np.interp(JUNO_energy, Epr_centers, prediction_model, left=0.0, right=0.0)
            prediction_fit = np.clip(prediction_fit, EPSILON, None) 

            chi2_data = chi2_function(JUNO_fit, prediction_fit)
            chi2_pull = z_norm**2

            return chi2_data + chi2_pull
        
        fit_result = minimize_scalar(chi2_for_this_point, bounds=(-5.0, 5.0), method="bounded") 
        chi2_grid[i_dm, i_theta] = fit_result.fun
        z_norm_grid[i_dm, i_theta] = fit_result.x
        success_grid[i_dm, i_theta] = fit_result.success
        
        completed_points += 1
        print(f"Completed {completed_points}/{total_points} | "
              f"chi2={fit_result.fun:.6f} | "
              f"z_norm={fit_result.x:+.6f} | "
              f"success={fit_result.success}")

np.savez_compressed(
    RESULTS_PATH,

    sin2_theta12_grid=sin2_theta12_grid,
    dm21_grid=dm21_grid,
    chi2_grid=chi2_grid,
    z_norm_grid=z_norm_grid,
    success_grid=success_grid,
    SIGMA_REACTOR_RATE=SIGMA_REACTOR_RATE,
    Total_background=Total_background)

print(f"Results saved to {RESULTS_PATH}")