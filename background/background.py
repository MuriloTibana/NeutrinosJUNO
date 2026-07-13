import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.background import normalizeToTable, interpolateToBins

# Define paths
BG_path = ROOT / "data" / "digitized_backgrounds.csv"

RAW_PLOT_PATH = ROOT / "img" / "backgrounds_raw.png"
NORM_PLOT_PATH = ROOT / "img" / "backgrounds_normalized.png"

# Normalization constants
LIVE_DAYS = 59.1

TABLE1_RATES_CPD = {
    "Li_He": 4.3,
    "geoneutrinos": 2.4,
    "world_reactors": 0.88,
    "bi_po": 0.18,
    "others": 0.04 + 0.02 + 0.05 + 0.08 + 4.9e-2}

TABLE1_TOTAL_EVENTS = {
    name: rate * LIVE_DAYS
    for name, rate in TABLE1_RATES_CPD.items()}

# Model bins for interpolation and normalization
BIN_WIDTH = 0.1

E_edges = np.arange(0.0, 10.0 + BIN_WIDTH, BIN_WIDTH)
E_prompt_bins = 0.5 * (E_edges[:-1] + E_edges[1:])

# Loading raw digitized data
df_raw = pd.read_csv(BG_path)
df_raw.columns = [str(c).strip() for c in df_raw.columns]

for col in df_raw.columns:
    df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")

df_raw = df_raw.dropna(subset=["E_prompt"])

E_raw = df_raw["E_prompt"].to_numpy(dtype=float)

# Plot raw digitized curves
fig_raw, ax_raw = plt.subplots(figsize=(9, 5.5))

ax_raw.plot(df_raw["E_prompt"], df_raw["Li_He"], lw=1.8, label=r"$^9$Li/$^8$He")
ax_raw.plot(df_raw["E_prompt"], df_raw["geoneutrinos"], lw=1.8, label="Geoneutrinos")
ax_raw.plot(df_raw["E_prompt"], df_raw["world_reactors_digitized"], lw=1.8, label="World reactors")
ax_raw.plot(df_raw["E_prompt"], df_raw["bi_po"], lw=1.8, label=r"$^{214}$Bi-$^{214}$Po")
ax_raw.plot(df_raw["E_prompt"], df_raw["others"], lw=1.8, label="Others")
ax_raw.set_xlabel(r"$E_{\rm prompt}$ [MeV]")
ax_raw.set_ylabel("Raw digitized curve value")
ax_raw.set_title("Raw digitized Fig. 3 background curves")
ax_raw.set_xlim(0.6, 10.0)
ax_raw.set_ylim(bottom=0.0)
ax_raw.grid(alpha=0.3)
ax_raw.legend()
fig_raw.tight_layout()

fig_raw.savefig(RAW_PLOT_PATH, dpi=300)
print(f"Saved raw plot to:\n  {RAW_PLOT_PATH}")

# Interpolate raw curves onto model bins
Li_He_shape = interpolateToBins(E_raw, df_raw["Li_He"].to_numpy(dtype=float), E_prompt_bins)
geoneutrinos_shape = interpolateToBins(E_raw, df_raw["geoneutrinos"].to_numpy(dtype=float), E_prompt_bins)
world_reactors_shape = interpolateToBins(E_raw, df_raw["world_reactors_digitized"].to_numpy(dtype=float), E_prompt_bins)
bi_po_shape = interpolateToBins(E_raw, df_raw["bi_po"].to_numpy(dtype=float), E_prompt_bins)
others_shape = interpolateToBins(E_raw, df_raw["others"].to_numpy(dtype=float), E_prompt_bins)

# Normalize interpolated curves using Table 1
Li_He = normalizeToTable(Li_He_shape, "Li_He", BIN_WIDTH, TABLE1_TOTAL_EVENTS)
geoneutrinos = normalizeToTable(geoneutrinos_shape, "geoneutrinos", BIN_WIDTH, TABLE1_TOTAL_EVENTS)
world_reactors = normalizeToTable(world_reactors_shape, "world_reactors", BIN_WIDTH, TABLE1_TOTAL_EVENTS)
bi_po = normalizeToTable(bi_po_shape, "bi_po", BIN_WIDTH, TABLE1_TOTAL_EVENTS)
others = normalizeToTable(others_shape, "others", BIN_WIDTH, TABLE1_TOTAL_EVENTS)

Total_background = (Li_He + geoneutrinos + world_reactors + bi_po + others)

# Plot normalized curves
fig_norm, ax_norm = plt.subplots(figsize=(9, 5.5))

ax_norm.step(E_prompt_bins, Li_He, where="mid", lw=1.8, label=r"$^9$Li/$^8$He")
ax_norm.step(E_prompt_bins, geoneutrinos, where="mid", lw=1.8, label="Geoneutrinos")
ax_norm.step(E_prompt_bins, world_reactors, where="mid", lw=1.8, label="World reactors")
ax_norm.step(E_prompt_bins, bi_po, where="mid", lw=1.8, label=r"$^{214}$Bi-$^{214}$Po")
ax_norm.step(E_prompt_bins, others, where="mid", lw=1.8, label="Others")
ax_norm.step(E_prompt_bins, Total_background, where="mid", lw=2.5, color="black", label="Total background")
ax_norm.set_xlabel(r"$E_{\rm prompt}$ [MeV]")
ax_norm.set_ylabel("Events per 0.1 MeV")
ax_norm.set_title("Fig. 3 backgrounds interpolated and normalized to Table 1")
ax_norm.set_xlim(0.6, 10.0)
ax_norm.set_ylim(bottom=0.0)
ax_norm.grid(alpha=0.3)
ax_norm.legend()
fig_norm.tight_layout()
fig_norm.savefig(NORM_PLOT_PATH, dpi=300)
print(f"\nSaved final normalized plot to:\n  {NORM_PLOT_PATH}")


# Compare with JUNO background data
JUNO_path = ROOT / "data" / "spect-fit.txt"
JUNO_PLOT_PATH = ROOT / "img" / "backgrounds_JUNO-Model.png"

df = pd.read_csv(JUNO_path, sep=r"\s+", header=None)
df.columns = ["energy", "reactor_signal", "reactor_background", "data", "unoscillated_signal"]
df["background"] = df["reactor_background"] - df["reactor_signal"]

fig_JUNO, ax_JUNO = plt.subplots(figsize=(9, 5.5))
ax_JUNO.step(df["energy"], df["background"], where="mid", lw=1.8, label="JUNO background")
ax_JUNO.step(E_prompt_bins, Total_background, where="mid", lw=1.8, label="Reconstructed background")
ax_JUNO.set_xlabel(r"$E_{\rm prompt}$ [MeV]")
ax_JUNO.set_ylabel("Events per 0.1 MeV")
ax_JUNO.set_title("Comparison with JUNO background data")
ax_JUNO.set_xlim(0.6, 10.0)
ax_JUNO.set_ylim(bottom=0.0)
ax_JUNO.grid(alpha=0.3)
ax_JUNO.legend()
fig_JUNO.tight_layout()
fig_JUNO.savefig(JUNO_PLOT_PATH, dpi=300)
print(f"\nSaved final JUNO comparison plot to:\n  {JUNO_PLOT_PATH}")

plt.show()