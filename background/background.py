from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

BG_path = ROOT / "data" / "fig3_backgrounds_digitized_raw.csv"

RAW_PLOT_PATH = ROOT / "img" / "backgrounds_raw.png"
NORM_PLOT_PATH = ROOT / "img" / "backgrounds_normalized.png"

# ============================================================
# Table 1 normalization
# ============================================================

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


# ============================================================
# Model bins
# ============================================================

BIN_WIDTH = 0.1

E_edges = np.arange(0.0, 10.0 + BIN_WIDTH, BIN_WIDTH)
E_prompt_bins = 0.5 * (E_edges[:-1] + E_edges[1:])


# ============================================================
# Load raw digitized backgrounds
# ============================================================

df_raw = pd.read_csv(BG_path)
df_raw.columns = [str(c).strip() for c in df_raw.columns]

for col in df_raw.columns:
    df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")

df_raw = df_raw.dropna(subset=["E_prompt"])

E_raw = df_raw["E_prompt"].to_numpy(dtype=float)


# ============================================================
# Save raw digitized plot
# ============================================================

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


# ============================================================
# Helper functions
# ============================================================

def interpolate_to_bins(E_raw, y_raw, E_bins):
    """
    Interpolate raw digitized curve onto model bin centers.
    """

    good = np.isfinite(E_raw) & np.isfinite(y_raw)

    E = E_raw[good]
    y = y_raw[good]

    order = np.argsort(E)

    E = E[order]
    y = y[order]

    y_interp = np.interp(
        E_bins,
        E,
        y,
        left=0.0,
        right=0.0,
    )

    return np.clip(y_interp, 0.0, None)


def normalize_to_table1(shape, component_name):
    """
    Normalize the interpolated shape so that its total number of events is:

        Table 1 rate x live days.

    The Fig. 3 y-axis is events per 0.1 MeV.
    Since BIN_WIDTH = 0.1 MeV, each bin value is already events per bin.
    """

    shape = np.asarray(shape, dtype=float)
    shape = np.clip(shape, 0.0, None)

    counts_per_bin = shape * (BIN_WIDTH / 0.1)

    current_total = np.sum(counts_per_bin)

    if current_total <= 0:
        raise ValueError(f"{component_name} has zero total before normalization.")

    target_total = TABLE1_TOTAL_EVENTS[component_name]

    return counts_per_bin * target_total / current_total


# ============================================================
# Interpolate raw curves onto model bins
# ============================================================

Li_He_shape = interpolate_to_bins(E_raw, df_raw["Li_He"].to_numpy(dtype=float), E_prompt_bins)

geoneutrinos_shape = interpolate_to_bins(E_raw, df_raw["geoneutrinos"].to_numpy(dtype=float), E_prompt_bins)

world_reactors_shape = interpolate_to_bins(
    E_raw,
    df_raw["world_reactors_digitized"].to_numpy(dtype=float),
    E_prompt_bins,
)

bi_po_shape = interpolate_to_bins(
    E_raw,
    df_raw["bi_po"].to_numpy(dtype=float),
    E_prompt_bins,
)

others_shape = interpolate_to_bins(
    E_raw,
    df_raw["others"].to_numpy(dtype=float),
    E_prompt_bins,
)


# ============================================================
# Normalize interpolated curves using Table 1
# ============================================================

Li_He = normalize_to_table1(Li_He_shape, "Li_He")
geoneutrinos = normalize_to_table1(geoneutrinos_shape, "geoneutrinos")
world_reactors = normalize_to_table1(world_reactors_shape, "world_reactors")
bi_po = normalize_to_table1(bi_po_shape, "bi_po")
others = normalize_to_table1(others_shape, "others")

Total_background = (
    Li_He
    + geoneutrinos
    + world_reactors
    + bi_po
    + others
)


# ============================================================
# Print normalization check
# ============================================================

print("\nTable 1 target totals:")
for name, total in TABLE1_TOTAL_EVENTS.items():
    print(f"  {name:16s}: {total:.6f}")

print("\nNormalized totals after interpolation:")
print(f"  {'Li_He':16s}: {np.sum(Li_He):.6f}")
print(f"  {'geoneutrinos':16s}: {np.sum(geoneutrinos):.6f}")
print(f"  {'world_reactors':16s}: {np.sum(world_reactors):.6f}")
print(f"  {'bi_po':16s}: {np.sum(bi_po):.6f}")
print(f"  {'others':16s}: {np.sum(others):.6f}")
print(f"  {'Total_background':16s}: {np.sum(Total_background):.6f}")


# ============================================================
# Save final normalized plot
# ============================================================

fig_final, ax_final = plt.subplots(figsize=(9, 5.5))

ax_final.step(
    E_prompt_bins,
    Li_He,
    where="mid",
    lw=1.8,
    label=r"$^9$Li/$^8$He",
)

ax_final.step(
    E_prompt_bins,
    geoneutrinos,
    where="mid",
    lw=1.8,
    label="Geoneutrinos",
)

ax_final.step(
    E_prompt_bins,
    world_reactors,
    where="mid",
    lw=1.8,
    label="World reactors",
)

ax_final.step(
    E_prompt_bins,
    bi_po,
    where="mid",
    lw=1.8,
    label=r"$^{214}$Bi-$^{214}$Po",
)

ax_final.step(
    E_prompt_bins,
    others,
    where="mid",
    lw=1.8,
    label="Others",
)

ax_final.step(
    E_prompt_bins,
    Total_background,
    where="mid",
    lw=2.5,
    color="black",
    label="Total background",
)

ax_final.set_xlabel(r"$E_{\rm prompt}$ [MeV]")
ax_final.set_ylabel("Events per 0.1 MeV")
ax_final.set_title("Fig. 3 backgrounds interpolated and normalized to Table 1")
ax_final.set_xlim(0.6, 10.0)
ax_final.set_ylim(bottom=0.0)
ax_final.grid(alpha=0.3)
ax_final.legend()
fig_final.tight_layout()

fig_final.savefig(NORM_PLOT_PATH, dpi=300)
print(f"\nSaved final normalized plot to:\n  {NORM_PLOT_PATH}")


# ============================================================
# Show plots
# ============================================================

plt.show()