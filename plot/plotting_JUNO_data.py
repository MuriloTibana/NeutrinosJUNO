from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

path = Path(__file__).resolve().parents[1] / "data" / "spect-fit.txt"
df = pd.read_csv(path, sep=r"\s+", header=None)
df.columns = ["energy", "reactor_signal", "reactor_background", "data", "unoscillated_signal"]

df["background"] = df["reactor_background"] - df["reactor_signal"]

# plt.figure()
# plt.plot(df["energy"], df["unoscillated_signal"], label = "no osc.")
# plt.plot(df["energy"], df["reactor_signal"], label = "react.")
# plt.plot(df["energy"], df["reactor_background"], label = "react. + BG")
# plt.xlabel("Energy")
# plt.ylabel("events per 0.1 MeV")
# plt.title("JUNO data")
# plt.legend()

# plt.figure()
# plt.plot(df["energy"], df["background"])
# plt.xlabel("Energy")
# plt.ylabel("events per 0.1 MeV")
# plt.title("Total backgrounds")
# plt.grid()

print(f"Size of reactor_signal: " f"{df['reactor_signal'].shape}")
print(f"Size of reactor_background: " f"{df['reactor_background'].shape}")
print(f"Size of data: " f"{df['data'].shape}")

print(f"Number of energy bins: " f"{len(df['energy'])}")

print(f"Total reactor_signal events: " f"{df['reactor_signal'].sum():.3f}")
print(f"Total reactor_background events: " f"{df['reactor_background'].sum():.3f}")
print(f"Total data events: " f"{df['data'].sum():.3f}")

plt.figure()
plt.plot(df["energy"], df["data"], label = "data")
plt.plot(df["energy"], df["reactor_background"], label = "react. + BG")
plt.plot(df["energy"], df["reactor_signal"], label = "react.")

plt.show()

