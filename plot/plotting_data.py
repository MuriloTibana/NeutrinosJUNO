from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

path = Path(__file__).resolve().parents[1] / "data" / "spect-fit.txt"
df = pd.read_csv(path, sep=r"\s+", header=None)
df.columns = ["energy", "reactor_signal", "reactor_background", "data", "unoscillated_signal"]

df["background"] = df["reactor_background"] - df["reactor_signal"]

plt.figure()
plt.plot(df["energy"], df["unoscillated_signal"], label = "no osc.")
plt.plot(df["energy"], df["reactor_signal"], label = "react.")
plt.plot(df["energy"], df["reactor_background"], label = "react. + BG")
plt.xlabel("Energy")
plt.ylabel("events per 0.1 MeV")
plt.title("JUNO data")
plt.legend()

plt.figure()
plt.plot(df["energy"], df["background"])
plt.xlabel("Energy")
plt.ylabel("events per 0.1 MeV")
plt.title("Total backgrounds")
plt.grid()
plt.show()