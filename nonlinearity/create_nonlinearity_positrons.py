import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

E_pts = np.array([0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
F_pts = np.array([0.90, 0.92, 0.94, 0.965, 0.98, 0.995, 1.005, 1.012, 1.018, 1.022, 1.025, 1.027, 1.028])

df = pd.DataFrame({
    "E_pr": E_pts,
    "F_nl": F_pts
})

df.to_csv("positron_nonlinearity.csv", index=False)

plt.figure()
plt.plot(E_pts, F_pts)
plt.title("Non-Linearity Function for Positrons")
plt.xlabel(r"$E_{\rm pr}$ (MeV)")
plt.ylabel(r"$F_{\rm n.l.}(E_{\rm pr})$")
plt.grid(True)
plt.show()