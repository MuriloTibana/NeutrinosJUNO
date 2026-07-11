import pandas as pd
from io import StringIO
import numpy as np

def read_total_flux(path, header_key):
    rows = []
    capture = False

    with open(path, "r") as f:
        for line in f:
            s = line.strip()

            if s.startswith(f"# {header_key}"):
                capture = True
                continue
            if not capture:
                continue
            if s == "" or s.startswith("#"):
                continue

            parts = s.split()
            if len(parts) != 4:
                break

            rows.append(line)

    df = pd.read_csv(StringIO("".join(rows)), sep=r"\s+", header=None)
    df.columns = ["E_low", "E_high", "Flux", "Error"]

    df["E_center"] = 0.5 * (df["E_high"] + df["E_low"])

    return df

def read_covariance_matrix(path):
    cov_floats = []
    in_cov = False
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if s.startswith("# The covariance matrix"):
                in_cov = True
                continue
            if not in_cov:
                continue
            if s == "" or s.startswith("#"):
                continue
            if "smearing" in s.lower():
                break
            cov_floats.extend([float(x) for x in s.split()])
            if len(cov_floats) >= 75*75:
                cov_floats = cov_floats[:75*75]
                break

    C75 = np.array(cov_floats).reshape(75, 75)
    C75 = 0.5 * (C75 + C75.T)
    C = C75[50:75, 50:75]
    return C

def recast_covariance_matrix(C_ij):

    D2, O = np.linalg.eigh(C_ij)
    D2 = np.clip(D2, 0.0, None)
    D = np.sqrt(D2)
    Psi_ik = O * D  
    return Psi_ik
