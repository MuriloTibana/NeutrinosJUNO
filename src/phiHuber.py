import numpy as np

def phi_huber(E_nu, a):
    E_nu = np.asarray(E_nu, dtype=float)
    poly = 0.0
    for k in range(6):
        poly += a[k] * E_nu**k
    return np.exp(poly)  # neutrinos / fission / MeV

def phi_huber_weighted(E_nu, f, alpha):
    # U238 omitted (Huber Table III doesn't provide it)
    return (f["U235"] * phi_huber(E_nu, alpha["U235"]) +
            f["Pu239"] * phi_huber(E_nu, alpha["Pu239"]) +
            f["Pu241"] * phi_huber(E_nu, alpha["Pu241"]))