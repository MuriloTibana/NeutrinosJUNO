import numpy as np
from scipy.interpolate import CubicSpline

def create_delta_basis(E_center, bc_type="natural"):

    E_center = np.asarray(E_center, float)
    nbin = len(E_center)

    I = np.eye(nbin)

    splines = [
        CubicSpline(E_center, I[n],
                    bc_type=bc_type,
                    extrapolate=True)
        for n in range(nbin)
    ]

    def delta(n, E):
        E = np.asarray(E, float)
        return splines[n](E)

    return delta, splines, I
