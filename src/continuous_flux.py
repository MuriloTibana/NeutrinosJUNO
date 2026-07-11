import numpy as np

def build_continuous_flux_model(
    E_center, E_low, E_high,
    Phi0, Psi_ik,
    delta,
    phi_huber_weighted, sigma_ibd,
    frac, alpha, Delta, m_e,
    N_int=500
):
    E_center = np.asarray(E_center, float)
    E_low    = np.asarray(E_low, float)
    E_high   = np.asarray(E_high, float)
    Phi0     = np.asarray(Phi0, float)
    Psi_ik   = np.asarray(Psi_ik, float)

    nbin = len(E_center)

    M_in = np.zeros((nbin, nbin), dtype=float)

    for i in range(nbin):
        a, b = E_low[i], E_high[i]
        xs = np.linspace(a, b, N_int)
        w  = phi_huber_weighted(xs, frac, alpha) * sigma_ibd(xs, Delta, m_e)

        for n in range(nbin):
            vals = w * delta(n, xs)
            M_in[i, n] = np.trapz(vals, xs) / (b - a)

    y0 = np.linalg.solve(M_in, Phi0)          
    Y  = np.linalg.solve(M_in, Psi_ik)        

    # ---- continuous components ----
    def phi0_cont(E):
        E = np.asarray(E, dtype=float)

        # IMPORTANT: initialize accumulator
        s = np.zeros_like(E, dtype=float)

        for n in range(nbin):
            s += y0[n] * delta(n, E)
            
        return phi_huber_weighted(E, frac, alpha) * s

    def psi_k_cont(E, k):
        E = np.asarray(E, dtype=float)

        # IMPORTANT: initialize accumulator
        s = np.zeros_like(E, dtype=float)

        for n in range(nbin):
            s += Y[n, k] * delta(n, E)

        return phi_huber_weighted(E, frac, alpha) * s

    def phi(E, xi):
        if xi.shape != (nbin,):
            raise ValueError(f"xi must have shape ({nbin},).")

        out = phi0_cont(E).copy()
        for k in range(nbin):
            out += psi_k_cont(E, k) * xi[k]
        return out

    extras = {
        "M_in": M_in,
        "y0": y0,
        "Y": Y,
        "phi0_cont": phi0_cont,
        "psi_k_cont": psi_k_cont,
        "nbin": nbin,
        "N_int": N_int,
    }

    return phi, extras
