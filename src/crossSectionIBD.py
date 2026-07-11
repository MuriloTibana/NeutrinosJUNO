import numpy as np

# def sigma_ibd(E_nu, Delta, m_e):
#     E_nu = np.asarray(E_nu, dtype=float)
#     E_e = E_nu - Delta
#     sigma = np.zeros_like(E_nu)

#     allowed = E_e > m_e
#     sigma[allowed] = 0.0952e-42 * E_e[allowed] * np.sqrt(E_e[allowed]**2 - m_e**2)
#     return sigma


def sigma_ibd(
    E_nu,
    Delta,
    m_e,
    *,
    f=1.0,
    g=1.26,
    f2=3.706,         
    m_p=938.2720813,
    m_n=939.5654133,
    n_cos=400
):
    E_nu = np.asarray(E_nu, dtype=float)

    M = 0.5 * (m_p + m_n)

    Ee0 = E_nu - Delta
    pe0_sq = Ee0**2 - m_e**2
    allowed = pe0_sq > 0.0

    sigma = np.zeros_like(E_nu)

    if not np.any(allowed):
        return sigma

    Ee0a = Ee0[allowed]
    pe0  = np.sqrt(pe0_sq[allowed])
    v0   = pe0 / Ee0a

    y2 = 0.5 * (Delta**2 - m_e**2)

    sigma0 = 0.0952e-42 / (f**2 + 3.0*g**2) 

    c = np.linspace(-1.0, 1.0, n_cos) 
    ca = c[None, :]

    Ee1 = Ee0a[:, None] * (1.0 - (E_nu[allowed][:, None] / M) * (1.0 - v0[:, None] * ca)) - (y2 / M)
    pe1_sq = Ee1**2 - m_e**2
    pe1_sq = np.maximum(pe1_sq, 0.0)
    pe1 = np.sqrt(pe1_sq)
    v1 = np.zeros_like(pe1)
    good1 = Ee1 > 0
    v1[good1] = pe1[good1] / Ee1[good1]

    Ee0c = Ee0a[:, None]
    v0c  = v0[:, None]

    Gamma = (
        2.0*(f + f2)*g * ((2.0*Ee0c + Delta)*(1.0 - v0c*ca) - (m_e**2 / Ee0c))
        + (f**2 + g**2) * (Delta*(1.0 + v0c*ca) + (m_e**2 / Ee0c))
        + (f**2 + 3.0*g**2) * ((Ee0c + Delta)*(1.0 - (1.0/v0c)*ca) - Delta)
        + (f**2 - g**2) * (((Ee0c + Delta)*(1.0 - (1.0/v0c)*ca) - Delta) * v0c * ca)
    )

    term1 = ((f**2 + 3.0*g**2) + (f**2 - g**2)*v1*ca) * Ee1 * pe1
    term2 = (Gamma / M) * (Ee0a * pe0)[:, None] 
    dsdct = (sigma0/2.0) * (term1 - term2)

    sig_allowed = np.trapz(dsdct, c, axis=1)

    sig_allowed = np.maximum(sig_allowed, 0.0)

    sigma[allowed] = sig_allowed
    return sigma