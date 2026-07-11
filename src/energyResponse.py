import numpy as np
import pandas as pd
from functools import lru_cache
from scipy.special import erf

m_e_default = 0.51099895
m_p_default = 938.2720813
m_n_default = 939.5654133
Delta_default = m_n_default - m_p_default

DEFAULT_NONLINEARITY_PATH = "data/positron_nonlinearity.csv"

def Ee_endpoints(E_nu, m_p=m_p_default, m_n=m_n_default, m_e=m_e_default):
    """
    Positron total-energy endpoints for inverse beta decay:

        anti-nu_e + p -> e^+ + n

    For a fixed neutrino energy E_nu, the positron energy is not exactly
    a single value once recoil/angular effects are included. Instead it
    lies between Ee_inf and Ee_sup.
    """

    E_nu = np.asarray(E_nu, dtype=float)

    s = m_p**2 + 2.0 * m_p * E_nu
    sqrt_s = np.sqrt(s)

    delta = (m_n**2 - m_p**2 - m_e**2) / (2.0 * m_p)

    Ecm_nu = (s - m_p**2) / (2.0 * sqrt_s)
    Ecm_e = (s - m_n**2 + m_e**2) / (2.0 * sqrt_s)

    lam = (s - (m_n - m_e)**2) * (s - (m_n + m_e)**2)
    lam = np.maximum(lam, 0.0)

    pcm_e = np.sqrt(lam) / (2.0 * sqrt_s)

    Ee_inf = E_nu - delta - (Ecm_nu / m_p) * (Ecm_e + pcm_e)
    Ee_sup = E_nu - delta - (Ecm_nu / m_p) * (Ecm_e - pcm_e)

    return Ee_inf, Ee_sup


def Epr_endpoints(E_nu, m_p=m_p_default, m_n=m_n_default, m_e=m_e_default):
    """
    Prompt visible-energy endpoints.

    The prompt visible energy is approximately

        Epr = Ee + m_e

    because the positron contributes its energy and then annihilates.
    """

    Ee_inf, Ee_sup = Ee_endpoints(E_nu, m_p=m_p, m_n=m_n, m_e=m_e)

    Epr_inf = Ee_inf + m_e
    Epr_sup = Ee_sup + m_e

    return Epr_inf, Epr_sup

def sigma_Epr(Epr, a=0.03, b=0.0, c=0.0):

    Epr = np.asarray(Epr, dtype=float)
    E = np.maximum(Epr, 1e-12)

    frac2 = (a / np.sqrt(E))**2 + b**2 + (c / E)**2

    return E * np.sqrt(frac2)

def g_func(x, sigma_e, m_e=m_e_default):
    """
    Auxiliary function used in the analytic Gaussian-smearing integral.

        g(u) = u erf(u) + exp(-u^2)/sqrt(pi)

    where

        u = (x - m_e)/(sqrt(2) sigma_e)
    """

    x = np.asarray(x, dtype=float)
    sigma_e = np.asarray(sigma_e, dtype=float)

    u = (x - m_e) / (np.sqrt(2.0) * sigma_e)

    return u * erf(u) + (1.0 / np.sqrt(np.pi)) * np.exp(-u * u)

def R_i(
    E_nu,
    Ei_lo,
    Ei_hi,
    a=0.03,
    b=0.0,
    c=0.0,
    sigma_prescription="midpoint",
):
    """
    Baseline detector response function.

    For a given true neutrino energy E_nu, this computes the probability
    that the reconstructed prompt energy lands inside the bin

        Ei_lo <= Epr <= Ei_hi.

    This version uses the positron-energy endpoint treatment.
    """

    E_nu = np.asarray(E_nu, dtype=float)

    E1, E2 = Epr_endpoints(E_nu)

    width = E2 - E1
    ok = width > 0.0

    if sigma_prescription == "midpoint":
        Eref = 0.5 * (E1 + E2)
    elif sigma_prescription == "bincenter":
        Eref = np.full_like(E_nu, 0.5 * (Ei_lo + Ei_hi))
    else:
        raise ValueError("sigma_prescription must be 'midpoint' or 'bincenter'")

    sigma_e = np.maximum(sigma_Epr(Eref, a=a, b=b, c=c), 1e-12)

    pref = np.zeros_like(E_nu, dtype=float)
    pref[ok] = (np.sqrt(2.0) * sigma_e[ok]) / (2.0 * width[ok])

    term = (
        g_func(Ei_hi - E1, sigma_e)
        - g_func(Ei_hi - E2, sigma_e)
        - g_func(Ei_lo - E1, sigma_e)
        + g_func(Ei_lo - E2, sigma_e)
    )

    W = np.zeros_like(E_nu, dtype=float)
    W[ok] = pref[ok] * term[ok]

    return W

@lru_cache(maxsize=8)
def load_nonlinearity_points(path=DEFAULT_NONLINEARITY_PATH):

    df = pd.read_csv(path)

    required_cols = {"E_pr", "F_nl"}
    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(
            f"Nonlinearity file {path!r} is missing columns: {missing}"
        )

    E_pts = df["E_pr"].to_numpy(dtype=float)
    F_pts = df["F_nl"].to_numpy(dtype=float)

    order = np.argsort(E_pts)
    E_pts = E_pts[order]
    F_pts = F_pts[order]

    unique_E, unique_idx = np.unique(E_pts, return_index=True)
    E_pts = unique_E
    F_pts = F_pts[unique_idx]

    return E_pts, F_pts


def F_nl(Epr, nonlinearity_path=DEFAULT_NONLINEARITY_PATH):
    """
    Positron nonlinearity function.

    It returns

        F_nl(Epr)

    by linearly interpolating the tabulated CSV values.
    """

    E_pts, F_pts = load_nonlinearity_points(nonlinearity_path)

    Epr = np.asarray(Epr, dtype=float)

    Ecl = np.clip(Epr, E_pts[0], E_pts[-1])

    return np.interp(Ecl, E_pts, F_pts)


def gaussian_bin_prob(mu, sig, lo, hi):
    """
    Probability that a Gaussian random variable lands inside one bin.

    If

        X ~ Normal(mu, sig^2),

    this returns

        P(lo <= X <= hi).
    """

    mu = np.asarray(mu, dtype=float)
    sig = np.asarray(sig, dtype=float)

    sig = np.clip(sig, 1e-12, None)

    z_hi = (hi - mu) / (np.sqrt(2.0) * sig)
    z_lo = (lo - mu) / (np.sqrt(2.0) * sig)

    return 0.5 * (erf(z_hi) - erf(z_lo))


def R_i_nl(
    E_nu,
    Ei_lo,
    Ei_hi,
    Delta=Delta_default,
    m_e=m_e_default,
    a=0.033,
    b=0.01,
    c=0.0,
    alpha=1.0,
    beta=0.0,
    nonlinearity_path=DEFAULT_NONLINEARITY_PATH,
):
    """
    Nonlinear detector response function.

    Chain:

        Evis(E_nu) = E_nu - Delta + m_e

        Epr0 = alpha * Evis + beta

        mu = Epr0 * F_nl(Epr0)

        sigma = sigma_Epr(mu)

        R_i_nl(E_nu) = P(Ei_lo <= Erec <= Ei_hi)

    where

        Erec ~ Normal(mu, sigma^2).
    """

    E_nu = np.asarray(E_nu, dtype=float)

    Evis = E_nu - Delta + m_e

    Epr0 = alpha * Evis + beta

    mu = Epr0 * F_nl(Epr0, nonlinearity_path=nonlinearity_path)

    sig = sigma_Epr(mu, a=a, b=b, c=c)

    mask = Epr0 > 0.0

    Ri = np.zeros_like(E_nu, dtype=float)

    Ri[mask] = gaussian_bin_prob(
        mu=mu[mask],
        sig=sig[mask],
        lo=Ei_lo,
        hi=Ei_hi,
    )

    return Ri

# ============================================================
# Spectrum calculator
# ============================================================

def compute_spectrum_with_response(
    E_nu,
    integrand_common,
    Epr_edges,
    response_type="baseline",
    normalize=True,
    # Resolution parameters
    a=0.033,
    b=0.01,
    c=0.0,
    # Prompt-energy calibration parameters
    prompt_alpha=1.0,
    prompt_beta=0.0,
    # Nonlinearity file
    nonlinearity_path=DEFAULT_NONLINEARITY_PATH,
    # Baseline response options
    sigma_prescription="midpoint",
):
    E_nu = np.asarray(E_nu, dtype=float)
    integrand_common = np.asarray(integrand_common, dtype=float)
    Epr_edges = np.asarray(Epr_edges, dtype=float)

    if E_nu.shape != integrand_common.shape:
        raise ValueError("E_nu and integrand_common must have the same shape.")

    Epr_centers = 0.5 * (Epr_edges[:-1] + Epr_edges[1:])
    Ni = np.zeros_like(Epr_centers, dtype=float)

    for i in range(len(Epr_centers)):
        Ei_lo = Epr_edges[i]
        Ei_hi = Epr_edges[i + 1]

        if response_type == "baseline":
            Ri = R_i(
                E_nu,
                Ei_lo,
                Ei_hi,
                a=a,
                b=b,
                c=c,
                sigma_prescription=sigma_prescription,
            )

        elif response_type == "nonlinear":
            Ri = R_i_nl(
                E_nu,
                Ei_lo,
                Ei_hi,
                a=a,
                b=b,
                c=c,
                alpha=prompt_alpha,
                beta=prompt_beta,
                nonlinearity_path=nonlinearity_path,
            )

        else:
            raise ValueError(
                "response_type must be either 'baseline' or 'nonlinear'."
            )

        Ni[i] = np.trapezoid(integrand_common * Ri, E_nu)

    if normalize:
        max_Ni = np.max(Ni)
        if max_Ni > 0.0:
            Ni = Ni / max_Ni

    return Epr_centers, Ni