import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from src.readDayaBay import (
    read_total_flux,
    read_covariance_matrix,
    recast_covariance_matrix,
)
from src.deltaSplines import create_delta_basis
from src.continuous_flux import build_continuous_flux_model
from src.phiHuber import phi_huber_weighted
from src.crossSectionIBD import sigma_ibd
from src.energyResponse import R_i


# ============================================================
# User options
# ============================================================
COMPARE_JUNO = True

PLOT_PULLS = True
N_PULLS = 8
PULL_SCALE = 1.0
PULL_SEED = 123

JUNO_PATH = "data/spect-fit.txt"
DYB_PATH = "data/DYB_unfolded_spectra_tot_U235_Pu239.txt"

FIGURE_PATH = "img/noosci_JUNO-Model_with_pulls.png"


# ============================================================
# Constants
# ============================================================
kg_to_MeV = 5.61e29

m_p = 1.6726219e-27 * kg_to_MeV
m_n = 1.6749275e-27 * kg_to_MeV
m_e = 9.1093837e-31 * kg_to_MeV

Delta = m_n - m_p


alpha_huber = {
    "U235": np.array([
        4.367,
        -4.577,
        2.100,
        -5.294e-1,
        6.186e-2,
        -2.777e-3,
    ]),
    "Pu239": np.array([
        4.757,
        -5.392,
        2.563,
        -6.596e-1,
        7.820e-2,
        -3.536e-3,
    ]),
    "Pu241": np.array([
        2.990,
        -2.882,
        1.278,
        -3.343e-1,
        3.905e-2,
        -1.754e-3,
    ]),
}

frac = {
    "U235": 0.564,
    "U238": 0.076,
    "Pu239": 0.304,
    "Pu241": 0.056,
}


reactors = [
    {"name": "Taishan-1",         "P_GWth": 4.6,  "L_km": 52.77},
    {"name": "Taishan-2",         "P_GWth": 4.6,  "L_km": 52.64},
    {"name": "Yangjiang-1",       "P_GWth": 2.9,  "L_km": 52.74},
    {"name": "Yangjiang-2",       "P_GWth": 2.9,  "L_km": 52.82},
    {"name": "Yangjiang-3",       "P_GWth": 2.9,  "L_km": 52.41},
    {"name": "Yangjiang-4",       "P_GWth": 2.9,  "L_km": 52.49},
    {"name": "Yangjiang-5",       "P_GWth": 2.9,  "L_km": 52.11},
    {"name": "Yangjiang-6",       "P_GWth": 2.9,  "L_km": 52.19},
    {"name": "DayaBay-effective", "P_GWth": 17.4, "L_km": 215.0},
]


# ============================================================
# Helper functions
# ============================================================
def load_juno_spectrum(path):
    """
    Load the JUNO spectrum file.
    """
    df = pd.read_csv(path, sep=r"\s+", header=None)

    df.columns = [
        "energy",
        "reactor_signal",
        "reactor_background",
        "data",
        "unoscillated_signal",
    ]

    c_norm = float(np.max(df["unoscillated_signal"]))

    return df, c_norm


def build_dyb_flux_model(path):
    """
    Build the continuous Daya Bay flux model.

    Returns
    -------
    phi_cont : callable
        Continuous flux model phi_cont(E_nu, xi).
    nbin : int
        Number of Daya Bay covariance modes.
    """
    df_total = read_total_flux(path, "Total")

    C_ij = read_covariance_matrix(path)
    Psi_ik = recast_covariance_matrix(C_ij)

    Phi0 = df_total["Flux"].to_numpy(dtype=float)

    E_high = df_total["E_high"].to_numpy(dtype=float)
    E_low = df_total["E_low"].to_numpy(dtype=float)
    E_center = df_total["E_center"].to_numpy(dtype=float)

    delta, splines, I = create_delta_basis(E_center)

    phi_cont, extras = build_continuous_flux_model(
        E_center=E_center,
        E_low=E_low,
        E_high=E_high,
        Phi0=Phi0,
        Psi_ik=Psi_ik,
        delta=delta,
        phi_huber_weighted=phi_huber_weighted,
        sigma_ibd=sigma_ibd,
        frac=frac,
        alpha=alpha_huber,
        Delta=Delta,
        m_e=m_e,
        N_int=500,
    )

    nbin = extras["nbin"]

    return phi_cont, nbin


def compute_reactor_weight_sum(reactor_list):
    """
    Compute the total reactor geometric weight:

        w = P_th / (4 pi L^2)

    with L converted from km to cm.
    """
    reactor_data = pd.DataFrame(reactor_list)

    km_to_cm = 1.0e5

    reactor_data["L_cm"] = reactor_data["L_km"] * km_to_cm
    reactor_data["w"] = reactor_data["P_GWth"] / (
        4.0 * np.pi * reactor_data["L_cm"]**2
    )

    w_sum = float(reactor_data["w"].sum())

    return w_sum


def make_prompt_bins():
    """
    Define prompt-energy binning.
    """
    prompt_alpha = 0.98
    prompt_beta = 0.57

    raw_edges = np.arange(0.0, 10.0 + 0.1, 0.1)

    Epr_edges = prompt_alpha * raw_edges + prompt_beta
    Epr_centers = 0.5 * (Epr_edges[:-1] + Epr_edges[1:])

    # This is the x-axis used in your original plot.
    Epr_plot = Epr_centers - prompt_beta

    return Epr_edges, Epr_centers, Epr_plot


def compute_Ni_from_pulls(
    xi_vec,
    phi_cont,
    E_nu,
    Pee,
    sig,
    Epr_edges,
    Epr_centers,
    w_sum,
    C_norm,
):
    """
    Compute the prompt spectrum for a given Daya Bay pull vector xi_vec.

    xi_vec = 0 gives the central Daya Bay flux.
    Nonzero xi_vec gives a pulled flux.
    """
    phi_E = np.asarray(phi_cont(E_nu, xi_vec), dtype=float).ravel()
    phi_E = np.clip(phi_E, 0.0, None)

    integrand_common = phi_E * sig * Pee

    Ni = np.zeros_like(Epr_centers, dtype=float)

    for i in range(len(Epr_centers)):
        Ei_lo = Epr_edges[i]
        Ei_hi = Epr_edges[i + 1]

        Ri = R_i(E_nu, Ei_lo, Ei_hi)

        Ni[i] = C_norm * w_sum * np.trapezoid(
            integrand_common * Ri,
            E_nu,
        )

    # Shape normalization to JUNO maximum
    Ni = Ni / np.max(Ni)
    Ni = C_norm * Ni

    return Ni


def generate_pull_spectra(
    n_pulls,
    pull_scale,
    pull_seed,
    nbin,
    compute_spectrum_function,
):
    """
    Generate several toy spectra from random Daya Bay pull vectors.
    """
    rng = np.random.default_rng(pull_seed)

    xi_toys = []
    Ni_toys = []

    for _ in range(n_pulls):
        xi = pull_scale * rng.normal(
            loc=0.0,
            scale=1.0,
            size=nbin,
        )

        Ni = compute_spectrum_function(xi)

        xi_toys.append(xi)
        Ni_toys.append(Ni)

    return xi_toys, Ni_toys


def plot_spectra(
    Epr_plot,
    Ni0,
    Ni_toys,
    df_JUNO,
    compare_JUNO,
    plot_pulls,
    figure_path,
):
    """
    Plot central spectrum, pull spectra, and JUNO reference.
    """
    Path(figure_path).parent.mkdir(exist_ok=True)

    plt.figure(figsize=(7.5, 4.8))

    # Central no-pull model
    plt.plot(
        Epr_plot,
        Ni0,
        "-",
        lw=2.5,
        label="no osc. Model, central",
    )

    # Pull toys
    if plot_pulls:
        for t, Ni in enumerate(Ni_toys):
            plt.plot(
                Epr_plot,
                Ni,
                lw=1.0,
                alpha=0.7,
                color="gray",
                label="DYB pull toys" if t == 0 else None,
            )

    # JUNO no-oscillation reference
    if compare_JUNO:
        plt.plot(
            df_JUNO["energy"],
            df_JUNO["unoscillated_signal"],
            "--",
            lw=2.5,
            color="red",
            label="no osc. JUNO",
        )

    plt.xlabel(r"$E_{\rm pr}$ [MeV]")
    plt.ylabel(r"Events per 0.1 MeV")
    plt.title("JUNO Reactor IBD Signal Spectrum, No Oscillation")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Save before show
    plt.savefig(figure_path, dpi=300)
    plt.show()

    print(f"Saved figure to: {figure_path}")


# ============================================================
# Main script
# ============================================================
def main():
    # --------------------------------------------------------
    # Load JUNO reference
    # --------------------------------------------------------
    df_JUNO, C_norm = load_juno_spectrum(JUNO_PATH)

    # --------------------------------------------------------
    # Build Daya Bay continuous flux model
    # --------------------------------------------------------
    phi_cont, nbin = build_dyb_flux_model(DYB_PATH)

    # --------------------------------------------------------
    # Neutrino-energy grid
    # --------------------------------------------------------
    E_nu = np.linspace(1.81, 10.0, 2000)

    # No oscillation case
    Pee = np.ones_like(E_nu)

    # IBD cross section
    sig = sigma_ibd(E_nu, Delta, m_e)

    # --------------------------------------------------------
    # Reactor weights
    # --------------------------------------------------------
    w_sum = compute_reactor_weight_sum(reactors)

    # --------------------------------------------------------
    # Prompt-energy bins
    # --------------------------------------------------------
    Epr_edges, Epr_centers, Epr_plot = make_prompt_bins()

    # --------------------------------------------------------
    # Define spectrum calculator for any pull vector
    # --------------------------------------------------------
    def compute_spectrum_for_xi(xi_vec):
        return compute_Ni_from_pulls(
            xi_vec=xi_vec,
            phi_cont=phi_cont,
            E_nu=E_nu,
            Pee=Pee,
            sig=sig,
            Epr_edges=Epr_edges,
            Epr_centers=Epr_centers,
            w_sum=w_sum,
            C_norm=C_norm,
        )

    # --------------------------------------------------------
    # Central spectrum, no pulls
    # --------------------------------------------------------
    xi0 = np.zeros(nbin, dtype=float)
    Ni0 = compute_spectrum_for_xi(xi0)

    # --------------------------------------------------------
    # Optional pull toy spectra
    # --------------------------------------------------------
    xi_toys = []
    Ni_toys = []

    if PLOT_PULLS:
        xi_toys, Ni_toys = generate_pull_spectra(
            n_pulls=N_PULLS,
            pull_scale=PULL_SCALE,
            pull_seed=PULL_SEED,
            nbin=nbin,
            compute_spectrum_function=compute_spectrum_for_xi,
        )

        print(f"Generated {N_PULLS} Daya Bay pull toy spectra.")
        print(f"Pull scale = {PULL_SCALE}")
        print(f"First pull vector, first five values = {xi_toys[0][:5]}")

    else:
        print("Using central Daya Bay flux only, no pull toys.")

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------
    plot_spectra(
        Epr_plot=Epr_plot,
        Ni0=Ni0,
        Ni_toys=Ni_toys,
        df_JUNO=df_JUNO,
        compare_JUNO=COMPARE_JUNO,
        plot_pulls=PLOT_PULLS,
        figure_path=FIGURE_PATH,
    )


if __name__ == "__main__":
    main()