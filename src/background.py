import numpy as np

def normalizeToTable(shape, component_name, BIN_WIDTH, TABLE1_TOTAL_EVENTS):
    """
    Normalize the interpolated shape so that its total number of events is:

        Table 1 rate x live days.

    The Fig. 3 y-axis is events per 0.1 MeV.
    Since BIN_WIDTH = 0.1 MeV, each bin value is already events per bin.
    """

    shape = np.asarray(shape, dtype=float)
    shape = np.clip(shape, 0.0, None)

    counts_per_bin = shape * (BIN_WIDTH / 0.1)

    current_total = np.sum(counts_per_bin)

    if current_total <= 0:
        raise ValueError(f"{component_name} has zero total before normalization.")

    target_total = TABLE1_TOTAL_EVENTS[component_name]

    return counts_per_bin * target_total / current_total

def interpolateToBins(E_raw, y_raw, E_bins):
    """
    Interpolate raw digitized curve onto model bin centers.
    """

    good = np.isfinite(E_raw) & np.isfinite(y_raw)

    E = E_raw[good]
    y = y_raw[good]

    order = np.argsort(E)

    E = E[order]
    y = y[order]

    y_interp = np.interp(
        E_bins,
        E,
        y,
        left=0.0,
        right=0.0,
    )

    return np.clip(y_interp, 0.0, None)

def draw_physical_pull(rng, fractional_uncertainty):
    """
    Draw xi ~ N(0, 1), requiring

        1 + sigma * xi >= 0

    so that the resulting event rate cannot become negative.
    """

    while True:
        xi = rng.normal(loc=0.0, scale=1.0)

        normalization_factor = (
            1.0
            + fractional_uncertainty * xi
        )

        if normalization_factor >= 0.0:
            return xi