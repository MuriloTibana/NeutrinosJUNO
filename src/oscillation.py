import numpy as np

def neutrino_oscillation(E_nu, L_km, sin2_theta12, sin2_theta13, dm21, dm31):
    """This is a function that calculates the 3-flavor neutrino oscillation probability 

    Args:
        E_nu (_type_): _description_
        L_km (_type_): _description_
        sin2_theta12 (_type_): _description_
        sin2_theta13 (_type_): _description_
        dm21 (_type_): _description_
        dm31 (_type_): _description_

    Returns:
        _type_: _description_
    """
    E_nu = np.asarray(E_nu, dtype=float)

    s12_2 = sin2_theta12
    c12_2 = 1.0 - s12_2

    s13_2 = sin2_theta13
    c13_2 = 1.0 - s13_2

    sin2_2theta12 = 4.0 * s12_2 * c12_2
    sin2_2theta13 = 4.0 * s13_2 * c13_2

    dm32 = dm31 - dm21

    phase21 = 1.267e3 * dm21 * L_km / E_nu
    phase31 = 1.267e3 * dm31 * L_km / E_nu
    phase32 = 1.267e3 * dm32 * L_km / E_nu

    Pee = (
        1.0
        - c13_2**2 * sin2_2theta12 * np.sin(phase21)**2
        - sin2_2theta13 * (
            c12_2 * np.sin(phase31)**2
            + s12_2 * np.sin(phase32)**2
        )
    )

    return Pee