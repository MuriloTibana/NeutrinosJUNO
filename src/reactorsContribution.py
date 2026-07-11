import pandas as pd
import numpy as np

def reactorsContribution(reactors):
    
    reactor_data = pd.DataFrame(reactors)
    reactor_data["w"] = reactor_data["P_GWth"] / (4*np.pi*reactor_data["L_km"]**2)

    return reactor_data["w"].sum()
