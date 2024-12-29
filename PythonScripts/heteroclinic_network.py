import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import helpers

name = "heteroclinic_network"

ai = -0.6
ae = 0.4

def VectorField(states):
    X = states["forward"] ** 2 + states["reverse"] ** 2 + states["turn"] ** 2
    dforward = states["forward"] * (1 - X + ai * states["reverse"] ** 2 + ae * states["turn"] ** 2)
    dreverse = states["reverse"] * (1 - X + ai * states["turn"] ** 2 + ae * states["forward"] ** 2)
    dturn = states["turn"] * (1 - X + ai * states["forward"] ** 2 + ae * states["reverse"] ** 2)
    rv =  {"forward": dforward, "reverse": dreverse, "turn": dturn}
    for key in rv.keys():
        if abs(rv[key]) < 0.02: rv[key] = 0
    return rv


results = helpers.integrate(
    {"forward": 0.1, "reverse": 0.1, "turn": 0.1},
    VectorField, 150, 0.01, noise_mean=0, noise_std=0.001
)
helpers.plot_time_series(results, name)
helpers.plot_autocorrelation(results, name)
helpers.plot_dwell_times(results, name)