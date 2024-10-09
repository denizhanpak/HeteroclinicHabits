import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import helpers

name = "excitable_network"

a = 0.5
b = 2.5
c = 2
d = 10
e = 4
f = 2

def g(y, l):
    return -y * ((y ** 2 - 1) ** 2 + l)

def VectorField(states):
    psq = states["forward"] ** 2 + states["reverse"] ** 2 + states["turn"] ** 2
    pq = states["forward"] ** 4 + states["reverse"] ** 4 + states["turn"] ** 4
    ysq = states["p1"] ** 2 + states["p2"] ** 2 + states["p3"] ** 2
    dforward = states["forward"] * (f * (1 - psq) + d * (states["forward"] * psq - pq)) + e * (-states["p1"] ** 2 * states["forward"] * states["reverse"] + states["p3"] ** 2 * states["turn"] ** 2)
    dreverse = states["reverse"] * (f * (1 - psq) + d * (states["reverse"] * psq - pq)) + e * (-states["p2"] ** 2 * states["reverse"] * states["turn"] + states["p1"] ** 2 * states["forward"] ** 2)
    dturn = states["turn"] * (f * (1 - psq) + d * (states["turn"] * psq - pq)) + e * (-states["p3"] ** 2 * states["turn"] * states["forward"] + states["p2"] ** 2 * states["reverse"] ** 2)
    dp1 = g(states["p1"], a - b * states["forward"] ** 2 + c * (ysq - states["p1"] ** 2))
    dp2 = g(states["p2"], a - b * states["reverse"] ** 2 + c * (ysq - states["p2"] ** 2))
    dp3 = g(states["p3"], a - b * states["turn"] ** 2 + c * (ysq - states["p3"] ** 2))
    rv =  {"forward": dforward, "reverse": dreverse, "turn": dturn, "p1": dp1, "p2": dp2, "p3": dp3}
    for key in rv.keys():
        if abs(rv[key]) < 0.02: rv[key] = 0
    return rv


results = helpers.integrate(
    {"forward": 0.1, "reverse": 0.1, "turn": 0.1, "p1": 0.1, "p2": 0, "p3": 0},
    VectorField, 200, 0.01, noise_mean=0, noise_std=0.001
)
results.drop(["p1", "p2", "p3"], axis=1, inplace=True)
helpers.plot_time_series(results, name)
helpers.plot_autocorrelation(results, name)
helpers.plot_dwell_times(results, name)