import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import helpers

name = "critical_network"

delta = 0.4
epsilon = delta/8
theta = .5
wm = -0.68
wp = 0.22
ws = 1
pro = 0.04 # range 0.01 to 0.04

def Sigmoid(x):
    return 1 / (1 + np.exp(-(x - delta) / epsilon))

def VectorField(states):
    dforward = -states["forward"] + ws * Sigmoid(states["forward"]) + wm * Sigmoid(states["reverse"]) + wp * Sigmoid(states["turn"]) - states["body"] * pro * Sigmoid(states["turn"])
    dreverse = -states["reverse"] + ws * Sigmoid(states["reverse"]) + wm * Sigmoid(states["turn"]) + wp * Sigmoid(states["forward"]) - states["body"] * pro * Sigmoid(states["forward"])
    dturn = -states["turn"] + ws * Sigmoid(states["turn"]) + wm * Sigmoid(states["forward"]) + wp * Sigmoid(states["reverse"]) - states["body"] * pro * Sigmoid(states["reverse"])
    dbody = - states["body"] ** 3 + states["body"] * (states["forward"] ** 2 + states["reverse"] ** 2 + states["turn"] ** 2) / 10
    rv =  {"forward": dforward, "reverse": dreverse, "turn": dturn, "body": dbody}
    for key in rv.keys():
        if abs(rv[key]) < 0.002: rv[key] = 0
    return rv

results = helpers.integrate(
    {"forward": .3, "reverse": 1, "turn": 0.1, "body": 0.1},
    VectorField, 200, 0.01, noise_mean=0, noise_std=0.00
)
helpers.plot_time_series(results, name)
helpers.plot_autocorrelation(results, name)
helpers.plot_dwell_times(results, name)