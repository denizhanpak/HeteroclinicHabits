import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import helpers

name = "markov_chain"

stay = 0.985
noise = 0
trans = 1 - (stay + noise)
mc = {0:[stay,trans,noise],1:[noise,stay,trans],2:[trans,noise,stay]}

state = 0
results = {"forward": [], "reverse": [], "turn": []}

n = 10000
for i in range(n):
    state = np.random.choice([0,1,2],p=mc[state])
    for j in range(2):
        if state == 0:
            results["forward"].append(1)
            results["reverse"].append(0)
            results["turn"].append(0)
        elif state == 1:
            results["forward"].append(0)
            results["reverse"].append(1)
            results["turn"].append(0)
        else:
            results["forward"].append(0)
            results["reverse"].append(0)
            results["turn"].append(1)

results = pd.DataFrame(results)
helpers.plot_time_series(results, name)
helpers.plot_autocorrelation(results, name)
helpers.plot_dwell_times(results, name)