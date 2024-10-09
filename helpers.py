import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.style.use('ggplot')
plt.style.use('ggplot')
plt.rcParams['font.family'] = 'sans-serif' 
plt.rcParams['font.serif'] = 'Ubuntu' 
plt.rcParams['font.monospace'] = 'Ubuntu Mono' 
plt.rcParams['font.size'] = 14 
plt.rcParams['axes.labelsize'] = 12 
plt.rcParams['axes.labelweight'] = 'bold' 
plt.rcParams['axes.titlesize'] = 12 
plt.rcParams['xtick.labelsize'] = 12 
plt.rcParams['ytick.labelsize'] = 12 
plt.rcParams['legend.fontsize'] = 12 
plt.rcParams['figure.titlesize'] = 12 
plt.rcParams['image.cmap'] = 'jet' 
plt.rcParams['image.interpolation'] = 'none' 
plt.rcParams['figure.figsize'] = (12, 10) 
plt.rcParams['axes.grid']=False
plt.rcParams['lines.linewidth'] = 2 
plt.rcParams['lines.markersize'] = 8
colors = ['xkcd:pale orange', 'xkcd:sea blue', 'xkcd:pale red', 'xkcd:sage green', 'xkcd:terra cotta', 'xkcd:dull purple', 'xkcd:teal', 'xkcd: goldenrod', 'xkcd:cadet blue',
'xkcd:scarlet']

def count_consecutive_sequences(series):
    # Find when the value changes
    changes = series != series.shift(1)
    # Use cumsum to create a group for each sequence
    groups = changes.cumsum()
    # Count the size of each group and return as a list
    sequence_lengths = series.groupby(groups).size()
    return sequence_lengths.values

def calc_dwell_time(ts, state, threshold=0.5):
    dwell_times = []
    ts = (ts-state).abs() < threshold
    return count_consecutive_sequences(ts)

def plot_time_series(data, name):
    sns.lineplot(data)
    plt.xlabel("Time")
    plt.ylabel("Active State")
    plt.savefig(name + "_timeseries.png")
    plt.clf()

def plot_autocorrelation(data, name):
    pd.plotting.autocorrelation_plot(data["forward"])
    pd.plotting.autocorrelation_plot(data["reverse"])
    pd.plotting.autocorrelation_plot(data["turn"])
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.legend(["forward", "reverse", "turn"])
    plt.savefig(name + "_autocorrelation.png")
    plt.clf()

def plot_dwell_times(data, name):
    sns.lineplot(calc_dwell_time(data["forward"],1), label="forward")
    sns.lineplot(calc_dwell_time(data["reverse"],1), label="reverse")
    sns.lineplot(calc_dwell_time(data["turn"],1), label="turn")
    plt.xlabel("Iterations")
    plt.ylabel("Dwell Time")
    plt.savefig(name + "dwell_time.png")
    plt.clf()

def integrate(states, df, time, dt, noise_mean = 0, noise_std = 0.01):
    steps = int(time / dt)
    rv = {}
    for key in states.keys():
        rv[key] = []

    for j in range(steps):
        for key in states.keys():
            states[key] += np.random.normal(noise_mean, noise_std)
            states[key] += df(states)[key] * dt
            rv[key].append(states[key])
            
    return pd.DataFrame(rv)


def VectorField(states):
    dx = -(states["x"] ** 3) + states["x"]
    dy = -(states["y"] ** 3) + states["y"] * 0.5
    rv = {"x": dx, "y": dy}
    if abs(dx) < 0.02: rv["x"] = 0
    if abs(dy) < 0.02: rv["y"] = 0
    return rv

plot_time_series(integrate(
    {"x": 1,"y": 0.2}, VectorField, 20, 0.01, 
    noise_mean = 0, noise_std = 0.001), "integration_test")