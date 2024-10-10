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
    plt.savefig(name + "_dwell_time.png")
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

def plot_phase_plane(data, name):
    sns.scatterplot(x = data["forward"], y = data["reverse"])
    plt.xlabel("Forward")
    plt.ylabel("Reverse")
    plt.savefig(name + "_phase_plane.png")
    plt.clf()

def plot_phase_plane_3d(data, name):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data["forward"], data["reverse"], data["turn"])
    ax.set_xlabel("Forward")
    ax.set_ylabel("Reverse")
    ax.set_zlabel("Turn")
    plt.savefig(name + "_phase_plane_3d.png")
    plt.clf()

def run_simulation(name, states, df, time, dt, noise_mean = 0, noise_std = 0.01):
    results = integrate(states, df, time, dt, noise_mean, noise_std)
    plot_time_series(results, name)
    plot_autocorrelation(results, name)
    plot_dwell_times(results, name)
    plot_phase_plane(results, name)
    plot_phase_plane_3d(results, name)
    return results