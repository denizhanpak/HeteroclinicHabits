import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def vector_field(t, y):
    dy0 = y[0] * (1 - y[0] - y[1])
    dy1 = y[1] * (1 - y[0] - y[1]) - 0 * y[0]
    return [dy0, dy1]

def simulate(fineness):
    t_span = (0, 20)  # Time span for the simulation
    t_eval = np.linspace(t_span[0], t_span[1], 1000)  # Time points where the solution is computed

    initial_conditions = np.linspace(0, 1, fineness)
    fig = plt.figure(figsize=(15, 15))

    # 2D Phase Portrait
    ax1 = fig.add_subplot(211)
    for i, x0 in enumerate(initial_conditions):
        for j, y0 in enumerate(initial_conditions):
            sol = solve_ivp(vector_field, t_span, [x0, y0], t_eval=t_eval)
            ax1.plot(sol.y[0], sol.y[1], label=f'IC: [{x0}, {y0}]')

    ax1.set_xlabel('y0')
    ax1.set_ylabel('y1')
    #ax1.legend()

    # Time Series of y0 and y1
    ax2 = fig.add_subplot(212)
    for i, x0 in enumerate(initial_conditions):
        for j, y0 in enumerate(initial_conditions):
            sol = solve_ivp(vector_field, t_span, [x0, y0], t_eval=t_eval)
            ax2.plot(t_eval, sol.y[0], label=f'y0 IC: [{x0}, {y0}]')
            ax2.plot(t_eval, sol.y[1], label=f'y1 IC: [{x0}, {y0}]', linestyle='--')

    ax2.set_xlabel('Time')
    ax2.set_ylabel('Values of y0 and y1')
    #ax2.legend()

    plt.show()

if __name__ == "__main__":
    simulate(3)