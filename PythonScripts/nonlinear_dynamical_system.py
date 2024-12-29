import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def vector_field(t, y):
    dy0 = y[0] * (1 - y[0] - y[1])
    dy1 = y[1] * (1 - y[1] - y[0])
    return [dy0, dy1]

def simulate(fineness):
    t_span = (0, 10)  # Time span for the simulation
    t_eval = np.linspace(t_span[0], t_span[1], 1000)  # Time points where the solution is computed

    initial_conditions = np.linspace(0, 1, fineness)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # Phase Portrait
    for i, x0 in enumerate(initial_conditions):
        for j, y0 in enumerate(initial_conditions):
            sol = solve_ivp(vector_field, t_span, [x0, y0], t_eval=t_eval)
            ax1.plot(sol.y[0], sol.y[1], label=f'IC: [{x0}, {y0}]')

    ax1.set_xlabel('y0')
    ax1.set_ylabel('y1')
    ax1.set_title('Phase Portrait')

    # Time Series of y0
    for i, x0 in enumerate(initial_conditions):
        for j, y0 in enumerate(initial_conditions):
            sol = solve_ivp(vector_field, t_span, [x0, y0], t_eval=t_eval)
            ax2.plot(t_eval, sol.y[0], label=f'IC: [{x0}, {y0}]')

    ax2.set_xlabel('Time')
    ax2.set_ylabel('y0')
    ax2.set_title('Time Series of y0')

    plt.tight_layout()
    plt.show()

simulate(10)