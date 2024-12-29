import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def vector_field(t, y):
    dy0 = y[0] * (1 - y[0]) - y[0] * (y[1] + y[2])
    dy1 = y[1] * (1 - y[1]) - y[1] * (y[2] + y[0])
    dy2 = y[2] * (1 - y[2]) - y[2] * (y[0] + y[1])
    return [dy0, dy1, dy2]

def simulate(fineness):
    t_span = (0, 10)  # Time span for the simulation
    t_eval = np.linspace(t_span[0], t_span[1], 1000)  # Time points where the solution is computed

    initial_conditions = np.linspace(0, 1, fineness)
    fig = plt.figure(figsize=(15, 15))

    # 3D Phase Portrait
    ax1 = fig.add_subplot(211, projection='3d')
    for i, x0 in enumerate(initial_conditions):
        for j, y0 in enumerate(initial_conditions):
            for k, z0 in enumerate(initial_conditions):
                sol = solve_ivp(vector_field, t_span, [x0, y0, z0], t_eval=t_eval)
                ax1.plot(sol.y[0], sol.y[1], sol.y[2], label=f'IC: [{x0}, {y0}, {z0}]')

    ax1.set_xlabel('y0')
    ax1.set_ylabel('y1')
    ax1.set_zlabel('y2')
    #ax1.legend()

    # Time Series of y0
    ax2 = fig.add_subplot(212)
    for i, x0 in enumerate(initial_conditions):
        for j, y0 in enumerate(initial_conditions):
            for k, z0 in enumerate(initial_conditions):
                sol = solve_ivp(vector_field, t_span, [x0, y0, z0], t_eval=t_eval)
                ax2.plot(sol.t, sol.y[0], label=f'IC: [{x0}, {y0}, {z0}]')

    ax2.set_xlabel('Time')
    ax2.set_ylabel('y0')
    #ax2.legend()

    plt.show()

if __name__ == "__main__":
    fineness = 5  # Example fineness of the grid
    simulate(fineness)