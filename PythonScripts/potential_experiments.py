import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Define the potential function (not plotted, used for dynamics)
def potential(x, y, a):
    return 0.5 * (x**2 + y**2) - 0.25 * (x**4 + y**4) + a * (x**2 * y - x * y**2)

# Define the gradient (vector field)
def vector_field(x, y, a):
    dx = (x - x**3 + 2 * a * x * y - a * y**2)
    dy = (y - y**3 + 2 * a * x * y - a * x**2)
    return dx, dy

# Find zeros of the vector field
def find_zeros(a):
    zeros = []
    for x in np.linspace(-1.5, 1.5, 100):
        for y in np.linspace(-1.5, 1.5, 100):
            dx, dy = vector_field(x, y, a)
            if np.hypot(dx, dy) < 1e-1:  # Threshold for zero
                zeros.append((x, y))
    return zeros

# Determine stability of zeros
def is_stable(x, y, a):
    dx, dy = vector_field(x, y, a)
    return np.hypot(dx, dy) < 1e-2  # Threshold for stability

# Create a grid
x = np.linspace(-1.5, 1.5, 20)
y = np.linspace(-1.5, 1.5, 20)
X, Y = np.meshgrid(x, y)

# Initial parameter value
initial_a = 0.0

# Compute the initial vector field
DX, DY = vector_field(X, Y, initial_a)
M = np.hypot(DX, DY)  # Magnitude of the vectors
M[M == 0] = 1  # Avoid division by zero
DX /= M
DY /= M

# Create the figure and axes
fig, ax = plt.subplots(figsize=(8, 6))
plt.subplots_adjust(bottom=0.25)  # Space for slider

# Plot the vector field
quiver = ax.quiver(X, Y, DX, DY, M, scale=20, scale_units='xy', norm=plt.Normalize(), cmap='viridis')
ax.set_title("Vector Field with Parameter a")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
ax.axvline(0, color='gray', linewidth=0.5, linestyle='--')
ax.grid()
plt.colorbar(quiver, ax=ax, label='Vector Magnitude')

# Plot zeros
zeros = find_zeros(initial_a)
stable_zeros = [zero for zero in zeros if is_stable(*zero, initial_a)]
unstable_zeros = [zero for zero in zeros if not is_stable(*zero, initial_a)]
stable_zeros = np.array(stable_zeros)
unstable_zeros = np.array(unstable_zeros)
stable_plot, = ax.plot([], [], 'bo', label='Stable Zeros')
unstable_plot, = ax.plot([], [], 'ro', label='Unstable Zeros')
if stable_zeros.size > 0:
    stable_plot.set_data(stable_zeros[:, 0], stable_zeros[:, 1])
if unstable_zeros.size > 0:
    unstable_plot.set_data(unstable_zeros[:, 0], unstable_zeros[:, 1])
ax.legend()

# Add a slider for the parameter 'a'
ax_a = plt.axes([0.2, 0.1, 0.6, 0.03], facecolor='lightgoldenrodyellow')
slider_a = Slider(ax_a, "a", -2.0, 2.0, valinit=initial_a)

# Update function for the slider
def update(val):
    a = slider_a.val
    DX, DY = vector_field(X, Y, a)
    M = np.hypot(DX, DY)  # Magnitude of the vectors
    M[M == 0] = 1  # Avoid division by zero
    DX /= M
    DY /= M
    quiver.set_UVC(DX, DY, M)  # Update vector field
    
    # Update zeros
    zeros = find_zeros(a)
    stable_zeros = [zero for zero in zeros if is_stable(*zero, a)]
    unstable_zeros = [zero for zero in zeros if not is_stable(*zero, a)]
    stable_zeros = np.array(stable_zeros)
    unstable_zeros = np.array(unstable_zeros)
    stable_plot.set_data([], [])
    unstable_plot.set_data([], [])
    if stable_zeros.size > 0:
        stable_plot.set_data(stable_zeros[:, 0], stable_zeros[:, 1])
    if unstable_zeros.size > 0:
        unstable_plot.set_data(unstable_zeros[:, 0], unstable_zeros[:, 1])
    ax.legend()
    
    fig.canvas.draw_idle()

# Connect slider to update function
slider_a.on_changed(update)

plt.show()
