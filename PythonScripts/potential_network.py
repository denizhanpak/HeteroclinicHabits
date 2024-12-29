import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the potential function
def potential(x, y, z, alpha=1.0, beta=1.0, gamma=1.0):
    quadratic_term = alpha * (x**2 + y**2 + z**2)
    quartic_term = beta * (x**2 + y**2 + z**2)**2
    cyclic_term = gamma * (
        x**3 - 3*x*y**2 - 3*x*z**2 +
        y**3 - 3*y*x**2 - 3*y*z**2 +
        z**3 - 3*z*x**2 - 3*z*y**2
    )
    return quadratic_term + quartic_term + cyclic_term

# Define the gradient (vector field) of the potential
def gradient(x, y, z, alpha=1.0, beta=1.0, gamma=1.0):
    dV_dx = (
        2*alpha*x + 4*beta*x*(x**2 + y**2 + z**2) +
        3*gamma*(x**2 - y**2 - z**2)
    )
    dV_dy = (
        2*alpha*y + 4*beta*y*(x**2 + y**2 + z**2) +
        3*gamma*(y**2 - x**2 - z**2)
    )
    dV_dz = (
        2*alpha*z + 4*beta*z*(x**2 + y**2 + z**2) +
        3*gamma*(z**2 - x**2 - y**2)
    )
    return np.array([dV_dx, dV_dy, dV_dz])

# Create a grid of points for visualization
x = np.linspace(-1.5, 1.5, 10)
y = np.linspace(-1.5, 1.5, 10)
z = np.linspace(-1.5, 1.5, 10)
X, Y, Z = np.meshgrid(x, y, z)

# Evaluate the vector field on the grid
U, V, W = gradient(X, Y, Z)

# Define attractor positions
attractors = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])

# Visualize the vector field
fig = plt.figure(figsize=(14, 7))
ax = fig.add_subplot(121, projection='3d')
ax.quiver(X, Y, Z, U, V, W, length=0.3, normalize=True)
ax.set_title("Vector Field")
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")

# Visualize the flow plot
ax2 = fig.add_subplot(122, projection='3d')
for attractor in attractors:
    ax2.scatter(*attractor, color='red', s=100, label=f'Attractor {tuple(attractor)}')

# Create streamlines (approximated by trajectories)
def simulate_flow(x0, y0, z0, steps=200, dt=0.05):
    trajectory = [[x0, y0, z0]]
    for _ in range(steps):
        x, y, z = trajectory[-1]
        dx, dy, dz = gradient(x, y, z)
        norm = np.linalg.norm([dx, dy, dz])
        if norm == 0:
            break
        x_new = x - dt * dx / norm
        y_new = y - dt * dy / norm
        z_new = z - dt * dz / norm
        trajectory.append([x_new, y_new, z_new])
    return np.array(trajectory)

# Plot trajectories starting from random points
np.random.seed(42)
initial_points = np.random.uniform(-1.5, 1.5, (15, 3))
for x0, y0, z0 in initial_points:
    trajectory = simulate_flow(x0, y0, z0)
    ax2.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], label=f"Flow from ({x0:.1f}, {y0:.1f}, {z0:.1f})")

ax2.set_title("Flow Plot")
ax2.set_xlabel("X-axis")
ax2.set_ylabel("Y-axis")
ax2.set_zlabel("Z-axis")

# Remove duplicate labels
handles, labels = ax2.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax2.legend(by_label.values(), by_label.keys())

plt.tight_layout()
plt.show()
