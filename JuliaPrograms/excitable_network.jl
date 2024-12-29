using Random
using Distributions
using Plots

# Define the potential function V(x, y, z; ε)
function potential(x, y, z, ε)
    return 0.5 * (x^2 + y^2 + z^2) - ε * (x^2 * y + y^2 * z + z^2 * x)
end

# Define the gradient of the potential (vector field)
function vector_field(x, y, z, ε)
    # Compute the partial derivatives of V with respect to x, y, z
    Fx = - (x - 2 * ε * x * y - ε * z^2)
    Fy = - (y - 2 * ε * y * z - ε * x^2)
    Fz = - (z - 2 * ε * z * x - ε * y^2)
    return Fx, Fy, Fz
end

# Define the stochastic integration function
function stochastic_integration(x0, y0, z0, ε, dt, T, noise_level)
    x, y, z = x0, y0, z0
    n_steps = Int(T / dt)
    trajectory = [(x, y, z)]
    for _ in 1:n_steps
        Fx, Fy, Fz = vector_field(x, y, z, ε)
        x += Fx * dt + noise_level * sqrt(dt) * rand(Normal(0, 1))
        y += Fy * dt + noise_level * sqrt(dt) * rand(Normal(0, 1))
        z += Fz * dt + noise_level * sqrt(dt) * rand(Normal(0, 1))
        push!(trajectory, (x, y, z))
    end
    return trajectory
end

# Integrate from a list of initial conditions
function integrate_from_initial_conditions(initial_conditions, ε, dt, T, noise_level)
    results = []
    for (x0, y0, z0) in initial_conditions
        trajectory = stochastic_integration(x0, y0, z0, ε, dt, T, noise_level)
        push!(results, trajectory)
    end
    return results
end

# Example usage
initial_conditions = [(1.0, 1.0, 1.0), (0.5, 0.5, 0.5), (1.5, 1.5, 1.5)]  # List of initial conditions
ε = 0.1                  # Coupling parameter
dt = 0.001                 # Time step
T = 100.0                  # Total integration time
noise_level = 0.01     # Noise level

results = integrate_from_initial_conditions(initial_conditions, ε, dt, T, noise_level)

# Plot the trajectories
plot()
for (i, trajectory) in enumerate(results)
    xs, ys, zs = [p[1] for p in trajectory], [p[2] for p in trajectory], [p[3] for p in trajectory]
    plot!(xs, ys, zs, label="Trajectory $i")
end
xlabel!("x")
ylabel!("y")
zlabel!("z")
title!("Stochastic Trajectories")
savefig("stochastic_trajectories.png")

# Plot the time series for the x, y, and z axes of the first trajectory
time_points = 0:dt:T
x_series = [p[1] for p in results[1]]
y_series = [p[2] for p in results[1]]
z_series = [p[3] for p in results[1]]

plot(time_points, x_series, label="x(t)", xlabel="Time", ylabel="x", title="Time Series of x-axis")
plot!(time_points, y_series, label="y(t)", xlabel="Time", ylabel="y", title="Time Series of y-axis")
plot!(time_points, z_series, label="z(t)", xlabel="Time", ylabel="z", title="Time Series of z-axis")
savefig("time_series_xyz.png")
