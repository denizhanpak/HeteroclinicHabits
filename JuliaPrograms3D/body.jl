using DifferentialEquations, Plots

# Parameters
m1 = 1.0       # Mass of Rod 1
m2 = 1.0       # Mass of Rod 2
L1 = 1.0       # Length of Rod 1
L2 = 1.0       # Length of Rod 2
I1 = (1/12) * m1 * L1^2  # Moment of inertia of Rod 1
I2 = (1/12) * m2 * L2^2  # Moment of inertia of Rod 2
k = 10.0       # Torsional spring constant
damping = 0.3  # Damping coefficient for the torsional spring

# Initial conditions: [x1, y1, θ1, x2, y2, θ2, x1_dot, y1_dot, θ1_dot, x2_dot, y2_dot, θ2_dot]
u0 = [0.0, 0.0, 0.0, L1 + L2, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, -0.1]

# Time span
tspan = (0.0, 100.0)

# Function defining the system of ODEs
function rods_system!(du, u, p, t)
    # Unpack the state variables
    x1, y1, θ1, x2, y2, θ2, x1_dot, y1_dot, θ1_dot, x2_dot, y2_dot, θ2_dot = u

    # Compute torques due to the torsional spring
    torque_spring = -k * (θ2 - θ1) - damping * (θ2_dot - θ1_dot)

    # Equations of motion for Rod 1
    du[1] = x1_dot
    du[2] = y1_dot
    du[3] = θ1_dot
    du[7] = 0.0  # No external force in x-direction (ignore inertia of rods along x and y)
    du[8] = 0.0  # No external force in y-direction
    du[9] = torque_spring / I1

    # Equations of motion for Rod 2
    du[4] = x2_dot
    du[5] = y2_dot
    du[6] = θ2_dot
    du[10] = 0.0  # No external force in x-direction
    du[11] = 0.0  # No external force in y-direction
    du[12] = -torque_spring / I2
end

# Solve the system of ODEs
prob = ODEProblem(rods_system!, u0, tspan)
sol = solve(prob, Tsit5(), reltol=1e-8, abstol=1e-8)

# Visualization function
function visualize_solution(sol, L1, L2)
    plot_title = "Simulation of Two Connected Rods with a Torsional Spring"
    anim = @animate for i in 1:10:length(sol.t)
        # Extract positions and angles at time step i
        x1, y1, θ1, x2, y2, θ2 = sol.u[i][1:6]

        # Compute endpoints of the rods
        x1_end = x1 + L1/2 * cos(θ1)
        y1_end = y1 + L1/2 * sin(θ1)
        x2_start = x2 - L2/2 * cos(θ2)
        y2_start = y2 - L2/2 * sin(θ2)
        x2_end = x2 + L2/2 * cos(θ2)
        y2_end = y2 + L2/2 * sin(θ2)

        # Plot rods
        plot([x1, x1_end], [y1, y1_end], lw=4, label="Rod 1", xlim=(-2, 4), ylim=(-2, 2), legend=false)
        plot!([x2_start, x2_end], [y2_start, y2_end], lw=4, label="Rod 2")
        scatter!([x1, x2], [y1, y2], color=:red, ms=6, label="Masses")

        title!(plot_title)
        xlabel!("X-axis")
        ylabel!("Y-axis")
        aspect_ratio=:equal
    end

    gif(anim, "two_rods_simulation.gif", fps=30)
end

# Run visualization
visualize_solution(sol, L1, L2)
