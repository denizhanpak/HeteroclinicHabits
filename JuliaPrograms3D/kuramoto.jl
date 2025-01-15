using DifferentialEquations
using StaticArrays
using Plots
using LinearAlgebra
using Plots
"""
Kuramoto-model setup and preallocated member variables (N oscillators). This is mutable, and the differential equation does not have any memory, allowing for reuse of the same model, or for altering variables without creating a new instance.

Constructor:

u0 - Initial phase angles (radians) for each oscillator, given as Vector{Float64}

ω - Natural frequencies (Hz) for each oscillator, given as Vector{Float64}

K - Coupling constant (Hz), in the form of an adjacency matrix Matrix{Float64, Float64} (allows for non-uniform coupling)

tstart - Start time of the simulation, given as Float64

tend - End time of the simulation, given as Float64

dt - Time step of the simulation, given as Float64

All other variables are reserved for preallocation. Altering them will have no effect, and if it does you're deliberately trying to mess it up.
"""
mutable struct Kuramoto
    # Set elements
    u0::Vector{Float64}
    ω::Vector{Float64}
    K::Matrix{Float64}
    tstart::Float64
    tend::Float64
    dt::Float64
    D::Float64

    # Private (preallocated)
    tspan::Tuple{Float64, Float64}
    du::Vector{Float64}
    A1::Matrix{Float64}
    A2::Matrix{Float64}
    v1::Vector{Float64}
    v2::Vector{Float64}
    sol::Any

    Kuramoto(u0, ω, K, tstart, tend, dt) = new(u0, ω, K, tstart, tend, dt, 0, (tstart, tend),
                                               zeros(N), zeros(N, N), zeros(N, N), zeros(N), zeros(N))
    Kuramoto(u0, ω, K, tstart, tend, dt, D) = new(u0, ω, K, tstart, tend, dt, D, (tstart, tend),
                                               zeros(N), zeros(N, N), zeros(N, N), zeros(N), zeros(N))
end

function kuramoto_static!(du, u, p, t)
    ω, K, N, uT, A1, A2, v1, v2, D = p;

    uT .= u';

    A1 .= u .- uT; # Finding phase differences
    A2 .= sin.(A1);

    sum!(v1, A2); # Summing along sin(phase)
    mul!(v2, K, v1); # Adjancency matrix -> Interaction term

    du .= ω .+ v2; # Step
end

"""
Static solver for the N-oscillator ordinary Kuramoto problem.

model - Kuramoto model object, set up prior to the run.
"""
function run_kuramoto_static(model::Kuramoto, abstol, reltol)
    p = (model.ω, model.K, length(model.u0), (model.u0)', model.A1, model.A2, model.v1, model.v2, model.D);
    prob = ODEProblem(kuramoto_static!, model.u0, model.tspan, p);
    model.sol = solve(prob, Tsit5(), abstol=abstol,reltol=reltol, dt=model.dt);

    nothing
end

function kuramoto!(du, u, p, t)
    ω, K, N = p

    du .= ω .+ K * sum(sin.(u .- u'), dims=2); # Step
end

"""
Dynamic solver for the N-oscillator ordinary Kuramoto problem.

model - Kuramoto model object, set up prior to the run.
"""
function run_kuramoto(model::Kuramoto, abstol, reltol)
    p = (model.ω, model.K, length(model.u0), (model.u0)', model.A1, model.A2, model.v1, model.v2, model.D);
    prob = ODEProblem(kuramoto!, model.u0, model.tspan, p);
    model.sol = solve(prob, Tsit5(), abstol=abstol,reltol=reltol, dt=model.dt);
    nothing
end

function kuramoto_stochastic(u, p, t)
    ω, K, N, uT, A1, A2, v1, v2, nV = p

    A1 = u' .- u # Phase differences
    A2 = sin.(-A1)

    v1 = sum(A2, dims=2) # Summing along rows
    v2 = K * v1 # Adjancency matrix -> Interaction term

    return SVector{N, Float64}(ω .+ v2)
end

"""
Regular white noise generator used for the stochastic Kuramoto model

Simply returns the nV (noise vector), which is composed of the white noise strength D (Hz)
"""
function kuramoto_noise(u, p, t)
    ω, K, N, uT, A1, A2, v1, v2, nV = p
    return nV # Uncorrelated white noise of power D
end

"""
Static and stochastic solver for the N-oscillator ordinary Kuramoto problem.

model - Kuramoto model object, set up prior to the run.
"""
function run_kuramoto_static_stochastic(model::Kuramoto, abstol, reltol)
    W = WienerProcess(0.0,0.0,0.0)
    N = length(model.u0)
    nV = @SVector fill(model.D, N)
    nV = nV ./ N # Normalizing the noise
    p = (model.ω, model.K, length(model.u0), (model.u0)', model.A1, model.A2, model.v1, model.v2, nV)
    prob = SDEProblem(kuramoto_stochastic, kuramoto_noise, model.u0, model.tspan, p, noise=W)
    model.sol = solve(prob, abstol=abstol,reltol=reltol, dt=model.dt)
end

function create_standard_K(k, N)
    K = fill(k, N, N) / N
    K[diagind(K)] .= 0.0
    return K
end

function plot_model_frequencies(model::Kuramoto, step::Int64=1, legend::Bool=true)
    sol = model.sol
    t = sol.t
    p = plot(legend=false)
    if legend
        for i in 1:length(model.ω)
            plot!(p, t[2:step:end], (diff(sol[i, :]) ./ diff(t))[1:step:end] / 1e9, label="Oscillator " * string(i), linewidth=5)
        end
    else
        for i in 1:length(model.ω)
            plot!(p, t[2:step:end], (diff(sol[i, :]) ./ diff(t))[1:step:end] / 1e9, linewidth=5)
        end
    end
    xlabel!(p, "Time")
    ylabel!(p, "Frequency")
    savefig(p, "kuramoto_frequencies.png") # Save the plot to a file
end

function plot_model_phases(model::Kuramoto)
    sol = model.sol
    t = sol.t
    p = plot()
    for i in 1:length(model.ω)
        plot!(p, t, sol[i, :], label="Oscillator " * string(i))
    end
    xlabel!(p, "Time")
    ylabel!(p, "Frequency")
    savefig(p,"kuramoto.png")
    #display(p)
end

function order_parameter(model::Kuramoto)
    sol = model.sol
    N = length(model.u0)
    cosmat = cos.(model.sol)
    sinmat = sin.(model.sol)
    sum_ = similar(cosmat)
    fill!(sum_, 0.0)
    for i in 1:N
        for j in 1:N
            if i != j
                sum_[i,:] .= cosmat[i,:].*cosmat[j,:] .+ sinmat[i,:].*sinmat[j,:]
            end
        end
    end
    sum_ = sqrt.(N .+ sum(sum_, dims=1)) ./ N
    return sum_
end

function run_and_plot(model::Kuramoto, abstol, reltol)
    run_kuramoto_static(model, abstol, reltol)
    plot_model_frequencies(model)
    plot_model_phases(model)
end

# Example usage:
u0 = randn
model = Kuramoto(u0, ω, K, tstart, tend, dt)
run_and_plot(model, 1e-6, 1e-6)