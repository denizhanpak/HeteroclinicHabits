using DifferentialEquations
using Plots
using Random
using StatsBase
using NonlinearSolve  # Use NonlinearSolve instead of NLsolve
using LinearAlgebra  # Import norm function
using PlotlyJS  # Import PlotlyJS for interactive plots

name = "oscillator_network"

#delta = 0.4
#epsilon = epsilon/8
theta = 0.5
wm = -0.68
wp = 0.23
ws = 1.0

function sigmoid(x, d, e)
    return 1 ./ (1 .+ exp.(-(x .- d) ./ e))
end

function sigmoid_prime(x, d, e)
    sig = sigmoid(x, d, e)
    return (1 .- sig) .* sig ./ e
end


for i in range(0,1,5)
    for j in range(1,9,4)
        delta = i
        epsilon = i/j
        r = 10
        x = range(-r,r,length=200)
        y1 = sigmoid(x, delta, epsilon)
        y2 = sigmoid_prime(x, delta, epsilon)
        Plots.plot(x,[y1,y2])
        Plots.savefig("ctrnn.png_$(delta)_$(epsilon).png")
    end
end