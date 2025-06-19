using Plots
using ReLambertW

ν = 1.0
β = 1.0
τ = 1.0
κ = 1.0/10.0
ϵ = 0.01

γ = 0.577215664901533

function GetGrowthRate(t, b, v, k, e, y)
    term1 = - v / (2 * t)
    term2 = 2 * (log(e) - log(k * b))
    term3 = log(1 + 1/v)
    term4 = log(t/2) - y
    term5 = -4/v * log(1/k - 1) - 1im

    print(term2 + term3 + term4 + term5)

    return term1 * womega(term2 + term3 + term4 + term5)
end

GetGrowthRate(τ, β, ν, κ, ϵ, γ)