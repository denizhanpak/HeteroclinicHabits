using Revise
using Plots
using BifurcationKit
using DifferentialEquations
using FFTW
const BK = BifurcationKit

function sigmoid(x)
	return 1 / (1 + exp(-x))
end

function CTRNNvf(z, p)
	(;we, wi) = p
	f, p, r = z
	we *= 11
	wi *= 11
	ws = 2
	df = -f + ws * sigmoid(f) - we * sigmoid(p) - wi * sigmoid(r)
	dp = -p + ws * sigmoid(p) - we * sigmoid(r) - wi * sigmoid(f)
	dr = -r + ws * sigmoid(r) - we * sigmoid(f) - wi * sigmoid(p)
	return [df, dp, dr]
end

function CTRNNvf2(du, u, p, t)
	(;we, wi) = p
	f, p, r = u
	we *= 11
	wi *= 11
	ws = 2
	du[1] = -f + ws * sigmoid(f) - we * sigmoid(p) - wi * sigmoid(r)
	du[2] = -p + ws * sigmoid(p) - we * sigmoid(r) - wi * sigmoid(f)
	du[3] = -r + ws * sigmoid(r) - we * sigmoid(f) - wi * sigmoid(p)
end

# parameter values
par_tm = (we = .5, wi = .2)
#par_tm = (we = .01, wi = .01)

# initial condition
z0 = [0.99, 0.01, 0.01]
z1 = [0.01, 0.99, 0.01]
z2 = [0.01, 0.01, 0.99]
z =  [-1.5, -1.4, -1.5]

# Solve and plot for initial condition (0, 0, 0)
z_init = [1.0, 2.0, 3.0]
tspan = (0.0, 1000.0)  # Define the time span
prob2 = ODEProblem(CTRNNvf2, z0, tspan, par_tm)
save_at = 0.001
sol = solve(prob2, Tsit5(), abstol=1e-8, reltol=1e-8, saveat=save_at)

plot(sol2, title="CTRNN Solution", xlabel="Time", ylabel="State Variables")

start = 900000
plot(sol.t[start:end], sol[1,start:end], title="CTRNN Solution", xlabel="Time", ylabel="State Variables")
ts = sol[1,start:end]

ft = fftshift(fft(ts))
freqs =  fftshift(fftfreq(length(s), 1/save_at))

# Plot
time_domain = plot(t, signal, title="Signal", xlims=(0, 4 / 60), xlabel="time (s)", label="")
freq_domain = plot(freqs, abs.(F), title="Spectrum", xlims=(0, 200), xlabel="frequency (Hz)", label="")
plot(time_domain, freq_domain, layout = 2)
savefig("Wave.pdf")

rft = real.(ft)
ift = imag.(ft)
mag = (ift.^2 + rft.^2).^(1/2)
phase = atan.(ift./rft)

plot(mag, ylims=(0,200))

# record the solution
rec_ctrnn(x, p; k...) = (f = x[1], p = x[2], r = x[3])#, α = p[1], β = p[2])


# Bifurcation Problem
prob_we = BifurcationProblem(CTRNNvf, z, par_tm, (@optic _.we);
	record_from_solution = rec_ctrnn,)

prob_wi = BifurcationProblem(CTRNNvf, z, par_tm, (@optic _.wi);
	record_from_solution = rec_ctrnn,)

# continuation options, we limit the parameter range for E0
opts_br = ContinuationPar(p_min = 0.0, p_max = 1.1, ds = 0.0001, dsmax = 0.001)

diagram = bifurcationdiagram(prob_we, PALC(), 4, opts_br,)

plot(diagram)

# continuation of equilibria
br_we = continuation(prob_we, PALC(), opts_br; normC = norminf, bothside=true)

br_wi = continuation(prob_wi, PALC(), opts_br; bothside=true)

scene = plot(br_we, legend=:topleft)
scene = plot(br_wi, legend=:topleft)

scene = plot(br_we, legend=:topleft)

br_ei = continuation(br_we, 2, (@optic _.wi), opts_br; normC = norminf, bothside=true)

br_ie = continuation(br_wi, 2, (@optic _.we), opts_br; normC = norminf, bothside=true)

# Hopf parameter chart
scene2 = plot(br_ei, vars=(:we,:wi), legend=:topleft, color="Blue", label="Hopf Bifurcation", aspect_ratio=1, ylims=(-0.1, 1.1), xlims=(-0.1, 1.1))
plot!(br_ie, color="Blue")

# Saddle bifurcation
prob_e = BifurcationProblem(CTRNNvf, z0, par_tm, (@optic _.we);
	record_from_solution = rec_ctrnn,)
prob_i = BifurcationProblem(CTRNNvf, z0, par_tm, (@optic _.wi);
	record_from_solution = rec_ctrnn,)

# continuation options, we limit the parameter range for E0
par_tm = (we = .4, wi = .6)
opts_br = ContinuationPar(p_min = 0.0, p_max = 1.0, ds = 0.0001, dsmax = 0.008)

# continuation of equilibria
sn_e = continuation(prob_e, PALC(), opts_br; normC = norminf, bothside=true)
sn_i = continuation(prob_i, PALC(), opts_br; normC = norminf, bothside=true)
sn_ei = continuation(sn_e, 2, (@optic _.wi), opts_br; normC = norminf, bothside=true)
sn_ie = continuation(sn_i, 2, (@optic _.we), opts_br; normC = norminf, bothside=true)

plot!(sn_ei, vars=(:we,:wi), legend=:topleft, color="red", label="Fold Bifurcation", aspect_ratio=1, ylims=(-0.1, 1.1), xlims=(-0.1, 1.1))
plot!(sn_ie, vars=(:we,:wi), legend=:topleft, color="red", label="Fold Bifurcation", aspect_ratio=1, ylims=(-0.1, 1.1), xlims=(-0.1, 1.1))