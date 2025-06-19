using Plots

function plot_dwell_time_histogram(d_state::Vector{Float64}, v_state::Vector{Float64}, rv::Vector{Vector{Float64}})
    states = ["Forward", "Pause 1", "Reverse", "Pause 2"]
    bar(states, d_state, yerr=sqrt.(v_state), legend=false, xlabel="States", ylabel="Mean Dwell Time", title="Mean Dwell Time Histogram with Variance", ylims=(0, 7), label="Real Data")
    
    # Add scatter points for rv
    locations = [0.5, 1.5, 2.5, 3.5]
    scatter!(locations, rv[1], color=:blue, marker=:circle, label="Excitable GLV")
    scatter!(locations, rv[2], color=:green, marker=:circle, label="Heteroclinic GLV")
    scatter!(locations, rv[3], color=:orange, marker=:circle, label="Excitable CTRNN")
    scatter!(locations, rv[4], color=:purple, marker=:circle, label="Oscillatory CTRNN")
    plot!(legend=:topright, legend_background=:white, legend_frame=true)
end

# Example usage:
d_state = [5.329, 0.441, 1.945, 0.208]  # Replace with actual mean dwell times
v_state = [0.245, 0.032, 0.043, 0.019]  # Replace with actual variances

#GLV excitable
m1 = [29.51, 4.08, 15.1, 0.8] .* minimum(d_state) 

#GLV heteroclinic
m2 = [28.85, 4.024, 14.255, 0.9] .* minimum(d_state) 

#CTRNN excitable
m3 = [28.08, 3.47, 9.81, 1.9] .* minimum(d_state)

#CTRNN oscillatory
m4 = [24.96, 3.87, 11.81, 1.2] .* minimum(d_state)


rv = [[1.0, 0.7, 0.7, 0.4],[1.0, 0.5, 0.8, 0.3]]  # Replace with actual rv values
plot_dwell_time_histogram(d_state, v_state, [m1, m2, m3, m4])