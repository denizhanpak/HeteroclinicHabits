using Plots

function plot_dwell_time_histogram(d_state::Vector{Float64}, v_state::Vector{Float64}, rv::Vector{Vector{Float64}})
    states = ["Forward", "Pause 1", "Reverse", "Pause 2"]
    bar(states, d_state, yerr=sqrt.(v_state), legend=false, xlabel="States", ylabel="Mean Dwell Time", title="Mean Dwell Time Histogram with Variance", ylims=(0, 7))
    
    # Add scatter points for rv
    locations = [0.5, 1.5, 2.5, 3.5]
    for dwell_times in rv
        scatter!(locations, dwell_times, color=:red, marker=:circle, label="Heteroclinic Dwell Times")
    end
    scatter!(locations, rv, color=:red, marker=:circle, label="RV Points")
end

# Example usage:
d_state = [5.329, 0.441, 1.945, 0.208]  # Replace with actual mean dwell times
v_state = [0.245, 0.032, 0.043, 0.019]  # Replace with actual variances
rv = [[1.0, 0.7, 0.7, 0.4],[1.0, 0.5, 0.8, 0.3]]  # Replace with actual rv values
plot_dwell_time_histogram(d_state, v_state, rv)