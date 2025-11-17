"""
Simple runner script for District Heating Capacity Expansion optimization

This script demonstrates how to use the DHCapEx module to run a complete
capacity expansion optimization with representative weeks and storage.

For better performance, run Julia with multiple threads:
  julia -t auto examples/run_capacity_expansion.jl
or set the number of threads manually (e.g., julia -t 10)
"""

# Add the src directory to the load path
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

using DHCapEx

##############################################################################
# Configuration
##############################################################################

# Define paths
project_dir = dirname(@__DIR__)
data_dir = joinpath(project_dir, "data")
print(data_dir)
output_dir = joinpath(project_dir, "output")
excel_file = joinpath(data_dir, "model_parameters.xlsx")

# Simulation settings
ITERATION_LIMIT = 1000       # Maximum number of SDDP training iterations (with auto-stop)
N_SIMULATIONS = 1000         # Number of Monte Carlo simulations
RANDOM_SEED = 1234           # Random seed for reproducibility
RISK_MEASURE = :Expectation         # Risk measure: :CVaR, :Expectation, or :WorstCase
CVAR_ALPHA = 0.95            # CVaR confidence level (if using CVaR)

##############################################################################
# Main Execution
##############################################################################

println("="^80)
println("District Heating Capacity Expansion Optimization")
println("="^80)
println()

# Step 1: Load parameters from Excel
println("Step 1/5: Loading parameters from Excel...")
params = load_parameters(excel_file)
println("  Loaded parameters for $(length(params.technologies)) technologies")
println("  Planning horizon: $(params.T) model years ($(params.T * params.T_years) actual years)")
println()

# Step 2: Load and process data
println("Step 2/5: Loading and processing data...")
data = load_all_data(params, data_dir)
println("  Loaded $(data.n_weeks) representative weeks")
println()

# Step 3: Build SDDP model
println("Step 3/5: Building SDDP model...")
model = build_sddp_model(params, data)
println("  Model constructed successfully")
println()

# Step 4: Run training and simulations
println("Step 4/5: Training and simulating...")

# Set risk measure
if RISK_MEASURE == :CVaR
    using SDDP
    risk_measure = SDDP.CVaR(CVAR_ALPHA)
elseif RISK_MEASURE == :Expectation
    using SDDP
    risk_measure = SDDP.Expectation()
elseif RISK_MEASURE == :WorstCase
    using SDDP
    risk_measure = SDDP.WorstCase()
else
    error("Unknown risk measure: $RISK_MEASURE")
end

simulations = run_simulation(
    model, params, data;
    risk_measure=risk_measure,
    iteration_limit=ITERATION_LIMIT,
    n_simulations=N_SIMULATIONS,
    random_seed=RANDOM_SEED
)
println()

# Step 5: Generate outputs
println("Step 5/5: Generating outputs...")

# Save simulation results for later analysis
println("  Saving simulation results for future analysis...")
saved_filepath = save_simulation_results_auto(simulations, params, data; output_dir=output_dir)
println("  Results saved to: $saved_filepath")

# Print summary statistics
print_summary_statistics(simulations, params, data)

# Export detailed results
results_file = joinpath(output_dir, "simulation_results.txt")
export_results(simulations, params, data, results_file)

# Calculate and export performance metrics
println("  Calculating performance metrics...")
metrics = calculate_performance_metrics(simulations, params, data)
metrics_file = joinpath(output_dir, "performance_metrics.txt")
export_performance_metrics(metrics, metrics_file)

# Generate visualizations
generate_visualizations(simulations, params, data; output_dir=output_dir)

println()
println("="^80)
println("Optimization Complete!")
println("="^80)
println()
println("Output files saved to: $output_dir")
println("  - $(basename(saved_filepath)): Complete simulation data (JLD2)")
println("  - simulation_results.txt: Detailed simulation results")
println("  - performance_metrics.txt: FLH, LCOH, and energy mix metrics")
println("  - *.png: Investment and operation plots")
println("  - spaghetti_plot.html: Interactive spaghetti plots")
println()
println("To analyze saved results later, run:")
println("  julia examples/analyze_saved_results.jl $saved_filepath")
println()
