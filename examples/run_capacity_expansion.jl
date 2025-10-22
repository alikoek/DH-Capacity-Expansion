"""
Simple runner script for District Heating Capacity Expansion optimization

This script demonstrates how to use the DHCapEx module to run a complete
capacity expansion optimization with representative weeks and storage.
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
ITERATION_LIMIT = 50        # Number of SDDP training iterations
N_SIMULATIONS = 100          # Number of Monte Carlo simulations
RANDOM_SEED = 1234          # Random seed for reproducibility
RISK_MEASURE = :CVaR        # Risk measure: :CVaR, :Expectation, or :WorstCase
CVAR_ALPHA = 0.95           # CVaR confidence level (if using CVaR)

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

# Print summary statistics
print_summary_statistics(simulations, params, data)

# Export detailed results
results_file = joinpath(output_dir, "simulation_results.txt")
export_results(simulations, params, data, results_file)

# Generate visualizations
generate_visualizations(simulations, params, data; output_dir=output_dir)

println()
println("="^80)
println("Optimization Complete!")
println("="^80)
println()
println("Output files saved to: $output_dir")
println("  - simulation_results.txt: Detailed simulation results")
println("  - *.png: Investment and operation plots")
println("  - spaghetti_plot.html: Interactive spaghetti plots")
println()
