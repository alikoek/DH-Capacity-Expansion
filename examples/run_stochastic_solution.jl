"""
Simple runner script for District Heating Capacity Expansion optimization

This script demonstrates how to use the DHCapEx module to run a complete
capacity expansion optimization with representative weeks and storage.
"""
##
# Add the src directory to the load path
using Revise
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src", "julia"))
using DHCapEx
using Plots
using SparseArrays
##
##############################################################################
# Configuration
##############################################################################

# Define paths
project_dir = dirname(@__DIR__)
data_dir = joinpath(project_dir, "data")
output_dir = joinpath(project_dir, "output")
excel_file = joinpath(data_dir, "model_parameters.xlsx")

# Simulation settings
ITERATION_LIMIT = 1000        # Number of SDDP training iterations
N_SIMULATIONS = 1000          # Number of Monte Carlo simulations
RANDOM_SEED = 1234          # Random seed for reproducibility
RISK_MEASURE = :Expectation        # Risk measure: :CVaR, :Expectation, or :WorstCase
CVAR_ALPHA = 0.95           # CVaR confidence level (if using CVaR)

##
##############################################################################
# Main Execution
##############################################################################

##
##################################################################
# SDDP execution: getting the value of the stochastic solution VSS
##################################################################
println("======================")
println("STOCHASTIC PROBLEM")
println("======================")

# Step 1: Load parameters from Excel
println("Step 1/6: Loading parameters from Excel...")
params = load_parameters(excel_file)
println("  Loaded parameters for $(length(params.technologies)) technologies")
println("  Planning horizon: $(params.config_dict[:T]) model years ($(params.config_dict[:T] * params.config_dict[:T_years]) actual years)")
println("  # policy: $(size(params.policy_proba_df,1)) policy scenarios")
println("  # price: $(size(params.price_proba_df,1)) price scenarios")
println("  # temperature: $(size(params.temperature_proba_df,1)) temperature scenarios")

# Step 2: Load and process data
println("Step 2/6: Loading and processing data...")
data = load_all_data(data_dir)

# small modifications
println("  Loaded $(length(data.period_indexes)) representative periods of $(length(data.hour_indexes)) timesteps")
println()

# Step 3: Build SDDP model
println("Step 3/6: Building SDDP model...")
model = build_sddp_model(params, data)
println("  Model constructed successfully")
println()

# # Step 4: Run training and simulations
println("Step 4/6: Training ...")

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

simulations = train_model(
    model;
    risk_measure=risk_measure,
    iteration_limit=ITERATION_LIMIT,
    # random_seed=RANDOM_SEED
)
SDDP.write_log_to_csv(model,joinpath(output_dir, "stochastic_solution",  "training_results.csv"))


# # Step 5: Run training and simulations
println("Step 5/6: Simulating ...")
simulations = run_simulation(
    model,
    n_simulations=N_SIMULATIONS,
    random_seed=RANDOM_SEED
)

# # Step 5: Generate outputs
println("Step 6/6: Generating outputs...")

# # Print summary statistics
print_summary_statistics(simulations, params, data)

# # Export detailed results
results_file = joinpath(output_dir, "stochastic_solution", "simulation_results.txt")
export_results(simulations, params, data, results_file)

# # Generate visualizations
generate_visualizations(simulations, params, data, output_dir=joinpath(output_dir,"stochastic_solution"))

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

