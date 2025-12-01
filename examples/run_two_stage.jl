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

using CSV, DataFrames
##
##############################################################################
# Configuration
##############################################################################

# Define paths
project_dir = dirname(@__DIR__)
data_dir = joinpath(project_dir, "data")
output_dir = joinpath(project_dir, "output", "two_stage")
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
println("="^80)
println("District Heating Capacity Expansion Optimization")
println("="^80)
println()
##
###########################################################################
# EEV execution: getting the value of the expected value problem's solution
###########################################################################
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


# Step 3bis: Constraints expansion
nodes = [idx for idx in keys(model.nodes)]
print(nodes)

for (node_index, node) in model.nodes
    # Check if this node belongs to the desired stage
    # For LinearPolicyGraph: node_index is just the stage number
    # For MarkovianPolicyGraph: node_index is (stage, state)
    if (isodd(node_index[1])) && node_index[1] > 1
        
        # Access the subproblem
        subproblem = node.subproblem
        
        @constraint(subproblem, KeepTech[tech in keys(params.tech_dict)], subproblem[:U_tech][tech] <= subproblem[:X_tech][tech,1].in)
        
    end
end

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

SDDP.write_log_to_csv(model, joinpath(output_dir,  "training_results.csv"))


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
results_file = joinpath(output_dir,  "simulation_results.txt")
export_results(simulations, params, data, results_file)

# # Generate visualizations
generate_visualizations(simulations, params, data, output_dir=output_dir)

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

