"""
DHCapEx - District Heating Capacity Expansion Model

A modular Julia package for optimizing capacity expansion decisions in district heating
systems using Stochastic Dual Dynamic Programming (SDDP).

# Main Functions
- `load_parameters(excel_path)`: Load model parameters from Excel file
- `load_all_data(params, data_dir)`: Load and process representative weeks and electricity prices
- `build_sddp_model(params, data)`: Construct the SDDP optimization model
- `run_simulation(model, params, data; kwargs...)`: Train model and run simulations
- `generate_visualizations(simulations, params, data; output_dir)`: Create all plots
- `export_results(simulations, params, data, output_file)`: Export detailed results
- `print_summary_statistics(simulations, params, data)`: Print summary statistics
- `calculate_performance_metrics(simulations, params, data)`: Calculate performance metrics
- `export_performance_metrics(metrics, output_file)`: Export performance metrics to file

# VSS Analysis Functions
- `build_deterministic_model(params, data)`: Build expected-value deterministic benchmark
- `extract_ev_investments(ev_model, ev_variables, params)`: Extract EV investment decisions
- `evaluate_ev_policy(sddp_model, ev_investments, params, n_scenarios)`: Evaluate EV policy under uncertainty

# Example Usage
```julia
using DHCapEx

# Load parameters and data
params = load_parameters("data/model_parameters.xlsx")
data = load_all_data(params, "data/")

# Build and solve model
model = build_sddp_model(params, data)
simulations = run_simulation(model, params, data; iteration_limit=100, n_simulations=400)

# Generate outputs
generate_visualizations(simulations, params, data; output_dir="output/")
export_results(simulations, params, data, "output/simulation_results.txt")
print_summary_statistics(simulations, params, data)

# Calculate performance metrics
metrics = calculate_performance_metrics(simulations, params, data)
export_performance_metrics(metrics, "output/performance_metrics.txt")
```
"""
module DHCapEx

# Import SDDP at module level so SDDP.JuMP is available to submodules
using SDDP

# Export main functions and types
export load_parameters, ModelParameters
export load_all_data, ProcessedData
export build_sddp_model
export run_simulation, export_results, print_summary_statistics
export calculate_performance_metrics, export_performance_metrics
export generate_visualizations
export decode_markov_state

# Export VSS analysis functions
export build_deterministic_model
export extract_ev_investments, evaluate_ev_policy

# Include all module files
include("parameters.jl")
include("data_processing.jl")
include("helper_functions.jl")
include("model_builder.jl")
include("simulation.jl")
include("simulation_metrics.jl")
include("visualization.jl")

# VSS analysis modules
include("deterministic_model.jl")
include("ev_policy_evaluation.jl")

end # module DHCapEx
