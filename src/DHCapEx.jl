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
- `extract_ev_investments(ev_model, ev_variables, params)`: Extract EV investment decisions from JuMP model
- `extract_ev_investments_from_simulations(ev_simulations, params)`: Extract EV investment decisions from SDDP simulations
- `evaluate_ev_policy(sddp_model, ev_investments, params, n_scenarios)`: Evaluate EV policy under uncertainty
- `build_ev_sddp_model_integrated(params, data)`: Build SDDP-based EV model with expected values

# Results I/O Functions
- `save_simulation_results(simulations, params, data, filepath)`: Save results to disk
- `save_simulation_results_auto(simulations, params, data)`: Save with auto-generated timestamp
- `load_simulation_results(filepath)`: Load previously saved results
- `list_saved_results(directory)`: List all saved result files
- `extract_extreme_events_info(simulations, params)`: Extract extreme event occurrences
- `get_simulation_costs(simulations)`: Extract total costs from simulations
- `filter_simulations_by_extreme_event(...)`: Filter simulations by extreme event type

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
# export build_deterministic_model  # Deprecated - use build_ev_sddp_model_integrated instead
export extract_ev_investments, evaluate_ev_policy, extract_ev_investments_from_simulations
export build_ev_sddp_model_integrated, verify_ev_model_structure  # EV model functions

# Export results I/O functions
export save_simulation_results, save_simulation_results_auto
export load_simulation_results, list_saved_results
export extract_extreme_events_info, get_simulation_costs
export filter_simulations_by_extreme_event

# Include all module files
include("parameters.jl")
include("data_processing.jl")
include("helper_functions.jl")
include("model_builder.jl")
include("simulation.jl")
include("simulation_metrics.jl")
include("visualization.jl")

# VSS analysis modules
# include("deterministic_model.jl")  # Deprecated - use ev_model_integrated.jl instead
include("ev_policy_evaluation.jl")
include("ev_model_integrated.jl")  # The WORKING EV model implementation
# include("two_stage_proper.jl")  # Conceptual two-stage model (incomplete, for future development)

# Results I/O module
include("results_io.jl")

end # module DHCapEx
