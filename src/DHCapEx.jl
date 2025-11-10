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

# Include all module files
include("parameters.jl")
include("data_processing.jl")
include("helper_functions.jl")
include("model_builder.jl")
include("simulation.jl")
include("simulation_metrics.jl")
include("visualization.jl")

end # module DHCapEx
