# District Heating Capacity Expansion Optimization

A modular Julia framework for optimizing capacity expansion decisions in district heating systems using Stochastic Dual Dynamic Programming (SDDP) with representative weeks and thermal storage.

## Features

- **Multi-technology optimization**: CHP, Boiler, Heat Pump, Geothermal
- **Thermal storage modeling**: With charge/discharge dynamics and heat losses
- **Representative weeks**: Efficient temporal resolution using TSAM clustering
- **Stochastic optimization**: Handles uncertainty in demand and energy prices
- **Risk measures**: CVaR, expectation, worst-case
- **Vintage tracking**: Tracks capacity by investment year and retirement
- **Comprehensive visualization**: Investment plots, load duration curves, violin plots, spaghetti plots

## Project Structure

```
DH-Capacity-Expansion/
├── src/
│   ├── DHCapEx.jl              # Main module
│   ├── parameters.jl            # Parameter loading from Excel
│   ├── data_processing.jl       # Representative weeks and electricity prices
│   ├── helper_functions.jl      # Utility functions
│   ├── model_builder.jl         # SDDP model construction
│   ├── simulation.jl            # Training and simulation
│   └── visualization.jl         # Plotting functions
├── data/
│   ├── model_parameters.xlsx    # All model parameters (single Excel file)
│   ├── typical_weeks.csv        # Representative weeks from TSAM
│   ├── ElectricityPrice2030.csv # Hourly electricity prices
│   └── ElectricityPrice2050.csv
├── examples/
│   └── run_capacity_expansion.jl # Simple runner script
└── output/                      # Results and plots
```

## Installation

### Prerequisites

- Julia 1.6 or higher
- Required packages: SDDP, Gurobi, XLSX, CSV, DataFrames, Plots, StatsPlots, Distributions

### Install Dependencies

```julia
using Pkg
Pkg.add(["SDDP", "Gurobi", "XLSX", "CSV", "DataFrames", "Plots", "StatsPlots", "Distributions", "Random", "Statistics", "Dates"])
```

**Note**: Gurobi requires a license. Academic licenses are free and available at [gurobi.com](https://www.gurobi.com/academia/).

## Quick Start

### 1. Prepare Data

Ensure your data files are in the `data/` directory:
- `model_parameters.xlsx`: Excel file with parameter sheets (auto-generated)
- `typical_weeks.csv`: Representative weeks from TSAM
- `ElectricityPrice2030.csv` and `ElectricityPrice2050.csv`: Hourly electricity prices

### 2. Run the Model

```bash
cd examples
julia run_capacity_expansion.jl
```

### 3. Customize Parameters

Edit `data/model_parameters.xlsx` to modify:
- Technology costs and efficiencies
- Storage parameters
- Carbon prices
- Demand scenarios
- Model settings (planning horizon, discount rate, etc.)

## Excel Parameter File Structure

The `model_parameters.xlsx` file contains all model parameters in separate sheets:

### Sheet 1: ModelConfig
Model-wide settings (T, T_years, discount_rate, base_annual_demand, etc.)

### Sheet 2: Technologies
Technology-specific parameters:
- Initial and maximum additional capacity
- Investment and O&M costs
- Thermal and electrical efficiencies
- Energy carrier type
- Lifetimes (new and initial)

### Sheet 3: Storage
Storage parameters (capacity cost, efficiency, loss rate, charge/discharge rates, lifetime)

### Sheet 4: EnergyCarriers
Emission factors for each energy carrier

### Sheet 5: CarbonPrice
Carbon price trajectory by model year

### Sheet 6: DemandMultipliers
Demand multiplier states for Markovian demand uncertainty

## Usage Example

```julia
# Add module to load path
push!(LOAD_PATH, "src")
using DHCapEx

# Load parameters and data
params = load_parameters("data/model_parameters.xlsx")
data = load_all_data(params, "data/")

# Build and solve model
model = build_sddp_model(params, data)
simulations = run_simulation(model, params, data;
                            iteration_limit=100,
                            n_simulations=400)

# Generate outputs
generate_visualizations(simulations, params, data; output_dir="output/")
export_results(simulations, params, data, "output/simulation_results.txt")
print_summary_statistics(simulations, params, data)
```

## Advanced Usage

### Custom Risk Measures

```julia
using SDDP

# CVaR
risk_measure = SDDP.CVaR(0.95)

# Expectation
risk_measure = SDDP.Expectation()

# Worst case
risk_measure = SDDP.WorstCase()

simulations = run_simulation(model, params, data; risk_measure=risk_measure)
```

### Modifying Individual Components

```julia
# Just build the model
model = build_sddp_model(params, data)

# Just generate visualizations from existing simulations
generate_visualizations(simulations, params, data; output_dir="custom_output/")

# Just load parameters
params = load_parameters("data/model_parameters.xlsx")
```

## Output Files

After running the optimization, the following files are generated in `output/`:

### Text Output
- `simulation_results.txt`: Detailed simulation results for each stage

### Plots
- `Investments_[Technology].png`: Investment band plots for each technology
- `Investments_Storage.png`: Storage investment band plot
- `LoadDurationCurve_Year[YYYY].png`: Load duration curves for each year
- `ViolinPlot_[Technology].png`: Production vs demand violin plots
- `ViolinPlot_Storage.png`: Storage operation violin plot
- `spaghetti_plot.html`: Interactive spaghetti plots for all variables

## Model Formulation

The model uses a two-stage structure alternating between investment and operational stages:

### Investment Stages (odd stages: 1, 3, 5, 7)
- Decision: Capacity expansion for each technology and storage
- Costs: Investment costs + fixed O&M costs

### Operational Stages (even stages: 2, 4, 6, 8)
- Decisions: Production levels, storage charge/discharge, unmet demand
- Uncertainty: Demand multipliers, natural gas prices
- Costs: Variable O&M + fuel costs + carbon costs - electricity sales revenue + unmet demand penalty

### Key Features
- **Vintage tracking**: Capacities tracked by investment year with retirement based on lifetime
- **Representative weeks**: 168-hour representative weeks with occurrence weights
- **Storage dynamics**: Hourly charge/discharge with efficiency losses and heat dissipation
- **Markovian demand**: 3-state demand multiplier process
- **Stochastic prices**: Log-normal natural gas price distribution

## Troubleshooting

### Common Issues

1. **XLSX package not found**: Run `import Pkg; Pkg.add("XLSX")`
2. **Gurobi license error**: Ensure Gurobi is installed and licensed
3. **Data files not found**: Check that `typical_weeks.csv` and electricity price files are in `data/`
4. **Module not found**: Ensure you've added `src/` to `LOAD_PATH`

## Contributing

Feel free to submit issues or pull requests for improvements.

## License

This project is available for academic and research use.

## Citation

If you use this code in your research, please cite:

```
[EEM/Journal publication details]
```

## Contact

For questions or support, please contact [koek@eeg.tuwien.ac.at].