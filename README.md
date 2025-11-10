# District Heating Capacity Expansion Optimization

A modular Julia framework for optimizing capacity expansion decisions in district heating systems using Stochastic Dual Dynamic Programming (SDDP) with representative weeks and thermal storage.

## Features

- **Multi-technology optimization**: CHP, Boiler, Heat Pump, Geothermal
- **Thermal storage modeling**: With charge/discharge dynamics and heat losses
- **Representative weeks**: Efficient temporal resolution using TSAM clustering
- **Three-uncertainty stochastic optimization**:
  - **Markovian energy prices**: State-dependent transitions between High/Medium/Low price regimes
  - **Stage-wise independent demand**: Random demand multipliers at each operational stage
  - **Early system temperature scenario branching**: Two DH system temperature scenarios (Low-temp/High-temp) affecting heat pump COP
- **Risk measures**: CVaR, expectation, worst-case
- **Vintage tracking**: Tracks capacity by investment year and retirement
- **Comprehensive visualization**: Investment plots, load duration curves, violin plots, spaghetti plots
- **Fully configurable**: All uncertainties and parameters externalized to Excel

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

Edit `data/model_parameters.xlsx` to modify any model parameters (see detailed guide below).
All changes take effect immediately - no code modifications needed!

## Excel Parameter File Structure

The `model_parameters.xlsx` file contains all model parameters in separate sheets:

### Core Configuration Sheets

**ModelConfig**: Model-wide settings
- `T`: Number of model years (planning horizon)
- `T_years`: Actual years per model year
- `discount_rate`: Annual discount rate
- `base_annual_demand`: Base annual heat demand (MWh)
- `salvage_fraction`: Fraction of investment value recovered at end of horizon
- `c_penalty`: Penalty cost for unmet demand (€/MWh)

**Technologies**: Technology-specific parameters (one row per technology)
- `technology`: Technology name (CHP, Boiler, HeatPump, Geothermal)
- `initial_capacity`: Existing capacity at start (MW)
- `max_additional_capacity`: Maximum capacity that can be added (MW)
- `investment_cost`: Investment cost (€/MW)
- `fixed_om`: Fixed O&M cost (€/MW/year)
- `variable_om`: Variable O&M cost (€/MWh)
- `efficiency_th`: Thermal efficiency (fraction)
- `efficiency_el`: Electrical efficiency for CHP (fraction)
- `energy_carrier`: Primary energy carrier (NaturalGas, Electricity, Geothermal)
- `lifetime_new`: Lifetime for new investments (years)
- `lifetime_initial`: Remaining lifetime for initial capacity (years)

**Storage**: Thermal storage parameters
- `capacity_cost`: Investment cost (€/MWh)
- `max_capacity`: Maximum storage capacity (MWh)
- `efficiency`: Round-trip efficiency (fraction)
- `loss_rate`: Hourly heat loss rate (fraction/hour)
- `max_charge_rate`: Maximum charge rate relative to capacity (fraction)
- `max_discharge_rate`: Maximum discharge rate relative to capacity (fraction)
- `lifetime`: Storage system lifetime (years)

**EnergyCarriers**: Emission factors
- `carrier`: Energy carrier name
- `emission_factor`: CO₂ emissions (tCO₂/MWh)

### Uncertainty Configuration Sheets

**EnergyPriceMap**: Deterministic price mapping for each Markovian energy state
- `state`: Energy price state (1=High, 2=Medium, 3=Low)
- `description`: State description
- `price_eur_per_mwh`: Natural gas price (€/MWh)

**CarbonTrajectory**: Single net-zero carbon price trajectory
- `year_1` to `year_4`: Carbon price (€/tCO₂) for each model year

**TemperatureScenarios**: System temperature scenario definitions
- `scenario`: Scenario number (1-2)
- `description`: Scenario description (Low-temp, High-temp)
- `cop_multiplier`: COP multiplier for heat pumps (e.g., 1.1 for low-temp DH, 0.9 for high-temp DH)
- `probability`: Branching probability at stage 1 (must sum to 1.0)

**DemandUncertainty**: Stage-wise independent demand multipliers
- `multiplier`: Demand multiplier value
- `probability`: Probability at each operational stage (must sum to 1.0)
- `description`: Multiplier description

**EnergyTransitions**: Markovian transition matrix for energy price states (3×3)
- `from_state`: Current energy state (High, Medium, Low)
- `to_high`: Probability of transitioning to High state
- `to_medium`: Probability of transitioning to Medium state
- `to_low`: Probability of transitioning to Low state
- **Note**: Each row must sum to 1.0

## Configuration Examples

### Example 1: Testing Different System Temperature Scenarios

To analyze sensitivity to DH system temperature levels, modify the **TemperatureScenarios** sheet:

**Scenario: High system temperature variability**
```
scenario | description | cop_multiplier | probability
1        | Low-temp    | 1.15           | 0.4
2        | High-temp   | 0.85           | 0.6
```

Or modify the **CarbonTrajectory** sheet for different carbon price assumptions:

**Aggressive carbon pricing**
```
year_1 | year_2 | year_3 | year_4
150    | 200    | 250    | 300
```

### Example 2: Modeling High Energy Price Volatility

To increase energy price uncertainty, modify the **EnergyTransitions** sheet to reduce persistence:

**Low persistence (high volatility)**
```
from_state | to_high | to_medium | to_low
High       | 0.4     | 0.4       | 0.2     (was 0.6, 0.3, 0.1)
Medium     | 0.3     | 0.4       | 0.3     (was 0.2, 0.6, 0.2)
Low        | 0.2     | 0.4       | 0.4     (was 0.1, 0.3, 0.6)
```

And update **EnergyPriceMap** for wider price ranges:
```
state | description | price_eur_per_mwh
1     | High        | 60.0               (was 45.0)
2     | Medium      | 30.0               (unchanged)
3     | Low         | 10.0               (was 20.0)
```

### Example 3: Modeling Demand Growth Uncertainty

To test scenarios with higher demand variability, modify **DemandUncertainty**:

**High demand variability**
```
multiplier | probability | description
1.3        | 0.15        | Very high demand
1.1        | 0.25        | High demand
1.0        | 0.3         | Normal demand
0.9        | 0.2         | Low demand
0.7        | 0.1         | Very low demand
```

**Note**: You can add up to ~5 multipliers. More multipliers increase computational time.

### Example 4: Technology Cost Sensitivity

To analyze impact of heat pump cost reductions, modify **Technologies** sheet:

**Original**
```
technology | investment_cost | fixed_om | variable_om
HeatPump   | 800000         | 8000     | 2.0
```

**Cost reduction scenario**
```
technology | investment_cost | fixed_om | variable_om
HeatPump   | 500000         | 5000     | 1.5
```

### Example 5: Longer Planning Horizon

To extend from 4 to 6 model years, modify **ModelConfig**:

```
parameter | value
T         | 6     (was 4)
```

**Important**: You must also extend **CarbonTrajectory** to include year_5 and year_6:
```
year_1 | year_2 | year_3 | year_4 | year_5 | year_6
100    | 150    | 200    | 250    | 300    | 350
```

### Example 6: Risk-Averse vs Risk-Neutral Planning

Risk aversion is controlled in code (not Excel). In `run_capacity_expansion.jl`:

**Risk-neutral (expected value)**
```julia
simulations = run_simulation(model, params, data;
                            risk_measure=SDDP.Expectation(),
                            iteration_limit=100)
```

**Risk-averse (CVaR 95%)**
```julia
simulations = run_simulation(model, params, data;
                            risk_measure=SDDP.AVaR(0.95),
                            iteration_limit=100)
```

**Worst-case robust**
```julia
simulations = run_simulation(model, params, data;
                            risk_measure=SDDP.WorstCase(),
                            iteration_limit=100)
```

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
- **Decision**: Capacity expansion for each technology and storage
- **Costs**: Investment costs + fixed O&M costs
- **Timing**: Beginning of each model year (years 0, 10, 20, 30 for T=4)

### Operational Stages (even stages: 2, 4, 6, 8)
- **Decisions**: Production levels, storage charge/discharge, unmet demand
- **Uncertainty**: Temperature scenarios, Markovian energy prices, stage-wise independent demand
- **Costs**: Variable O&M + fuel costs + carbon costs - electricity sales revenue + unmet demand penalty
- **Temporal resolution**: Representative weeks with hourly granularity

### Uncertainty Structure

The model incorporates three types of uncertainty with different probabilistic structures:

#### 1. Early System Temperature Scenario Branching
- **Scenarios**: Low-temp DH (COP multiplier = 1.1), High-temp DH (COP multiplier = 0.9)
- **Probabilities**: [50% Low-temp, 50% High-temp] (configurable)
- **Timing**: Branching occurs at stage 1 (before first investment decision)
- **Structure**: Early scenario tree - system temperature regime is revealed before any investments
- **Impact**: Affects heat pump coefficient of performance (COP) throughout entire planning horizon
- **Interpretation**: Long-term district heating network temperature regime uncertainty (e.g., transition to low-temperature district heating) that influences initial investment decisions

**Key feature**: This is **early branching** - the model knows the DH system temperature regime before making first investments, allowing temperature-aware capacity planning.

**Note**: Temperature scenarios refer to **district heating system supply/return temperatures** (e.g., 60/40°C vs 90/70°C), not outdoor ambient temperature. Lower system temperatures enable higher heat pump efficiency.

#### 2. Markovian Energy Prices (State-Dependent)
- **States**: High (45 €/MWh), Medium (30 €/MWh), Low (20 €/MWh)
- **Transitions**: Between investment → operational stages within each temperature scenario
- **Structure**: First-order Markov chain with persistence (diagonal elements = 0.6)
- **Initial distribution**: [30% High, 40% Medium, 30% Low]
- **Interpretation**: Energy price regimes that persist but can transition over time

**Example transition**: If energy prices are High in year 1, there's a 60% chance they stay High in year 2, 30% chance they drop to Medium, and 10% chance they drop to Low.

**Note**: Energy price transitions occur independently within each temperature scenario (6 total energy×temperature states).

#### 3. Stage-wise Independent Demand (i.i.d.)
- **Multipliers**: [1.1, 1.0, 0.9] (High, Normal, Low demand)
- **Probabilities**: [0.2, 0.5, 0.3]
- **Timing**: Sampled at each operational stage
- **Structure**: Independent across stages (no memory)
- **State variable**: Cumulative demand tracked through multiplicative state variable
- **Interpretation**: Short-term demand fluctuations (weather, economic activity)

**Note**: First model year (stage 2) is deterministic with multiplier = 1.0

### Node Structure

For a 4-year horizon (T=4), the model has **39 nodes**:

| Stage | Type | Description | Nodes | Scenario Paths |
|-------|------|-------------|-------|----------------|
| 1 | Investment | Initial (before temp branching) | 1 | 1 |
| 2 | Operational | Temperature branching | 2 | 2 |
| 3 | Investment | Energy branching | 6 | 6 |
| 4 | Operational | Energy transitions | 6 | 18 |
| 5 | Investment | - | 6 | 18 |
| 6 | Operational | - | 6 | 54 |
| 7 | Investment | - | 6 | 54 |
| 8 | Operational | - | 6 | 162 |

**Total unique paths**: ~486 scenarios (considering demand uncertainty at operational stages 4-8)

**Key difference from standard SDDP**: System temperature branching occurs at stage 1, allowing the first investment decision to be temperature-aware. This "early branching" structure enables proactive adaptation to long-term DH network temperature regime scenarios.

### Key Features
- **Vintage tracking**: Capacities tracked by investment year with retirement based on lifetime
- **Representative weeks**: 168-hour representative weeks with occurrence weights
- **Storage dynamics**: Hourly charge/discharge with efficiency losses and heat dissipation
- **Cumulative demand tracking**: Multiplicative state variable compounds demand multipliers over time
- **Deterministic first year**: No uncertainty in initial operational period for stability

## Testing and Validation

### Quick Validation Checklist

After making changes to parameters, verify your model is well-configured:

1. **Check node count**: For T=4, should have 39 nodes
   - Look for: `nodes : 39` in training output

2. **Check scenario count**: For T=4 with default settings, should have ~486 scenarios
   - Look for: `scenarios : 4.86000e+02` in training output

3. **Verify probabilities sum to 1.0**:
   - Each row in **EnergyTransitions** must sum to 1.0
   - **TemperatureScenarios** probabilities must sum to 1.0
   - **DemandUncertainty** probabilities must sum to 1.0

4. **Check for unmet demand**:
   - Some unmet demand is normal during uncertainty
   - Large persistent unmet demand may indicate:
     - Insufficient `max_additional_capacity` in Technologies
     - Too high `base_annual_demand` in ModelConfig
     - Technology lifetimes causing capacity to retire too early

5. **Verify numerical stability**:
   - Check for warning: `numerical stability issues detected`
   - If present, consider rescaling large coefficients (investment costs, demand levels)

### Interpreting Results

**Simulation output file** (`output/simulation_results.txt`):
- Shows one sample path through the scenario tree
- Investment decisions at each investment stage
- Demand multiplier state variable (cumulative effect)
- Production, storage operation, and unmet demand per stage

**Visualizations**:
- **Investment plots**: Show capacity expansion bands across scenarios
  - Wide bands = high uncertainty impact
  - Narrow bands = robust decisions across scenarios
- **Load duration curves**: Demand vs production duration
- **Violin plots**: Distribution of production across scenarios
- **Spaghetti plots**: Individual scenario paths for detailed analysis

### Recommended Testing Sequence

#### Phase 1: Baseline Validation
```bash
# Run with default parameters
julia examples/run_capacity_expansion.jl
```
**Check**: Does it complete? Are results reasonable?

#### Phase 2: Sensitivity to System Temperature Scenarios
```
1. Adjust COP multipliers in TemperatureScenarios (e.g., 1.15 low-temp, 0.85 high-temp)
2. Re-run and compare investment decisions
3. Expected: Wider temperature range → more flexible/robust capacity mix
```

#### Phase 3: Sensitivity to Energy Price Volatility
```
1. Reduce diagonal elements in EnergyTransitions (e.g., 0.6 → 0.4)
2. Re-run and compare CVaR vs Expectation risk measures
3. Expected: CVaR leads to more conservative/flexible investments
```

#### Phase 4: Longer Horizon
```
1. Increase T from 4 to 6 in ModelConfig
2. Extend CarbonTrajectory with year_5 and year_6
3. Re-run (expect longer solve time)
4. Expected: More nuanced long-term investment strategy
```

### Performance Tuning

**Training iterations**: Balance solution quality vs computation time
- `iteration_limit=10`: Fast testing (~30 seconds)
- `iteration_limit=100`: Good quality (~3-5 minutes)
- `iteration_limit=500`: High quality (~15-20 minutes)

**Simulation count**: More simulations = better statistical estimates
- `n_simulations=100`: Quick overview
- `n_simulations=400`: Default (good balance)
- `n_simulations=1000`: High precision for final results

**Convergence check**: Look for stabilizing bound in training output
```
iteration    simulation      bound        time (s)
    10      2.1e+12        1.5e+08          2.1
    20      1.9e+12        1.4e+08          4.2
    ...
    90      1.6e+12        1.2e+08         18.9
   100      1.6e+12        1.2e+08         21.0  ← Bound has stabilized
```

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