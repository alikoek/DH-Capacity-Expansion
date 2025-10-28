"""
Parameter loading and management for the District Heating Capacity Expansion model
"""

using XLSX, DataFrames

"""
Structure to hold all model parameters
"""
struct ModelParameters
    # Model configuration
    T::Int
    T_years::Int
    discount_rate::Float64
    base_annual_demand::Float64
    salvage_fraction::Float64
    c_penalty::Float64
    elec_taxes_levies::Float64

    # Technologies
    technologies::Vector{Symbol}
    c_initial_capacity::Dict{Symbol, Float64}
    c_max_additional_capacity::Dict{Symbol, Float64}
    c_investment_cost::Dict{Symbol, Float64}
    c_opex_fixed::Dict{Symbol, Float64}
    c_opex_var::Dict{Symbol, Float64}
    c_efficiency_th::Dict{Symbol, Float64}
    c_efficiency_el::Dict{Symbol, Float64}
    c_energy_carrier::Dict{Symbol, Symbol}
    c_lifetime_new::Dict{Symbol, Int}
    c_lifetime_initial::Dict{Symbol, Int}

    # Storage
    storage_params::Dict{Symbol, Float64}

    # Energy carriers
    c_emission_fac::Dict{Symbol, Float64}
    elec_emission_factors::Dict{Int, Float64}

    # Uncertainty configurations
    energy_price_map::Dict{Int, Dict{Int, Dict{Symbol, Float64}}}
    carbon_trajectories::Dict{Int, Vector{Float64}}
    carbon_probabilities::Dict{Int, Float64}
    demand_multipliers::Vector{Float64}
    demand_probabilities::Vector{Float64}
    energy_transitions::Matrix{Float64}
    initial_energy_dist::Vector{Float64}
    use_stochastic_demand::Bool

    # Investment stages
    investment_stages::Vector{Int}
end

"""
    load_parameters(excel_path::String)

Load all model parameters from an Excel file.

# Arguments
- `excel_path::String`: Path to the Excel parameter file

# Returns
- `ModelParameters`: Structure containing all model parameters
"""
function load_parameters(excel_path::String)
    # Open Excel file
    xf = XLSX.readxlsx(excel_path)

    # Load ModelConfig sheet
    config_sheet = xf["ModelConfig"]
    config_df = DataFrame(XLSX.gettable(config_sheet))

    # Helper function to get parameter value
    get_param(name) = config_df[config_df.parameter .== name, :value][1]

    T = Int(get_param("T"))
    T_years = Int(get_param("T_years"))
    discount_rate = Float64(get_param("discount_rate"))
    base_annual_demand = Float64(get_param("base_annual_demand"))
    salvage_fraction = Float64(get_param("salvage_fraction"))
    # Convert to TSEK: SEK → TSEK (÷1000)
    c_penalty = round(Float64(get_param("c_penalty")) / 1000, digits=2)
    elec_taxes_levies = round(Float64(get_param("elec_taxes_levies")) / 1000, digits=2)

    # Demand uncertainty switch (default to false/deterministic if not in Excel)
    use_stochastic_demand = try
        Bool(get_param("use_stochastic_demand"))
    catch
        false  # Default to deterministic
    end

    # Load Technologies sheet
    tech_sheet = xf["Technologies"]
    tech_df = DataFrame(XLSX.gettable(tech_sheet))

    # Helper function to extract column name (handles bracket notation like "initial_capacity[MW_th]")
    function get_col(df, base_name)
        for col_name in names(df)
            # Extract part before '[' if bracket exists
            col_base = occursin('[', col_name) ? split(col_name, '[')[1] : col_name
            if col_base == base_name
                return df[!, col_name]
            end
        end
        error("Column starting with '$base_name' not found in dataframe")
    end

    technologies = Symbol.(tech_df.technology)
    c_initial_capacity = Dict(zip(technologies, round.(Float64.(get_col(tech_df, "initial_capacity")), digits=2)))
    c_max_additional_capacity = Dict(zip(technologies, round.(Float64.(get_col(tech_df, "max_additional_capacity")), digits=2)))
    # Convert to TSEK: MSEK → TSEK (×1000)
    c_investment_cost = Dict(zip(technologies, round.(Float64.(get_col(tech_df, "investment_cost")) .* 1000, digits=2)))
    c_opex_fixed = Dict(zip(technologies, round.(Float64.(get_col(tech_df, "fixed_om")) .* 1000, digits=2)))
    # Convert to TSEK: SEK → TSEK (÷1000)
    c_opex_var = Dict(zip(technologies, round.(Float64.(get_col(tech_df, "variable_om")) ./ 1000, digits=2)))
    c_efficiency_th = Dict(zip(technologies, round.(Float64.(tech_df.efficiency_th), digits=4)))  # Keep 4 digits for efficiencies
    c_efficiency_el = Dict(zip(technologies, round.(Float64.(tech_df.efficiency_el), digits=4)))  # Keep 4 digits for efficiencies
    c_energy_carrier = Dict(zip(technologies, Symbol.(tech_df.energy_carrier)))
    c_lifetime_new = Dict(zip(technologies, Int.(tech_df.lifetime_new)))
    c_lifetime_initial = Dict(zip(technologies, Int.(tech_df.lifetime_initial)))

    # Load Storage sheet
    stor_sheet = xf["Storage"]
    stor_df = DataFrame(XLSX.gettable(stor_sheet))
    storage_params = Dict{Symbol, Float64}()
    for row in eachrow(stor_df)
        param_name = Symbol(row.parameter)
        value = Float64(row.value)

        # Convert cost parameters from SEK to TSEK (÷1000)
        if param_name in [:capacity_cost, :fixed_om, :variable_om]
            storage_params[param_name] = round(value / 1000, digits=4)
        else
            # Rates, efficiencies, lifetime, capacities - no conversion
            digits = (param_name in [:efficiency, :loss_rate, :max_charge_rate, :max_discharge_rate]) ? 4 : 2
            storage_params[param_name] = round(value, digits=digits)
        end
    end

    # Load EnergyCarriers sheet (static emission factors for non-electricity carriers)
    carrier_sheet = xf["EnergyCarriers"]
    carrier_df = DataFrame(XLSX.gettable(carrier_sheet))

    # Extract column name for emission_factor (handles bracket notation)
    ef_col_name = nothing
    for col in names(carrier_df)
        if startswith(col, "emission_factor")
            ef_col_name = col
            break
        end
    end

    # Skip electricity (has missing/time-varying emission factor)
    c_emission_fac = Dict{Symbol, Float64}()
    for row in eachrow(carrier_df)
        carrier = Symbol(row.carrier)
        if carrier != :elec && !ismissing(row[ef_col_name])
            c_emission_fac[carrier] = round(Float64(row[ef_col_name]), digits=4)
        end
    end

    # Load time-varying electricity emission factors
    elec_ef_sheet = xf["Emission Factor for electricity"]
    elec_ef_df = DataFrame(XLSX.gettable(elec_ef_sheet))

    # Map model years to data years: 1→2023, 2→2030, 3→2040, 4→2050
    year_mapping = Dict(1 => 2023, 2 => 2030, 3 => 2040, 4 => 2050)

    # Extract column name for emission factor
    elec_ef_col_name = nothing
    for col in names(elec_ef_df)
        if startswith(lowercase(col), "emission")
            elec_ef_col_name = col
            break
        end
    end

    elec_emission_factors = Dict{Int, Float64}()
    for model_year in 1:T
        data_year = year_mapping[model_year]
        year_row = elec_ef_df[elec_ef_df.Year .== data_year, :]
        if nrow(year_row) > 0
            elec_emission_factors[model_year] = round(Float64(year_row[1, elec_ef_col_name]), digits=4)
        else
            error("Year $data_year not found in Emission Factor for electricity sheet")
        end
    end

    # Load EnergyPriceMap sheet (multi-carrier, multi-year structure)
    energy_price_sheet = xf["EnergyPriceMap"]
    energy_price_df = DataFrame(XLSX.gettable(energy_price_sheet))

    # Map data years to model years
    data_year_to_model = Dict(2023 => 1, 2030 => 2, 2040 => 3, 2050 => 4)

    # Identify carrier columns (all columns with brackets containing price units)
    carrier_cols = Symbol[]
    for col in names(energy_price_df)
        if occursin('[', col) && !in(col, ["state", "description", "year"])
            # Extract carrier name before bracket: "waste[SEK/MWh]" → :waste
            carrier_name = Symbol(split(col, '[')[1])
            push!(carrier_cols, carrier_name)
        end
    end

    # Build nested structure: state → model_year → carrier → price
    energy_price_map = Dict{Int, Dict{Int, Dict{Symbol, Float64}}}()

    for state in 1:3  # States: 1=High, 2=Medium, 3=Low
        energy_price_map[state] = Dict{Int, Dict{Symbol, Float64}}()

        for (data_year, model_year) in data_year_to_model
            # Filter rows for this state and year
            rows = energy_price_df[(energy_price_df.state .== state) .& (energy_price_df.year .== data_year), :]

            if nrow(rows) == 0
                error("No price data found for state=$state, year=$data_year")
            end

            # Extract carrier prices
            carrier_prices = Dict{Symbol, Float64}()
            for carrier in carrier_cols
                # Find column name with brackets
                col_name = nothing
                for col in names(energy_price_df)
                    if startswith(col, string(carrier) * "[")
                        col_name = col
                        break
                    end
                end

                if col_name !== nothing
                    # Convert SEK to TSEK (÷1000)
                    carrier_prices[carrier] = round(Float64(rows[1, col_name]) / 1000, digits=4)
                end
            end

            energy_price_map[state][model_year] = carrier_prices
        end
    end

    # Load CarbonTrajectories sheet
    carbon_traj_sheet = xf["CarbonTrajectories"]
    carbon_traj_df = DataFrame(XLSX.gettable(carbon_traj_sheet))

    # Helper function to find column starting with base name (handles brackets)
    function find_col(df, base_name)
        for col_name in names(df)
            col_base = occursin('[', col_name) ? split(col_name, '[')[1] : col_name
            if col_base == base_name
                return col_name
            end
        end
        error("Column starting with '$base_name' not found")
    end

    carbon_trajectories = Dict{Int, Vector{Float64}}()
    for row in eachrow(carbon_traj_df)
        scenario = Int(row.scenario)
        # Convert to TSEK: SEK → TSEK (÷1000)
        trajectory = round.([
            Float64(row[find_col(carbon_traj_df, "year_1")]) / 1000,
            Float64(row[find_col(carbon_traj_df, "year_2")]) / 1000,
            Float64(row[find_col(carbon_traj_df, "year_3")]) / 1000,
            Float64(row[find_col(carbon_traj_df, "year_4")]) / 1000
        ], digits=4)
        carbon_trajectories[scenario] = trajectory
    end

    # Load CarbonProbabilities sheet
    carbon_prob_sheet = xf["CarbonProbabilities"]
    carbon_prob_df = DataFrame(XLSX.gettable(carbon_prob_sheet))
    carbon_probabilities = Dict(zip(Int.(carbon_prob_df.scenario), round.(Float64.(carbon_prob_df.probability), digits=4)))

    # Load DemandUncertainty sheet
    demand_unc_sheet = xf["DemandUncertainty"]
    demand_unc_df = DataFrame(XLSX.gettable(demand_unc_sheet))
    demand_multipliers = round.(Float64.(demand_unc_df.multiplier), digits=4)  # 4 digits for multipliers like 0.95, 1.05
    demand_probabilities = round.(Float64.(demand_unc_df.probability), digits=4)

    # Load EnergyTransitions sheet
    energy_trans_sheet = xf["EnergyTransitions"]
    energy_trans_df = DataFrame(XLSX.gettable(energy_trans_sheet))
    energy_transitions = Matrix{Float64}(undef, 3, 3)
    for (i, row) in enumerate(eachrow(energy_trans_df))
        # Round transition probabilities to 4 decimal places
        energy_transitions[i, :] = round.([Float64(row.to_high), Float64(row.to_medium), Float64(row.to_low)], digits=4)
    end

    # Set initial energy distribution (could also be added to Excel if needed)
    initial_energy_dist = round.([0.3, 0.4, 0.3], digits=4)

    # Calculate investment stages
    investment_stages = [0; collect(1:2:(2*T-1))]

    return ModelParameters(
        T, T_years, discount_rate, base_annual_demand, salvage_fraction,
        c_penalty, elec_taxes_levies,
        technologies, c_initial_capacity, c_max_additional_capacity,
        c_investment_cost, c_opex_fixed, c_opex_var, c_efficiency_th,
        c_efficiency_el, c_energy_carrier, c_lifetime_new, c_lifetime_initial,
        storage_params, c_emission_fac, elec_emission_factors,
        energy_price_map, carbon_trajectories, carbon_probabilities,
        demand_multipliers, demand_probabilities, energy_transitions, initial_energy_dist,
        use_stochastic_demand,
        investment_stages
    )
end
