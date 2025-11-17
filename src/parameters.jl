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
    c_initial_capacity::Dict{Symbol,Float64}
    c_max_additional_capacity::Dict{Symbol,Float64}
    c_investment_cost::Dict{Symbol,Float64}
    c_opex_fixed::Dict{Symbol,Float64}
    c_opex_var::Dict{Symbol,Float64}
    c_efficiency_th::Dict{Symbol,Float64}
    c_efficiency_el::Dict{Symbol,Float64}
    c_energy_carrier::Dict{Symbol,Symbol}
    c_lifetime_new::Dict{Symbol,Int}
    c_lifetime_initial::Dict{Symbol,Int}
    c_capacity_limits::Dict{Symbol,Vector{Float64}}

    # Storage
    storage_params::Dict{Symbol,Float64}
    storage_capacity_limits::Vector{Float64}

    # Energy carriers
    c_emission_fac::Dict{Symbol,Float64}
    elec_emission_factors::Dict{Int,Float64}

    # Uncertainty configurations
    energy_price_map::Dict{Int,Dict{Int,Dict{Symbol,Float64}}}
    carbon_trajectory::Vector{Float64}  # Single net-zero trajectory
    temp_scenarios::Vector{Symbol}
    temp_cop_multipliers::Dict{Symbol,Float64}
    temp_scenario_probabilities::Dict{Int,Float64}
    demand_multipliers::Vector{Float64}
    demand_probabilities::Vector{Float64}
    energy_transitions::Matrix{Float64}
    initial_energy_dist::Vector{Float64}
    use_stochastic_demand::Bool

    # Investment stages
    investment_stages::Vector{Int}

    # Existing capacity retirement and time-varying parameters
    c_existing_capacity_schedule::Dict{Symbol,Vector{Float64}}
    waste_chp_efficiency_schedule::Vector{Float64}
    waste_availability::Vector{Float64}

    # Extreme events (optional)
    enable_extreme_events::Bool
    apply_to_year::Int
    extreme_events::Union{Nothing,DataFrame}
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
    get_param(name) = config_df[config_df.parameter.==name, :value][1]

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

    # Load CapacityLimits sheet
    cap_limit_sheet = xf["CapacityLimits"]
    cap_limit_df = DataFrame(XLSX.gettable(cap_limit_sheet))

    # Helper function to get column name for a year (handles bracket notation)
    function get_year_col(df, base_name)
        for col_name in names(df)
            col_base = occursin('[', col_name) ? split(col_name, '[')[1] : col_name
            if col_base == base_name
                return col_name
            end
        end
        error("Column starting with '$base_name' not found in CapacityLimits sheet")
    end

    # Build capacity limits dictionary: tech -> [year_1, year_2, ..., year_T]
    # Note: -1 in Excel indicates "no limit" and is converted to Inf
    c_capacity_limits = Dict{Symbol,Vector{Float64}}()
    storage_capacity_limits = Float64[]

    for row in eachrow(cap_limit_df)
        tech_or_storage = String(row.technology)
        limits = Float64[]

        for year in 1:T
            col_name = get_year_col(cap_limit_df, "year_$(year)")
            limit_value = Float64(row[col_name])

            # Convert -1 to Inf (meaning no limit)
            if limit_value < 0
                push!(limits, Inf)
            else
                push!(limits, round(limit_value, digits=2))
            end
        end

        # Check if this is storage or a technology
        if tech_or_storage == "Storage"
            storage_capacity_limits = limits
        else
            c_capacity_limits[Symbol(tech_or_storage)] = limits
        end
    end

    # Ensure storage limits were found, otherwise default to no limit
    if isempty(storage_capacity_limits)
        storage_capacity_limits = fill(Inf, T)
    end

    # Load Storage sheet
    stor_sheet = xf["Storage"]
    stor_df = DataFrame(XLSX.gettable(stor_sheet))
    storage_params = Dict{Symbol,Float64}()
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
    c_emission_fac = Dict{Symbol,Float64}()
    for row in eachrow(carrier_df)
        carrier = Symbol(row.carrier)
        if carrier != :elec && !ismissing(row[ef_col_name])
            c_emission_fac[carrier] = round(Float64(row[ef_col_name]), digits=4)
        end
    end

    # Load time-varying electricity emission factors
    elec_ef_sheet = xf["Emission_Factor_for_electricity"]
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

    elec_emission_factors = Dict{Int,Float64}()
    for model_year in 1:T
        data_year = year_mapping[model_year]
        year_row = elec_ef_df[elec_ef_df.Year.==data_year, :]
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
    energy_price_map = Dict{Int,Dict{Int,Dict{Symbol,Float64}}}()

    for state in 1:3  # States: 1=High, 2=Medium, 3=Low
        energy_price_map[state] = Dict{Int,Dict{Symbol,Float64}}()

        for (data_year, model_year) in data_year_to_model
            # Filter rows for this state and year
            rows = energy_price_df[(energy_price_df.state.==state).&(energy_price_df.year.==data_year), :]

            if nrow(rows) == 0
                error("No price data found for state=$state, year=$data_year")
            end

            # Extract carrier prices
            carrier_prices = Dict{Symbol,Float64}()
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

    # Load CarbonTrajectory sheet
    carbon_traj_sheet = xf["CarbonTrajectory"]
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

    # Load single carbon trajectory (net-zero path)
    # Convert to TSEK: SEK → TSEK (÷1000)
    carbon_trajectory = round.([
            Float64(carbon_traj_df[1, find_col(carbon_traj_df, "year_1")]) / 1000,
            Float64(carbon_traj_df[1, find_col(carbon_traj_df, "year_2")]) / 1000,
            Float64(carbon_traj_df[1, find_col(carbon_traj_df, "year_3")]) / 1000,
            Float64(carbon_traj_df[1, find_col(carbon_traj_df, "year_4")]) / 1000
        ], digits=4)

    # Load TemperatureScenarios sheet
    temp_scen_sheet = xf["TemperatureScenarios"]
    temp_scen_df = DataFrame(XLSX.gettable(temp_scen_sheet))
    temp_scenarios = [Symbol(row.description) for row in eachrow(temp_scen_df)]
    temp_cop_multipliers = Dict(zip(temp_scenarios, round.(Float64.(temp_scen_df.cop_multiplier), digits=4)))

    # Load TemperatureProbabilities sheet
    temp_prob_sheet = xf["TemperatureProbabilities"]
    temp_prob_df = DataFrame(XLSX.gettable(temp_prob_sheet))
    temp_scenario_probabilities = Dict(zip(Int.(temp_prob_df.scenario), round.(Float64.(temp_prob_df.probability), digits=4)))

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

    # Calculate investment stages (excluding stage 0 - no longer used with retirement schedule)
    investment_stages = collect(1:2:(2*T-1))

    # Load existing capacity retirement schedule
    c_existing_capacity_schedule = Dict{Symbol,Vector{Float64}}()
    if "ExistingCapacitySchedule" in XLSX.sheetnames(xf)
        schedule_df = DataFrame(XLSX.gettable(xf["ExistingCapacitySchedule"]))
        for row in eachrow(schedule_df)
            tech = Symbol(row.technology)
            schedule = [Float64(row[Symbol("year_$i")]) for i in 1:T]
            c_existing_capacity_schedule[tech] = schedule
        end
        println("  Loaded retirement schedules for $(length(c_existing_capacity_schedule)) technologies")
    else
        println("  WARNING: 'ExistingCapacitySchedule' sheet not found - using zero existing capacity")
    end

    # Load Waste_CHP time-varying efficiency
    waste_chp_efficiency_schedule = zeros(Float64, T)
    if "WasteChpEfficiency" in XLSX.sheetnames(xf)
        eff_df = DataFrame(XLSX.gettable(xf["WasteChpEfficiency"]))
        if nrow(eff_df) > 0
            for i in 1:T
                waste_chp_efficiency_schedule[i] = Float64(eff_df[1, Symbol("year_$i")])
            end
            println("  Loaded Waste_CHP time-varying efficiency")
        end
    else
        println("  WARNING: 'WasteChpEfficiency' sheet not found - Waste_CHP will use static efficiency")
    end

    # Load waste fuel availability constraint
    waste_availability = zeros(Float64, T)
    if "WasteAvailability" in XLSX.sheetnames(xf)
        avail_df = DataFrame(XLSX.gettable(xf["WasteAvailability"]))
        if nrow(avail_df) > 0
            for i in 1:T
                waste_availability[i] = Float64(avail_df[1, Symbol("year_$i")])
            end
            println("  Loaded waste availability constraints")
        end
    else
        println("  WARNING: 'WasteAvailability' sheet not found - no waste fuel constraint will be applied")
        # Set to large values so constraint is non-binding
        waste_availability = fill(1e6, T)
    end

    # Load Extreme Events (optional)
    enable_extreme_events = false
    apply_to_year = 2  # Default to Year 2
    extreme_events = nothing

    if "ExtremeEventControl" in XLSX.sheetnames(xf)
        control_sheet = xf["ExtremeEventControl"]
        control_df = DataFrame(XLSX.gettable(control_sheet))

        # Load enable flag
        enable_rows = control_df[control_df.parameter.=="enable_extreme_events", :value]
        if !isempty(enable_rows)
            enable_extreme_events = Bool(enable_rows[1])
        end

        # Load target year
        year_rows = control_df[control_df.parameter.=="apply_to_year", :value]
        if !isempty(year_rows)
            apply_to_year = Int(year_rows[1])
        end

        if enable_extreme_events && "ExtremeEvents" in XLSX.sheetnames(xf)
            events_sheet = xf["ExtremeEvents"]
            extreme_events = DataFrame(XLSX.gettable(events_sheet))

            # Validate required columns exist
            required_cols = [:probability, :demand_multiplier, :elec_price_multiplier, :dc_availability]
            missing_cols = filter(col -> !(col in propertynames(extreme_events)), required_cols)
            if !isempty(missing_cols)
                error("ExtremeEvents sheet missing required columns: $missing_cols")
            end

            # Convert string columns to Float64 (handles decimal separator issues)
            # Wrap conversion with clearer error reporting for malformed entries
            function _parse_extreme_event_value(val, row_idx, col_name)
                if val isa Number
                    return Float64(val)
                end

                normalized = replace(string(val), "," => ".")
                try
                    return parse(Float64, normalized)
                catch err
                    error("Malformed value in 'ExtremeEvents' sheet at row=$(row_idx), column=$(col_name): '$(val)'. Parsing error: $(err.msg)")
                end
            end

            for col in [:probability, :demand_multiplier, :elec_price_multiplier, :dc_availability]
                parsed_values = Vector{Float64}(undef, nrow(extreme_events))
                for (idx, val) in enumerate(extreme_events[!, col])
                    parsed_values[idx] = _parse_extreme_event_value(val, idx, col)
                end
                extreme_events[!, col] = parsed_values
            end

            # Validate probabilities sum to 1.0 after successful parsing
            prob_sum = sum(extreme_events.probability)
            if abs(prob_sum - 1.0) > 0.001
                error("ExtremeEvents probabilities sum to $prob_sum, must sum to 1.0")
            end

            println("  Extreme events ENABLED:")
            println("    Applied to Year $apply_to_year (stage $(apply_to_year * 2))")
            println("    Number of scenarios: $(nrow(extreme_events))")
            for row in eachrow(extreme_events)
                println("      - $(row.scenario): p=$(row.probability)")
            end
        else
            println("  Extreme events configuration found but DISABLED or missing ExtremeEvents sheet")
        end
    else
        println("  No extreme event configuration found (optional)")
    end

    return ModelParameters(
        T, T_years, discount_rate, base_annual_demand, salvage_fraction,
        c_penalty, elec_taxes_levies,
        technologies, c_initial_capacity, c_max_additional_capacity,
        c_investment_cost, c_opex_fixed, c_opex_var, c_efficiency_th,
        c_efficiency_el, c_energy_carrier, c_lifetime_new, c_lifetime_initial,
        c_capacity_limits,
        storage_params, storage_capacity_limits, c_emission_fac, elec_emission_factors,
        energy_price_map, carbon_trajectory, temp_scenarios, temp_cop_multipliers, temp_scenario_probabilities,
        demand_multipliers, demand_probabilities, energy_transitions, initial_energy_dist,
        use_stochastic_demand,
        investment_stages,
        c_existing_capacity_schedule, waste_chp_efficiency_schedule, waste_availability,
        enable_extreme_events, apply_to_year, extreme_events
    )
end
