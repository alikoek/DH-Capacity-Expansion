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
    n_typical_weeks::Int  # Number of typical weeks to use (1-6)
    calendar_years::Vector{Int}  # Mapping: model_year -> calendar_year (e.g., [2023, 2030, 2035, ...])

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
    temp_demand_multipliers::Dict{Symbol,Vector{Float64}}  # Year-varying demand multiplier per temperature scenario
    temp_scenario_probabilities::Dict{Int,Float64}
    energy_transitions::Matrix{Float64}
    initial_energy_dist::Vector{Float64}

    # Heat pump COP trajectories (temperature-scenario and technology-specific, year-varying)
    # Structure: temp_scenario -> tech -> calendar_year -> COP
    heatpump_cop_trajectories::Dict{Symbol,Dict{Symbol,Dict{Int,Float64}}}

    # Investment stages
    investment_stages::Vector{Int}

    # Existing capacity retirement and time-varying parameters
    c_existing_capacity_schedule::Dict{Symbol,Vector{Float64}}
    waste_chp_efficiency_schedule::Vector{Float64}
    waste_emission_factor_schedule::Vector{Float64}  # Time-varying waste EF linked to efficiency
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

    # Load n_typical_weeks (optional, default to 6)
    n_typical_weeks_rows = config_df[config_df.parameter.=="n_typical_weeks", :value]
    n_typical_weeks = isempty(n_typical_weeks_rows) ? 6 : Int(n_typical_weeks_rows[1])

    # Helper function to calculate calendar year for a model year
    # Year 1 → 2023, Year 2 → 2030, Year 3+ → 2030 + (m-2)*T_years
    function model_year_to_calendar(m::Int)
        if m == 1
            return 2023
        elseif m == 2
            return 2030
        else
            return 2030 + (m - 2) * T_years
        end
    end

    # Pre-calculate calendar years for all model years
    calendar_years = [model_year_to_calendar(m) for m in 1:T]
    println("  Calendar year mapping: ", join(["$m → $(calendar_years[m])" for m in 1:T], ", "))

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
    # Scale lifetime_new based on T_years (Excel values assume T_years=10)
    # E.g., lifetime=3 with T_years=10 → 30 actual years
    #       lifetime=6 with T_years=5  → 30 actual years (scaled by 10/T_years)
    lifetime_scale = div(10, T_years)
    c_lifetime_new = Dict(zip(technologies, Int.(tech_df.lifetime_new) .* lifetime_scale))
    if lifetime_scale != 1
        println("  Scaled lifetime_new by $lifetime_scale (T_years=$T_years)")
    end

    # Load CapacityLimits sheet
    cap_limit_sheet = xf["CapacityLimits"]
    cap_limit_df = DataFrame(XLSX.gettable(cap_limit_sheet))

    # Helper function to get column name for a year (handles bracket notation with units)
    function get_year_col(df, base_name)
        for col_name in names(df)
            col_base = occursin('[', col_name) ? strip(split(col_name, '[')[1]) : col_name
            if col_base == base_name
                return col_name
            end
        end
        error("Column starting with '$base_name' not found")
    end

    # Helper function to find column starting with base name (handles units in brackets/parentheses)
    function find_col_startswith(df, base_name)
        for col_name in names(df)
            col_lower = lowercase(string(col_name))
            base_lower = lowercase(base_name)
            # Check if column starts with base_name (ignoring units in brackets)
            if startswith(col_lower, base_lower)
                return Symbol(col_name)
            end
        end
        error("Column starting with '$base_name' not found")
    end

    # Helper to get value from row with flexible column name matching
    function get_row_value(row, base_name)
        col_sym = find_col_startswith(parent(row), base_name)
        return row[col_sym]
    end

    # Build capacity limits dictionary: tech -> [year_1, year_2, ..., year_T]
    # Note: -1 in Excel indicates "no limit" and is converted to Inf
    c_capacity_limits = Dict{Symbol,Vector{Float64}}()
    storage_capacity_limits = Float64[]

    for row in eachrow(cap_limit_df)
        tech_or_storage = String(get_row_value(row, "technology"))
        limits = Float64[]

        for year in 1:T
            cal_year = calendar_years[year]
            col_name = get_year_col(cap_limit_df, string(cal_year))
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
        elseif param_name == :lifetime
            # Scale lifetime based on T_years (Excel values assume T_years=10)
            storage_params[param_name] = round(value * lifetime_scale, digits=2)
        else
            # Rates, efficiencies, capacities - no conversion
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

    # Map model years to data years (dynamic based on T_years)
    # First year is always 2023, second is 2030, then 2030 + (m-2)*T_years
    year_mapping = Dict{Int,Int}(1 => 2023, 2 => 2030)
    for m in 3:T
        year_mapping[m] = 2030 + (m - 2) * T_years
    end

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

    # Map data years to model years (dynamic based on T_years)
    # First year is always 2023 (base year with data)
    # Second year is always 2030
    # Subsequent years: 2030 + (m-2)*T_years
    # T_years=5:  2023, 2030, 2035, 2040
    # T_years=10: 2023, 2030, 2040, 2050
    data_year_to_model = Dict{Int,Int}(2023 => 1, 2030 => 2)
    for m in 3:T
        data_year_to_model[2030 + (m - 2) * T_years] = m
    end

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
            Float64(carbon_traj_df[1, find_col(carbon_traj_df, string(calendar_years[m]))]) / 1000
            for m in 1:T
        ], digits=4)

    # Load TemperatureScenarios sheet
    temp_scen_sheet = xf["TemperatureScenarios"]
    temp_scen_df = DataFrame(XLSX.gettable(temp_scen_sheet))
    # Get description column (may have units in name like "description (demand_multiplier [-])")
    desc_col = find_col_startswith(temp_scen_df, "description")
    temp_scenarios = [Symbol(row[desc_col]) for row in eachrow(temp_scen_df)]

    # Helper function to parse values (handle comma decimal separator)
    parse_temp_value(val) = begin
        if val isa Number
            return Float64(val)
        else
            normalized = replace(string(val), "," => ".")
            return parse(Float64, normalized)
        end
    end

    # Check for calendar year columns for demand multipliers
    calendar_year_cols = [string(calendar_years[m]) for m in 1:T]
    has_calendar_year_cols = all(col in names(temp_scen_df) for col in calendar_year_cols)

    # Also check for legacy year_i_demand format
    legacy_year_cols = ["year_$(i)_demand" for i in 1:T]
    has_legacy_cols = all(col in names(temp_scen_df) for col in legacy_year_cols)

    if has_calendar_year_cols
        # Load year-varying demand multipliers from calendar year columns
        temp_demand_multipliers = Dict{Symbol,Vector{Float64}}()
        for row in eachrow(temp_scen_df)
            scenario = Symbol(row[desc_col])
            demands = [parse_temp_value(row[col]) for col in calendar_year_cols]
            temp_demand_multipliers[scenario] = round.(demands, digits=4)
        end
        println("  Loaded year-varying temperature demand multipliers:")
        for (scen, demands) in temp_demand_multipliers
            println("    $scen: $(demands)")
        end
    elseif has_legacy_cols
        # Load from legacy year_i_demand format
        temp_demand_multipliers = Dict{Symbol,Vector{Float64}}()
        for row in eachrow(temp_scen_df)
            scenario = Symbol(row[desc_col])
            demands = [parse_temp_value(row[col]) for col in legacy_year_cols]
            temp_demand_multipliers[scenario] = round.(demands, digits=4)
        end
        println("  Loaded year-varying temperature demand multipliers (legacy format):")
        for (scen, demands) in temp_demand_multipliers
            println("    $scen: $(demands)")
        end
    elseif "demand_multiplier" in names(temp_scen_df)
        # Backward compatibility: static multiplier replicated across years
        temp_demand_multipliers = Dict{Symbol,Vector{Float64}}()
        for row in eachrow(temp_scen_df)
            scenario = Symbol(row[desc_col])
            value = parse_temp_value(row.demand_multiplier)
            temp_demand_multipliers[scenario] = fill(round(value, digits=4), T)
        end
        println("  Loaded static temperature demand multipliers (replicated across $T years):")
        for (scen, demands) in temp_demand_multipliers
            println("    $scen: $(demands)")
        end
    else
        # Default: 1.0 for all years and scenarios
        temp_demand_multipliers = Dict(scen => fill(1.0, T) for scen in temp_scenarios)
        println("  No demand multiplier columns - using default 1.0 for all years")
    end

    # Load TemperatureProbabilities sheet
    temp_prob_sheet = xf["TemperatureProbabilities"]
    temp_prob_df = DataFrame(XLSX.gettable(temp_prob_sheet))
    temp_scenario_probabilities = Dict(zip(Int.(temp_prob_df.scenario), round.(Float64.(temp_prob_df.probability), digits=4)))

    # Load EnergyTransitions sheet
    energy_trans_sheet = xf["EnergyTransitions"]
    energy_trans_df = DataFrame(XLSX.gettable(energy_trans_sheet))
    energy_transitions = Matrix{Float64}(undef, 3, 3)
    for (i, row) in enumerate(eachrow(energy_trans_df))
        # Round transition probabilities to 4 decimal places
        energy_transitions[i, :] = round.([Float64(row.to_high), Float64(row.to_medium), Float64(row.to_low)], digits=4)
    end

    # Set initial energy distribution: start from Medium state (row 2 of transition matrix)
    # This ensures consistency with the Markov chain dynamics
    initial_energy_dist = round.(energy_transitions[2, :], digits=4)  # [0.05, 0.9, 0.05] for Medium start

    # Calculate investment stages (excluding stage 0 - no longer used with retirement schedule)
    investment_stages = collect(1:2:(2*T-1))

    # Load existing capacity retirement schedule
    c_existing_capacity_schedule = Dict{Symbol,Vector{Float64}}()
    if "ExistingCapacitySchedule" in XLSX.sheetnames(xf)
        schedule_df = DataFrame(XLSX.gettable(xf["ExistingCapacitySchedule"]))
        tech_col = find_col_startswith(schedule_df, "technology")
        for row in eachrow(schedule_df)
            tech = Symbol(row[tech_col])
            schedule = [Float64(row[Symbol(string(calendar_years[i]))]) for i in 1:T]
            c_existing_capacity_schedule[tech] = schedule
        end
        println("  Loaded retirement schedules for $(length(c_existing_capacity_schedule)) technologies")
    else
        println("  WARNING: 'ExistingCapacitySchedule' sheet not found - using zero existing capacity")
    end

    # Load Waste_CHP time-varying efficiency and emission factors from same sheet
    # Sheet structure: parameter | 2023 | 2030 | 2035 | ... with rows for efficiency and emission_factor
    # Row names include units: "efficiency [-]" and "emission_factor [tCO2/MWh]"
    waste_chp_efficiency_schedule = zeros(Float64, T)
    waste_emission_factor_schedule = zeros(Float64, T)

    if "WasteChpEfficiency" in XLSX.sheetnames(xf)
        waste_df = DataFrame(XLSX.gettable(xf["WasteChpEfficiency"]))

        # Load efficiency (row where parameter starts with "efficiency")
        eff_row = findfirst(row -> startswith(lowercase(string(row.parameter)), "efficiency"), eachrow(waste_df))
        if eff_row !== nothing
            for i in 1:T
                waste_chp_efficiency_schedule[i] = Float64(waste_df[eff_row, Symbol(string(calendar_years[i]))])
            end
            println("  Loaded Waste_CHP time-varying efficiency: ", round.(waste_chp_efficiency_schedule, digits=2))
        else
            println("  WARNING: 'efficiency' row not found in WasteChpEfficiency sheet")
        end

        # Load emission factor (row where parameter starts with "emission_factor")
        # Values in Excel are already in tCO2/MWh
        ef_row = findfirst(row -> startswith(lowercase(string(row.parameter)), "emission_factor"), eachrow(waste_df))
        if ef_row !== nothing
            for i in 1:T
                waste_emission_factor_schedule[i] = Float64(waste_df[ef_row, Symbol(string(calendar_years[i]))])
            end
            println("  Loaded waste emission factors (tCO2/MWh): ", round.(waste_emission_factor_schedule, digits=3))
        else
            println("  WARNING: 'emission_factor' row not found in WasteChpEfficiency sheet")
            # Fallback to static emission factor
            base_ef = haskey(c_emission_fac, :waste) ? c_emission_fac[:waste] : 0.347
            waste_emission_factor_schedule = fill(base_ef, T)
            println("  Using static waste emission factor: $base_ef tCO2/MWh")
        end
    else
        println("  WARNING: 'WasteChpEfficiency' sheet not found - using default values")
        waste_chp_efficiency_schedule = fill(0.9, T)
        base_ef = haskey(c_emission_fac, :waste) ? c_emission_fac[:waste] : 0.347
        waste_emission_factor_schedule = fill(base_ef, T)
    end

    # Load waste fuel availability constraint
    # Excel values are in GWh, model uses MWh → multiply by 1000
    waste_availability = zeros(Float64, T)
    if "WasteAvailability" in XLSX.sheetnames(xf)
        avail_df = DataFrame(XLSX.gettable(xf["WasteAvailability"]))
        if nrow(avail_df) > 0
            for i in 1:T
                waste_availability[i] = Float64(avail_df[1, Symbol(string(calendar_years[i]))]) * 1000  # GWh → MWh
            end
            println("  Loaded waste availability constraints (converted GWh → MWh)")
        end
    else
        println("  WARNING: 'WasteAvailability' sheet not found - no waste fuel constraint will be applied")
        # Set to large values so constraint is non-binding
        waste_availability = fill(1e9, T)  # 1e9 MWh = very large
    end

    # Load HeatPumpCOP sheet (temperature-scenario, technology-specific, year-varying COP values)
    # Structure: temp_scenario -> tech -> calendar_year -> COP
    heatpump_cop_trajectories = Dict{Symbol,Dict{Symbol,Dict{Int,Float64}}}()

    if "HeatPumpCOP" in XLSX.sheetnames(xf)
        cop_df = DataFrame(XLSX.gettable(xf["HeatPumpCOP"]))
        cop_tech_col = find_col_startswith(cop_df, "technology")

        # Row 1 in Excel = "High Temperature" = High-temp DH scenario
        # Row 2 in Excel = "Low Temperature" = Low-temp DH scenario
        high_temp_scenario = temp_scenarios[1]  # First scenario is High_Temp
        low_temp_scenario = temp_scenarios[2]   # Second scenario is Low_Temp

        # Low-temp DH scenario: COPs increase over time (from HeatPumpCOP sheet)
        heatpump_cop_trajectories[low_temp_scenario] = Dict{Symbol,Dict{Int,Float64}}()

        for row in eachrow(cop_df)
            tech = Symbol(row[cop_tech_col])
            cop_by_year = Dict{Int,Float64}()
            for i in 1:T
                cal_year = calendar_years[i]
                col_name = Symbol(string(cal_year))
                cop_by_year[cal_year] = round(Float64(row[col_name]), digits=4)
            end
            heatpump_cop_trajectories[low_temp_scenario][tech] = cop_by_year
        end

        # High-temp DH scenario: COPs stay constant at 2023 values
        heatpump_cop_trajectories[high_temp_scenario] = Dict{Symbol,Dict{Int,Float64}}()

        first_year = calendar_years[1]  # 2023
        for row in eachrow(cop_df)
            tech = Symbol(row[cop_tech_col])
            first_year_cop = round(Float64(row[Symbol(string(first_year))]), digits=4)
            cop_by_year = Dict{Int,Float64}()
            for i in 1:T
                cal_year = calendar_years[i]
                cop_by_year[cal_year] = first_year_cop  # Keep constant
            end
            heatpump_cop_trajectories[high_temp_scenario][tech] = cop_by_year
        end

        println("  Loaded heat pump COP trajectories:")
        for (scenario, techs) in heatpump_cop_trajectories
            println("    $scenario:")
            for (tech, cops) in techs
                cop_values = [cops[calendar_years[i]] for i in 1:T]
                println("      $tech: $(cop_values)")
            end
        end
    else
        # Fallback: use efficiency_th from Technologies sheet as constant COP for heat pumps
        println("  WARNING: 'HeatPumpCOP' sheet not found - using efficiency_th as constant COP for heat pumps")
        for scenario in temp_scenarios
            heatpump_cop_trajectories[scenario] = Dict{Symbol,Dict{Int,Float64}}()
            for tech in technologies
                if occursin("HeatPump", string(tech))
                    cop_value = c_efficiency_th[tech]
                    cop_by_year = Dict{Int,Float64}()
                    for i in 1:T
                        cop_by_year[calendar_years[i]] = cop_value
                    end
                    heatpump_cop_trajectories[scenario][tech] = cop_by_year
                    println("    $scenario / $tech: constant COP = $cop_value")
                end
            end
        end
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

            # Map required base names to actual column names (handles units in column names)
            required_base_cols = ["probability", "demand_multiplier", "elec_price_multiplier", "dc_availability"]
            col_mapping = Dict{String, Symbol}()
            for base_name in required_base_cols
                found = false
                for col_name in names(extreme_events)
                    if startswith(lowercase(string(col_name)), lowercase(base_name))
                        col_mapping[base_name] = Symbol(col_name)
                        found = true
                        break
                    end
                end
                if !found
                    error("ExtremeEvents sheet missing column starting with: $base_name")
                end
            end

            # Rename columns to simple names for internal use
            for (base_name, full_col) in col_mapping
                if full_col != Symbol(base_name)
                    rename!(extreme_events, full_col => Symbol(base_name))
                end
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
        c_penalty, elec_taxes_levies, n_typical_weeks, calendar_years,
        technologies, c_initial_capacity, c_max_additional_capacity,
        c_investment_cost, c_opex_fixed, c_opex_var, c_efficiency_th,
        c_efficiency_el, c_energy_carrier, c_lifetime_new,
        c_capacity_limits,
        storage_params, storage_capacity_limits, c_emission_fac, elec_emission_factors,
        energy_price_map, carbon_trajectory, temp_scenarios, temp_demand_multipliers, temp_scenario_probabilities,
        energy_transitions, initial_energy_dist,
        heatpump_cop_trajectories,
        investment_stages,
        c_existing_capacity_schedule, waste_chp_efficiency_schedule, waste_emission_factor_schedule, waste_availability,
        enable_extreme_events, apply_to_year, extreme_events
    )
end
