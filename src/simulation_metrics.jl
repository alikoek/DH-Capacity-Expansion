"""
Performance metrics calculation for simulation results
"""

using Statistics

"""
    calculate_performance_metrics(simulations, params::ModelParameters, data::ProcessedData)

Calculate performance metrics across all simulations.

Returns a dictionary with aggregated metrics.
"""
function calculate_performance_metrics(simulations, params::ModelParameters, data::ProcessedData)
    n_sims = length(simulations)
    n_years = params.T

    # Initialize storage for metrics across simulations
    flh_by_tech = Dict{Symbol,Vector{Float64}}()
    capacity_factor_by_tech = Dict{Symbol,Vector{Float64}}()
    energy_mix_by_tech = Dict{Symbol,Vector{Float64}}()
    lcoh_by_tech = Dict{Symbol,Vector{Float64}}()
    system_lcoh = Float64[]

    for tech in params.technologies
        flh_by_tech[tech] = Float64[]
        capacity_factor_by_tech[tech] = Float64[]
        energy_mix_by_tech[tech] = Float64[]
        lcoh_by_tech[tech] = Float64[]
    end

    # Process each simulation
    for sim in 1:n_sims
        # Collect data for each year (operational stages)
        for year in 1:n_years
            op_stage = 2 * year  # Operational stages are even numbers

            if op_stage > length(simulations[sim])
                continue
            end

            sp = simulations[sim][op_stage]

            # Extract Markov state from simulation to get correct scenario-specific demand
            t_sim, markov_state = sp[:node_index]
            energy_state, temp_scenario = decode_markov_state(t_sim, markov_state)
            scenario_sym = [:high, :medium, :low][energy_state]

            # Calculate annual demand using scenario-specific demand
            annual_demand = 0.0
            for week in 1:data.n_weeks
                week_demand = sum(data.scaled_weeks[year][scenario_sym][week])
                annual_demand += week_demand * data.week_weights_normalized[week]
            end

            # Calculate production and installed capacity for each technology
            total_production = 0.0
            tech_productions = Dict{Symbol,Float64}()
            tech_capacities = Dict{Symbol,Float64}()
            tech_new_capacities = Dict{Symbol,Float64}()
            tech_existing_capacities = Dict{Symbol,Float64}()

            for tech in params.technologies
                # Calculate annual production
                tech_production = 0.0
                for week in 1:data.n_weeks
                    week_prod = sum(value(sp[:u_production][tech, week, hour]) for hour in 1:data.hours_per_week)
                    tech_production += week_prod * data.week_weights_normalized[week]
                end
                tech_productions[tech] = tech_production
                total_production += tech_production

                # Get installed capacity (alive capacity from previous investment stage)
                inv_stage = 2 * year - 1
                new_cap = 0.0
                existing_cap = 0.0

                if inv_stage >= 1 && inv_stage <= length(simulations[sim])
                    sp_inv = simulations[sim][inv_stage]

                    # Sum up alive NEW vintage capacities
                    for stage in params.investment_stages
                        vintage_symbol = Symbol("cap_vintage_tech_$(stage)")
                        if haskey(sp_inv, vintage_symbol)
                            vintage_data = sp_inv[vintage_symbol]
                            if isa(vintage_data, Dict) && haskey(vintage_data, tech)
                                vintage_cap = vintage_data[tech]
                                # Check if vintage is alive at operational stage
                                if is_alive(stage, op_stage, params.c_lifetime_new, tech)
                                    new_cap += vintage_cap
                                end
                            end
                        end
                    end

                    # Add existing capacity from retirement schedule
                    if haskey(params.c_existing_capacity_schedule, tech)
                        existing_cap = params.c_existing_capacity_schedule[tech][year]
                    end
                end

                total_cap = new_cap + existing_cap
                tech_capacities[tech] = total_cap
                tech_new_capacities[tech] = new_cap
                tech_existing_capacities[tech] = existing_cap

                # Calculate Full Load Hours (FLH)
                if total_cap > 0.001
                    flh = tech_production / total_cap
                    push!(flh_by_tech[tech], flh)

                    # Calculate Capacity Factor
                    cf = flh / 8760.0  # 8760 hours per year
                    push!(capacity_factor_by_tech[tech], cf)
                end
            end

            # Calculate energy mix (share of total production) AFTER loop completes
            # This ensures all technologies use the same total_production denominator
            # Always push energy share (including zeros) to reflect overall contribution
            if total_production > 0.001
                for tech in params.technologies
                    energy_share = tech_productions[tech] / total_production
                    push!(energy_mix_by_tech[tech], energy_share)
                end
            end

            # Calculate LCOH by technology using ACTUAL simulation path
            model_year = year  # Model year corresponds to current year

            # Get actual realized state from this simulation path
            t_sim, markov_state = sp[:node_index]
            energy_state, temp_scenario = decode_markov_state(t_sim, markov_state)

            for tech in params.technologies
                if tech_productions[tech] > 0.001
                    # Get capacities for this technology
                    new_cap = tech_new_capacities[tech]
                    total_cap = tech_capacities[tech]

                    # Annualized CAPEX (using capital recovery factor)
                    # Only apply to NEW capacity (existing capacity already paid for)
                    crf = params.discount_rate * (1 + params.discount_rate)^params.c_lifetime_new[tech] /
                          ((1 + params.discount_rate)^params.c_lifetime_new[tech] - 1)
                    annual_capex = params.c_investment_cost[tech] * new_cap * crf  # Already TSEK/MW

                    # Annual fixed O&M - apply to TOTAL capacity (new + existing)
                    annual_fixed_om = params.c_opex_fixed[tech] * total_cap  # Already TSEK/MW

                    # Variable O&M
                    annual_var_om = params.c_opex_var[tech] * tech_productions[tech]  # Already TSEK/MWh

                    # Get efficiency based on technology type and temperature scenario
                    carrier = params.c_energy_carrier[tech]
                    calendar_year = params.calendar_years[model_year]
                    temp_scenario_symbol = params.temp_scenarios[temp_scenario]

                    if tech == :Waste_CHP && params.waste_chp_efficiency_schedule[model_year] > 0.0
                        # Time-varying efficiency for Waste_CHP
                        efficiency_th = params.waste_chp_efficiency_schedule[model_year]
                    elseif haskey(params.heatpump_cop_trajectories, temp_scenario_symbol) &&
                           haskey(params.heatpump_cop_trajectories[temp_scenario_symbol], tech)
                        # Use technology-specific, temperature-scenario-dependent, year-varying COP
                        efficiency_th = params.heatpump_cop_trajectories[temp_scenario_symbol][tech][calendar_year]
                    else
                        efficiency_th = params.c_efficiency_th[tech]
                    end

                    # Electrical efficiency (maintain constant power-to-heat ratio for Waste_CHP)
                    if tech == :Waste_CHP && params.waste_chp_efficiency_schedule[model_year] > 0.0
                        # Calculate base power-to-heat ratio and apply to time-varying thermal efficiency
                        α_waste_chp = params.c_efficiency_el[tech] / params.c_efficiency_th[tech]
                        efficiency_el = α_waste_chp * efficiency_th
                    else
                        efficiency_el = params.c_efficiency_el[tech]
                    end

                    # Energy consumption
                    energy_consumption = tech_productions[tech] / efficiency_th  # MWh of fuel

                    # Calculate fuel cost using prices from realized energy state
                    if carrier == :elec
                        # Production-weighted electricity price (exact, not approximation)
                        scenario_symbol = [:high, :medium, :low][energy_state]
                        elec_prices = data.purch_elec_prices[model_year][scenario_symbol]

                        # Calculate weighted average: sum(production × price) / sum(production)
                        weighted_cost = 0.0
                        for week in 1:data.n_weeks
                            for hour in 1:data.hours_per_week
                                prod_hour = value(sp[:u_production][tech, week, hour])
                                price_hour = elec_prices[week, hour]
                                weighted_cost += prod_hour * price_hour * data.week_weights_normalized[week]
                            end
                        end
                        carrier_price = weighted_cost / tech_productions[tech]  # TSEK/MWh
                    else
                        # Non-electricity carriers: single price from realized energy state
                        carrier_price = params.energy_price_map[energy_state][model_year][carrier]
                    end

                    # Fuel purchase cost
                    fuel_cost = carrier_price * energy_consumption  # Already TSEK

                    # Carbon costs
                    carbon_price = params.carbon_trajectory[model_year]  # Already TSEK
                    if carrier == :elec
                        emission_factor = params.elec_emission_factors[model_year]
                    else
                        emission_factor = params.c_emission_fac[carrier]
                    end
                    carbon_cost = carbon_price * emission_factor * energy_consumption  # TSEK

                    # Electricity sales revenue (for CHP) - production-weighted sale price
                    elec_revenue = 0.0
                    if efficiency_el > 0.0 && haskey(data.sale_elec_prices, model_year)
                        scenario_symbol = [:high, :medium, :low][energy_state]
                        sale_prices = data.sale_elec_prices[model_year][scenario_symbol]

                        # Calculate production-weighted average sale price
                        weighted_revenue = 0.0
                        for week in 1:data.n_weeks
                            for hour in 1:data.hours_per_week
                                prod_hour = value(sp[:u_production][tech, week, hour])
                                price_hour = sale_prices[week, hour]
                                weighted_revenue += prod_hour * price_hour * data.week_weights_normalized[week]
                            end
                        end
                        sale_price = weighted_revenue / tech_productions[tech]  # TSEK/MWh
                        elec_revenue = efficiency_el * sale_price * energy_consumption  # Already TSEK
                    end

                    # Total annual cost
                    total_annual_cost = annual_capex + annual_fixed_om + annual_var_om + fuel_cost + carbon_cost - elec_revenue

                    # LCOH
                    lcoh = total_annual_cost / tech_productions[tech]  # TSEK/MWh
                    push!(lcoh_by_tech[tech], lcoh)
                end
            end

            # Calculate system-wide LCOH
            # This would require full system cost from objective function
            # For now, approximate as weighted average of technology LCOHs
            if total_production > 0.001
                weighted_lcoh = 0.0
                for tech in params.technologies
                    if tech_productions[tech] > 0.001
                        tech_share = tech_productions[tech] / total_production
                        if !isempty(lcoh_by_tech[tech])
                            weighted_lcoh += tech_share * lcoh_by_tech[tech][end]
                        end
                    end
                end
                push!(system_lcoh, weighted_lcoh)
            end
        end
    end

    # Aggregate statistics
    metrics = Dict{String,Any}()

    # Full Load Hours statistics
    metrics["FLH"] = Dict{Symbol,Dict{String,Float64}}()
    for tech in params.technologies
        if !isempty(flh_by_tech[tech])
            metrics["FLH"][tech] = Dict(
                "mean" => mean(flh_by_tech[tech]),
                "median" => median(flh_by_tech[tech]),
                "std" => std(flh_by_tech[tech]),
                "min" => minimum(flh_by_tech[tech]),
                "max" => maximum(flh_by_tech[tech])
            )
        end
    end

    # Capacity Factor statistics
    metrics["Capacity_Factor"] = Dict{Symbol,Dict{String,Float64}}()
    for tech in params.technologies
        if !isempty(capacity_factor_by_tech[tech])
            metrics["Capacity_Factor"][tech] = Dict(
                "mean" => mean(capacity_factor_by_tech[tech]),
                "median" => median(capacity_factor_by_tech[tech]),
                "std" => std(capacity_factor_by_tech[tech]),
                "min" => minimum(capacity_factor_by_tech[tech]),
                "max" => maximum(capacity_factor_by_tech[tech])
            )
        end
    end

    # Energy Mix statistics
    metrics["Energy_Mix"] = Dict{Symbol,Dict{String,Float64}}()
    for tech in params.technologies
        if !isempty(energy_mix_by_tech[tech])
            metrics["Energy_Mix"][tech] = Dict(
                "mean" => mean(energy_mix_by_tech[tech]),
                "median" => median(energy_mix_by_tech[tech]),
                "std" => std(energy_mix_by_tech[tech])
            )
        end
    end

    # LCOH statistics
    metrics["LCOH_by_Tech"] = Dict{Symbol,Dict{String,Float64}}()
    for tech in params.technologies
        if !isempty(lcoh_by_tech[tech])
            metrics["LCOH_by_Tech"][tech] = Dict(
                "mean" => mean(lcoh_by_tech[tech]),
                "median" => median(lcoh_by_tech[tech]),
                "std" => std(lcoh_by_tech[tech]),
                "min" => minimum(lcoh_by_tech[tech]),
                "max" => maximum(lcoh_by_tech[tech])
            )
        end
    end

    # System LCOH statistics
    if !isempty(system_lcoh)
        metrics["System_LCOH"] = Dict(
            "mean" => mean(system_lcoh),
            "median" => median(system_lcoh),
            "std" => std(system_lcoh),
            "min" => minimum(system_lcoh),
            "max" => maximum(system_lcoh)
        )
    end

    return metrics
end


"""
    export_performance_metrics(metrics::Dict, output_file::String)

Export performance metrics to a text file.
"""
function export_performance_metrics(metrics::Dict, output_file::String)
    open(output_file, "w") do io
        println(io, "="^80)
        println(io, "PERFORMANCE METRICS SUMMARY")
        println(io, "="^80)
        println(io)

        # Full Load Hours
        if haskey(metrics, "FLH")
            println(io, "--- FULL LOAD HOURS (hours/year) ---")
            println(io)
            for (tech, stats) in metrics["FLH"]
                println(io, "$tech:")
                println(io, "  Mean:   $(round(stats["mean"], digits=2)) hours")
                println(io, "  Median: $(round(stats["median"], digits=2)) hours")
                println(io, "  Std Dev: $(round(stats["std"], digits=2)) hours")
                println(io, "  Range:  $(round(stats["min"], digits=2)) - $(round(stats["max"], digits=2)) hours")
                println(io)
            end
        end

        # Capacity Factor
        if haskey(metrics, "Capacity_Factor")
            println(io, "--- CAPACITY FACTOR (%) ---")
            println(io)
            for (tech, stats) in metrics["Capacity_Factor"]
                println(io, "$tech:")
                println(io, "  Mean:   $(round(stats["mean"] * 100, digits=2))%")
                println(io, "  Median: $(round(stats["median"] * 100, digits=2))%")
                println(io, "  Std Dev: $(round(stats["std"] * 100, digits=2))%")
                println(io, "  Range:  $(round(stats["min"] * 100, digits=2))% - $(round(stats["max"] * 100, digits=2))%")
                println(io)
            end
        end

        # Energy Mix
        if haskey(metrics, "Energy_Mix")
            println(io, "--- ENERGY MIX (share of total production) ---")
            println(io)
            for (tech, stats) in metrics["Energy_Mix"]
                println(io, "$tech:")
                println(io, "  Mean:   $(round(stats["mean"] * 100, digits=2))%")
                println(io, "  Median: $(round(stats["median"] * 100, digits=2))%")
                println(io, "  Std Dev: $(round(stats["std"] * 100, digits=2))%")
                println(io)
            end
        end

        # LCOH by Technology
        if haskey(metrics, "LCOH_by_Tech")
            println(io, "--- LEVELIZED COST OF HEAT BY TECHNOLOGY (TSEK/MWh) ---")
            println(io)
            for (tech, stats) in metrics["LCOH_by_Tech"]
                println(io, "$tech:")
                println(io, "  Mean:   $(round(stats["mean"], digits=4)) TSEK/MWh")
                println(io, "  Median: $(round(stats["median"], digits=4)) TSEK/MWh")
                println(io, "  Std Dev: $(round(stats["std"], digits=4)) TSEK/MWh")
                println(io, "  Range:  $(round(stats["min"], digits=4)) - $(round(stats["max"], digits=4)) TSEK/MWh")
                println(io)
            end
        end

        # System LCOH
        if haskey(metrics, "System_LCOH")
            println(io, "--- SYSTEM-WIDE LEVELIZED COST OF HEAT (TSEK/MWh) ---")
            println(io)
            stats = metrics["System_LCOH"]
            println(io, "  Mean:   $(round(stats["mean"], digits=4)) TSEK/MWh")
            println(io, "  Median: $(round(stats["median"], digits=4)) TSEK/MWh")
            println(io, "  Std Dev: $(round(stats["std"], digits=4)) TSEK/MWh")
            println(io, "  Range:  $(round(stats["min"], digits=4)) - $(round(stats["max"], digits=4)) TSEK/MWh")
            println(io)
        end

        println(io, "="^80)
    end

    println("Performance metrics exported to: $output_file")
end
