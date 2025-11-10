"""
Performance metrics calculation for simulation results
"""

using Statistics

"""
    evolve_energy_probabilities(initial_dist, transitions)

Evolve energy state probabilities through Markov transitions for operational stages.
Matches SDDP graph structure: p_t = p_{t-2} × energy_transitions for t ∈ {4,6,8}.

# Arguments
- `initial_dist::Vector{Float64}`: Initial energy state distribution (for stage 2)
- `transitions::Matrix{Float64}`: 3×3 Markov transition matrix

# Returns
Dictionary mapping stage number → probability vector for operational stages {2,4,6,8}
"""
function evolve_energy_probabilities(initial_dist::Vector{Float64}, transitions::Matrix{Float64})
    probs = Dict{Int, Vector{Float64}}()
    probs[2] = copy(initial_dist)

    # Evolve probabilities for subsequent operational stages
    for t in [4, 6, 8]
        probs[t] = vec(probs[t-2]' * transitions)  # vec() converts Adjoint to Vector
    end

    return probs
end

"""
    calculate_performance_metrics(simulations, params::ModelParameters, data::ProcessedData)

Calculate performance metrics across all simulations.

Returns a dictionary with aggregated metrics.
"""
function calculate_performance_metrics(simulations, params::ModelParameters, data::ProcessedData)
    n_sims = length(simulations)
    n_years = params.T

    # Initialize storage for metrics across simulations
    flh_by_tech = Dict{Symbol, Vector{Float64}}()
    capacity_factor_by_tech = Dict{Symbol, Vector{Float64}}()
    energy_mix_by_tech = Dict{Symbol, Vector{Float64}}()
    lcoh_by_tech = Dict{Symbol, Vector{Float64}}()
    system_lcoh = Float64[]

    for tech in params.technologies
        flh_by_tech[tech] = Float64[]
        capacity_factor_by_tech[tech] = Float64[]
        energy_mix_by_tech[tech] = Float64[]
        lcoh_by_tech[tech] = Float64[]
    end

    # Evolve energy state probabilities through Markov transitions
    # This ensures LCOH calculations use stage-specific probabilities (not just initial distribution)
    energy_probs = evolve_energy_probabilities(params.initial_energy_dist, params.energy_transitions)

    # Process each simulation
    for sim in 1:n_sims
        # Collect data for each year (operational stages)
        for year in 1:n_years
            op_stage = 2 * year  # Operational stages are even numbers

            if op_stage > length(simulations[sim])
                continue
            end

            sp = simulations[sim][op_stage]

            # Calculate annual demand
            annual_demand = 0.0
            for week in 1:data.n_weeks
                week_demand = sum(data.scaled_weeks[week])
                annual_demand += week_demand * data.week_weights_normalized[week]
            end

            # Calculate production and installed capacity for each technology
            total_production = 0.0
            tech_productions = Dict{Symbol, Float64}()
            tech_capacities = Dict{Symbol, Float64}()

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
                if inv_stage >= 1 && inv_stage <= length(simulations[sim])
                    sp_inv = simulations[sim][inv_stage]
                    alive_cap = 0.0

                    # Sum up alive vintage capacities
                    for stage in params.investment_stages
                        vintage_symbol = Symbol("cap_vintage_tech_$(stage)")
                        if haskey(sp_inv, vintage_symbol)
                            vintage_data = sp_inv[vintage_symbol]
                            if isa(vintage_data, Dict) && haskey(vintage_data, tech)
                                vintage_cap = vintage_data[tech]
                                # Check if vintage is alive at operational stage
                                lifetime_dict = (stage == 0) ? params.c_lifetime_initial : params.c_lifetime_new
                                if is_alive(stage, op_stage, lifetime_dict, tech)
                                    alive_cap += vintage_cap
                                end
                            end
                        end
                    end

                    tech_capacities[tech] = alive_cap

                    # Calculate Full Load Hours (FLH)
                    if alive_cap > 0.001
                        flh = tech_production / alive_cap
                        push!(flh_by_tech[tech], flh)

                        # Calculate Capacity Factor
                        cf = flh / 8760.0  # 8760 hours per year
                        push!(capacity_factor_by_tech[tech], cf)
                    end
                end

                # Calculate energy mix (share of total production)
                if total_production > 0.001
                    energy_share = tech_production / total_production
                    push!(energy_mix_by_tech[tech], energy_share)
                end
            end

            # Calculate LCOH by technology
            model_year = year  # Model year corresponds to current year
            for tech in params.technologies
                if tech_productions[tech] > 0.001 && tech_capacities[tech] > 0.001
                    # Annualized CAPEX (using capital recovery factor)
                    crf = params.discount_rate * (1 + params.discount_rate)^params.c_lifetime_new[tech] /
                          ((1 + params.discount_rate)^params.c_lifetime_new[tech] - 1)

                    annual_capex = params.c_investment_cost[tech] * tech_capacities[tech] * crf * 1000  # Convert MSEK to TSEK

                    # Annual fixed O&M
                    annual_fixed_om = params.c_opex_fixed[tech] * tech_capacities[tech] * 1000  # Convert MSEK to TSEK

                    # Variable O&M
                    annual_var_om = params.c_opex_var[tech] * tech_productions[tech] / 1000  # Convert SEK to TSEK

                    # Fuel costs (energy carrier + carbon costs - electricity sales)
                    carrier = params.c_energy_carrier[tech]
                    # Use time-varying efficiency for Waste_CHP
                    if tech == :Waste_CHP && params.waste_chp_efficiency_schedule[model_year] > 0.0
                        efficiency_th = params.waste_chp_efficiency_schedule[model_year]
                    else
                        efficiency_th = params.c_efficiency_th[tech]
                    end
                    efficiency_el = params.c_efficiency_el[tech]

                    # Calculate average energy prices across scenarios (weighted by stage-evolved probabilities)
                    avg_carrier_price = 0.0
                    n_energy_states = size(params.energy_transitions, 1)

                    if carrier == :elec
                        # Electricity has time-varying hourly prices
                        if haskey(data.purch_elec_prices, model_year)
                            for energy_state in 1:n_energy_states
                                state_prob = energy_probs[op_stage][energy_state]
                                scenario_symbol = [:high, :medium, :low][energy_state]
                                elec_prices = data.purch_elec_prices[model_year][scenario_symbol]
                                avg_carrier_price += state_prob * mean(elec_prices)
                            end
                        end
                    else
                        # Other carriers have fixed prices in energy_price_map
                        for energy_state in 1:n_energy_states
                            state_prob = energy_probs[op_stage][energy_state]
                            carrier_price = params.energy_price_map[energy_state][model_year][carrier]
                            avg_carrier_price += state_prob * carrier_price
                        end
                    end

                    # Energy consumption
                    energy_consumption = tech_productions[tech] / efficiency_th  # MWh of fuel

                    # Carbon costs
                    carbon_price = params.carbon_trajectory[model_year]
                    # Get emission factor (time-varying for electricity, static for others)
                    if carrier == :elec
                        emission_factor = params.elec_emission_factors[model_year]
                    else
                        emission_factor = params.c_emission_fac[carrier]
                    end
                    carbon_cost = carbon_price * emission_factor * energy_consumption  # TSEK

                    # Fuel purchase cost
                    fuel_cost = avg_carrier_price * energy_consumption / 1000  # Convert SEK to TSEK

                    # Electricity sales revenue (for CHP)
                    # Use average electricity sale price across scenarios (stage-evolved probabilities)
                    avg_elec_sale_price = 0.0
                    if haskey(data.sale_elec_prices, model_year)
                        for energy_state in 1:n_energy_states
                            state_prob = energy_probs[op_stage][energy_state]
                            scenario_symbol = [:high, :medium, :low][energy_state]
                            elec_prices = data.sale_elec_prices[model_year][scenario_symbol]
                            avg_elec_sale_price += state_prob * mean(elec_prices)
                        end
                    end
                    elec_revenue = efficiency_el * avg_elec_sale_price * energy_consumption / 1000  # TSEK

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
    metrics = Dict{String, Any}()

    # Full Load Hours statistics
    metrics["FLH"] = Dict{Symbol, Dict{String, Float64}}()
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
    metrics["Capacity_Factor"] = Dict{Symbol, Dict{String, Float64}}()
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
    metrics["Energy_Mix"] = Dict{Symbol, Dict{String, Float64}}()
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
    metrics["LCOH_by_Tech"] = Dict{Symbol, Dict{String, Float64}}()
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
