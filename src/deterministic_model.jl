"""
Deterministic capacity expansion model (Expected-Value Benchmark)

This module creates a deterministic version of the SDDP model using expected values
for all uncertain parameters. This serves as a benchmark to demonstrate the value
of stochastic programming (VSS calculation).
"""

# Use SDDP's JuMP dependency (no separate JuMP package needed)
import SDDP.JuMP as JuMP
using Gurobi

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
    build_deterministic_model(params::ModelParameters, data::ProcessedData)

Build a deterministic capacity expansion model using expected values for all uncertainties.

This model:
- Uses weighted average COP multiplier (across DH system temperature scenarios)
- Uses probability-weighted energy prices (evolved via Markov transitions)
- Computes expected carrier prices and electricity prices at each stage
- Follows the same two-stage (investment/operational) structure
- Includes same constraints as SDDP model but without stochastic branching

# Arguments
- `params::ModelParameters`: Model parameters structure
- `data::ProcessedData`: Processed data structure

# Returns
- `model::JuMP.Model`: Deterministic JuMP optimization model
- `variables::Dict`: Dictionary of decision variables for post-processing
"""
function build_deterministic_model(params::ModelParameters, data::ProcessedData)
    println("Building deterministic (expected-value) model...")

    # Calculate expected values for uncertain parameters
    println("  Calculating expected parameter values...")

    # Expected COP multiplier across DH system temperature scenarios (weighted average)
    expected_cop_multiplier = sum(params.temp_scenario_probabilities[i] *
                                  params.temp_cop_multipliers[params.temp_scenarios[i]]
                                  for i in 1:2)
    println("    Expected COP multiplier: $(round(expected_cop_multiplier, digits=3))")

    # Evolve energy state probabilities through Markov transitions
    # This matches the SDDP graph structure: stage 2 uses initial distribution,
    # subsequent operational stages evolve via transitions
    energy_probs = evolve_energy_probabilities(params.initial_energy_dist, params.energy_transitions)

    println("    Stage-wise energy state probabilities (1=High, 2=Medium, 3=Low):")
    for t in [2, 4, 6, 8]
        prob_sum = sum(energy_probs[t])
        println("      Stage $t (Year $(Int(t/2))): $(round.(energy_probs[t], digits=3)) (sum=$(round(prob_sum, digits=3)))")
    end

    # Calculate expected values for extreme events
    expected_demand_mult = Dict{Int, Float64}()
    expected_elec_price_mult = Dict{Int, Float64}()
    expected_dc_avail = Dict{Int, Float64}()

    if params.enable_extreme_events && params.extreme_events !== nothing
        extreme_stage = params.apply_to_year * 2

        # Probability-weighted expected values
        exp_demand = sum(row.probability * row.demand_multiplier
                        for row in eachrow(params.extreme_events))
        exp_elec_price = sum(row.probability * row.elec_price_multiplier
                            for row in eachrow(params.extreme_events))
        exp_dc = sum(row.probability * row.dc_availability
                    for row in eachrow(params.extreme_events))

        expected_demand_mult[extreme_stage] = exp_demand
        expected_elec_price_mult[extreme_stage] = exp_elec_price
        expected_dc_avail[extreme_stage] = exp_dc

        println("    Extreme events expected values for stage $extreme_stage (Year $(params.apply_to_year)):")
        println("      Demand multiplier: $(round(exp_demand, digits=3))")
        println("      Electricity price multiplier: $(round(exp_elec_price, digits=3))")
        println("      DC availability: $(round(exp_dc, digits=3))")
    end

    # Default to 1.0 for all operational stages (overridden above if extreme events enabled)
    for t in [2, 4, 6, 8]
        if !haskey(expected_demand_mult, t)
            expected_demand_mult[t] = 1.0
            expected_elec_price_mult[t] = 1.0
            expected_dc_avail[t] = 1.0
        end
    end

    # Create JuMP model
    model = Model(Gurobi.Optimizer)
    set_silent(model)

    # Decision variables organized by stage
    inv_vars = Dict()  # Investment variables by stage
    opr_vars = Dict()  # Operational variables by stage

    # Vintage capacity tracking (same as SDDP model)
    last_inv_stage = 2 * params.T - 1
    vintage_stages = filter(s -> s != last_inv_stage, params.investment_stages)

    # Create vintage capacity variables
    cap_vintage_tech = Dict()
    cap_vintage_stor = Dict()

    for stage in vintage_stages
        cap_vintage_tech[stage] = @variable(model, [tech in params.technologies],
                                           lower_bound = 0,
                                           upper_bound = params.c_max_additional_capacity[tech],
                                           base_name = "cap_vintage_tech_$(stage)")

        cap_vintage_stor[stage] = @variable(model, lower_bound = 0,
                                           upper_bound = params.storage_params[:max_capacity],
                                           base_name = "cap_vintage_stor_$(stage)")
    end

    # Build each stage sequentially
    stage_objectives = []

    for t in 1:(2*params.T)
        model_year = Int(ceil(t / 2))

        if isodd(t)  # Investment stage
            println("  Building investment stage $t (Year $model_year)...")

            # Investment decision variables
            u_expansion_tech = @variable(model, [tech in params.technologies],
                                        lower_bound = 0,
                                        upper_bound = params.c_max_additional_capacity[tech],
                                        base_name = "u_expansion_tech_$(t)")

            u_expansion_storage = @variable(model, lower_bound = 0,
                                           upper_bound = params.storage_params[:max_capacity],
                                           base_name = "u_expansion_storage_$(t)")

            inv_vars[t] = (u_expansion_tech=u_expansion_tech, u_expansion_storage=u_expansion_storage)

            # Update vintage capacities
            # Set vintage capacity equal to expansion decision for current stage
            if t in vintage_stages
                @constraint(model, [tech in params.technologies],
                    cap_vintage_tech[t][tech] == u_expansion_tech[tech],
                    base_name = "vintage_tech_update_$(t)_$(tech)")

                @constraint(model,
                    cap_vintage_stor[t] == u_expansion_storage,
                    base_name = "vintage_stor_update_$(t)")
            end

            # Calculate alive capacity for next operational stage
            next_model_year = min(model_year + 1, params.T)
            capacity_alive_next = Dict{Symbol, Any}()

            for tech in params.technologies
                capacity_expr = sum(cap_vintage_tech[stage][tech]
                                   for stage in vintage_stages
                                   if is_alive(stage, t + 2, params.c_lifetime_new, tech);
                                   init=0.0)

                # Add existing capacity
                if haskey(params.c_existing_capacity_schedule, tech)
                    capacity_expr += params.c_existing_capacity_schedule[tech][next_model_year]
                end

                capacity_alive_next[tech] = capacity_expr
            end

            # Storage capacity alive next stage
            storage_cap_next = sum(cap_vintage_stor[stage]
                                  for stage in vintage_stages
                                  if is_storage_alive(stage, t + 2, Int(params.storage_params[:lifetime]));
                                  init=0.0)
            storage_cap_next += params.storage_params[:initial_capacity]

            # Multi-year look-ahead capacity constraints
            if t < 2 * params.T - 1
                for future_year in (model_year + 1):params.T
                    future_stage = 2 * future_year

                    capacity_alive_future = Dict{Symbol, Any}()
                    for tech in params.technologies
                        capacity_expr = sum(cap_vintage_tech[stage][tech]
                                           for stage in vintage_stages
                                           if is_alive(stage, future_stage, params.c_lifetime_new, tech);
                                           init=0.0)

                        if haskey(params.c_existing_capacity_schedule, tech)
                            capacity_expr += params.c_existing_capacity_schedule[tech][future_year]
                        end

                        capacity_alive_future[tech] = capacity_expr
                    end

                    # Apply capacity limits
                    for tech in params.technologies
                        limit = params.c_capacity_limits[tech][future_year]
                        if isfinite(limit)
                            @constraint(model, capacity_alive_future[tech] <= limit)
                        end
                    end

                    # Storage capacity limit
                    storage_cap_future = sum(cap_vintage_stor[stage]
                                            for stage in vintage_stages
                                            if is_storage_alive(stage, future_stage, Int(params.storage_params[:lifetime]));
                                            init=0.0)
                    storage_cap_future += params.storage_params[:initial_capacity]

                    storage_limit = params.storage_capacity_limits[future_year]
                    if isfinite(storage_limit)
                        @constraint(model, storage_cap_future <= storage_limit)
                    end
                end
            end

            # Investment stage objective
            df = discount_factor(t, params.T_years, params.discount_rate)

            expr_invest = sum(params.c_investment_cost[tech] * u_expansion_tech[tech]
                             for tech in params.technologies)
            expr_invest += params.storage_params[:capacity_cost] * u_expansion_storage

            # Fixed O&M costs for NEW investments
            expr_fix_om = 0.0
            for tech in params.technologies
                for stage in vintage_stages
                    if is_alive(stage, t, params.c_lifetime_new, tech)
                        expr_fix_om += params.c_opex_fixed[tech] * cap_vintage_tech[stage][tech]
                    end
                end

                # Existing capacity O&M
                if haskey(params.c_existing_capacity_schedule, tech)
                    expr_fix_om += params.c_opex_fixed[tech] *
                                  params.c_existing_capacity_schedule[tech][model_year]
                end
            end

            # Storage fixed O&M
            for stage in vintage_stages
                if is_storage_alive(stage, t, Int(params.storage_params[:lifetime]))
                    expr_fix_om += params.storage_params[:fixed_om] * cap_vintage_stor[stage]
                end
            end
            expr_fix_om += params.storage_params[:fixed_om] * params.storage_params[:initial_capacity]
            expr_fix_om *= params.T_years

            push!(stage_objectives, df * (expr_invest + expr_fix_om))

        else  # Operational stage
            println("  Building operational stage $t (Year $model_year)...")

            # Compute probability-weighted carrier prices using evolved probabilities
            # State mapping: 1=High, 2=Medium, 3=Low (matches Excel energy_price_map)
            # Get unique non-electricity carriers from technologies
            unique_carriers = unique([params.c_energy_carrier[tech] for tech in params.technologies])
            non_elec_carriers = filter(c -> c != :elec, unique_carriers)

            carrier_prices = Dict{Symbol, Float64}()
            for carrier in non_elec_carriers
                carrier_prices[carrier] = sum(
                    energy_probs[t][s] * params.energy_price_map[s][model_year][carrier]
                    for s in 1:3
                )
            end

            carbon_price = params.carbon_trajectory[model_year]

            # Apply expected COP multiplier (from system temperature scenarios) to efficiencies
            efficiency_th_adjusted = Dict{Symbol, Float64}()
            for tech in params.technologies
                base_eff = params.c_efficiency_th[tech]

                # Time-varying Waste_CHP efficiency
                if tech == :Waste_CHP && params.waste_chp_efficiency_schedule[model_year] > 0.0
                    base_eff = params.waste_chp_efficiency_schedule[model_year]
                end

                # Apply expected COP multiplier for heat pumps
                if occursin("HeatPump", string(tech))
                    efficiency_th_adjusted[tech] = base_eff * expected_cop_multiplier
                else
                    efficiency_th_adjusted[tech] = base_eff
                end
            end

            # Operational variables
            u_production = @variable(model, [tech in params.technologies,
                                            week=1:data.n_weeks,
                                            hour=1:data.hours_per_week],
                                    lower_bound = 0,
                                    base_name = "u_production_$(t)")

            u_charge = @variable(model, [week=1:data.n_weeks, hour=1:data.hours_per_week],
                                lower_bound = 0,
                                base_name = "u_charge_$(t)")

            u_discharge = @variable(model, [week=1:data.n_weeks, hour=1:data.hours_per_week],
                                   lower_bound = 0,
                                   base_name = "u_discharge_$(t)")

            u_level = @variable(model, [week=1:data.n_weeks, hour=1:data.hours_per_week],
                               lower_bound = 0,
                               base_name = "u_level_$(t)")

            u_unmet = @variable(model, [week=1:data.n_weeks, hour=1:data.hours_per_week],
                               lower_bound = 0,
                               base_name = "u_unmet_$(t)")

            opr_vars[t] = (u_production=u_production, u_charge=u_charge, u_discharge=u_discharge,
                          u_level=u_level, u_unmet=u_unmet)

            # Calculate alive capacities
            capacity_alive = Dict{Symbol, Any}()
            for tech in params.technologies
                capacity_expr = sum(cap_vintage_tech[stage][tech]
                                   for stage in vintage_stages
                                   if is_alive(stage, t, params.c_lifetime_new, tech);
                                   init=0.0)

                if haskey(params.c_existing_capacity_schedule, tech)
                    capacity_expr += params.c_existing_capacity_schedule[tech][model_year]
                end

                capacity_alive[tech] = capacity_expr
            end

            storage_cap = sum(cap_vintage_stor[stage]
                             for stage in vintage_stages
                             if is_storage_alive(stage, t, Int(params.storage_params[:lifetime]));
                             init=0.0)
            storage_cap += params.storage_params[:initial_capacity]

            # Demand balance constraints (with extreme event expected demand multiplier)
            @constraint(model, [week=1:data.n_weeks, hour=1:data.hours_per_week],
                sum(u_production[tech, week, hour] for tech in params.technologies) +
                u_discharge[week, hour] - u_charge[week, hour] + u_unmet[week, hour] ==
                data.scaled_weeks[week][hour] * expected_demand_mult[t],
                base_name = "demand_balance_$(t)_$(week)_$(hour)")

            # Capacity constraints (with extreme event expected DC availability)
            for tech in params.technologies
                if tech == :DataCenter_HeatPump
                    # Apply expected DC availability multiplier to DataCenter_HeatPump capacity
                    @constraint(model, [week=1:data.n_weeks, hour=1:data.hours_per_week],
                        u_production[tech, week, hour] <= capacity_alive[tech] * expected_dc_avail[t],
                        base_name = "tech_capacity_$(t)_$(tech)")
                else
                    @constraint(model, [week=1:data.n_weeks, hour=1:data.hours_per_week],
                        u_production[tech, week, hour] <= capacity_alive[tech],
                        base_name = "tech_capacity_$(t)_$(tech)")
                end
            end

            # Storage constraints
            @constraint(model, [week=1:data.n_weeks, hour=1:data.hours_per_week],
                u_charge[week, hour] <= params.storage_params[:max_charge_rate] * storage_cap)

            @constraint(model, [week=1:data.n_weeks, hour=1:data.hours_per_week],
                u_discharge[week, hour] <= params.storage_params[:max_discharge_rate] * storage_cap)

            @constraint(model, [week=1:data.n_weeks, hour=1:data.hours_per_week],
                u_level[week, hour] <= storage_cap)

            # Storage dynamics
            for week in 1:data.n_weeks
                for hour in 1:data.hours_per_week
                    if hour == 1
                        @constraint(model,
                            u_level[week, hour] ==
                            params.storage_params[:efficiency] * u_charge[week, hour] -
                            u_discharge[week, hour] / params.storage_params[:efficiency])
                    else
                        @constraint(model,
                            u_level[week, hour] ==
                            u_level[week, hour-1] * (1 - params.storage_params[:loss_rate] / 24) +
                            params.storage_params[:efficiency] * u_charge[week, hour] -
                            u_discharge[week, hour] / params.storage_params[:efficiency])
                    end
                end

                # End-of-week constraint
                @constraint(model, u_level[week, data.hours_per_week] <= 0.01 * storage_cap)
            end

            # Waste fuel availability
            if :Waste_CHP in params.technologies && params.waste_availability[model_year] > 0
                waste_eff = efficiency_th_adjusted[:Waste_CHP]
                @constraint(model,
                    sum(data.week_weights_normalized[week] *
                        sum(u_production[:Waste_CHP, week, hour] / waste_eff
                            for hour in 1:data.hours_per_week)
                        for week in 1:data.n_weeks)
                    <= params.waste_availability[model_year])
            end

            # Operational objective
            df = discount_factor(t, params.T_years, params.discount_rate)

            # Compute probability-weighted electricity prices (per week, hour)
            # State mapping: 1=High, 2=Medium, 3=Low (matches data price scenarios)
            # Apply extreme event expected electricity price multiplier
            expected_purch_elec = zeros(data.n_weeks, data.hours_per_week)
            expected_sale_elec = zeros(data.n_weeks, data.hours_per_week)

            for week in 1:data.n_weeks, hour in 1:data.hours_per_week
                for s in 1:3
                    scenario_sym = s == 1 ? :high : (s == 2 ? :medium : :low)
                    expected_purch_elec[week, hour] += energy_probs[t][s] * data.purch_elec_prices[model_year][scenario_sym][week, hour]
                    expected_sale_elec[week, hour] += energy_probs[t][s] * data.sale_elec_prices[model_year][scenario_sym][week, hour]
                end
                # Apply extreme event electricity price multiplier
                expected_purch_elec[week, hour] *= expected_elec_price_mult[t]
                expected_sale_elec[week, hour] *= expected_elec_price_mult[t]
            end

            # Print diagnostic for first operational stage
            if model_year == 1
                println("    Sample weighted electricity prices (week 1, hour 1):")
                for s in 1:3
                    scenario_sym = s == 1 ? :high : (s == 2 ? :medium : :low)
                    price = data.purch_elec_prices[model_year][scenario_sym][1, 1]
                    println("      State $s ($scenario_sym): $(round(price, digits=4)) TSEK/MWh × prob=$(round(energy_probs[t][s], digits=3))")
                end
                println("      Weighted average: $(round(expected_purch_elec[1, 1], digits=4)) TSEK/MWh")
            end

            expr_annual_cost = 0.0

            for week in 1:data.n_weeks
                week_cost = 0.0
                for hour in 1:data.hours_per_week
                    # Technology production costs
                    for tech in params.technologies
                        carrier = params.c_energy_carrier[tech]

                        fuel_cost = (carrier == :elec) ? expected_purch_elec[week, hour] : carrier_prices[carrier]
                        emission_factor = (carrier == :elec) ? params.elec_emission_factors[model_year] :
                                         params.c_emission_fac[carrier]

                        tech_cost = params.c_opex_var[tech] * u_production[tech, week, hour] +
                                   (fuel_cost + carbon_price * emission_factor) *
                                   (u_production[tech, week, hour] / efficiency_th_adjusted[tech]) -
                                   params.c_efficiency_el[tech] * expected_sale_elec[week, hour] *
                                   (u_production[tech, week, hour] / efficiency_th_adjusted[tech])
                        week_cost += tech_cost
                    end

                    # Storage and unmet demand costs
                    week_cost += params.storage_params[:variable_om] * u_discharge[week, hour]
                    week_cost += params.c_penalty * u_unmet[week, hour]
                end

                expr_annual_cost += data.week_weights_normalized[week] * week_cost
            end

            # Salvage value (only in last stage)
            salvage = 0.0
            if t == params.T * 2
                # Technology salvage
                for tech in params.technologies
                    for stage in vintage_stages
                        stage_year = Int(ceil(stage / 2))
                        if (model_year - stage_year) < params.c_lifetime_new[tech]
                            remaining_life = params.c_lifetime_new[tech] - (model_year - stage_year)
                            salvage += params.c_investment_cost[tech] * cap_vintage_tech[stage][tech] *
                                      (remaining_life / params.c_lifetime_new[tech])
                        end
                    end
                end

                # Storage salvage
                for stage in vintage_stages
                    stage_year = Int(ceil(stage / 2))
                    if (model_year - stage_year) < params.storage_params[:lifetime]
                        remaining_life = params.storage_params[:lifetime] - (model_year - stage_year)
                        salvage += params.storage_params[:capacity_cost] * cap_vintage_stor[stage] *
                                  (remaining_life / params.storage_params[:lifetime])
                    end
                end
            end

            opr_stage_cost = df * (params.T_years * expr_annual_cost - salvage * params.salvage_fraction)
            push!(stage_objectives, opr_stage_cost)
        end
    end

    # Set total objective
    @objective(model, Min, sum(stage_objectives))

    println("✓ Deterministic model built successfully")
    println("  Total stages: $(2*params.T)")
    println("  Total variables: $(num_variables(model))")
    println("  Total constraints: $(num_constraints(model; count_variable_in_set_constraints=false))")

    # Return model and variable references for post-processing
    variables = Dict(
        :inv_vars => inv_vars,
        :opr_vars => opr_vars,
        :cap_vintage_tech => cap_vintage_tech,
        :cap_vintage_stor => cap_vintage_stor
    )

    return model, variables
end
