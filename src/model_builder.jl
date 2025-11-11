"""
SDDP model construction for District Heating Capacity Expansion
"""

using SDDP, Gurobi, LinearAlgebra

# Note: helper_functions.jl is included by DHCapEx.jl module

"""
    build_transition_matrices(T::Int, energy_transitions, initial_energy_dist, temp_probs)

Build Markovian transition matrices for the policy graph with:
- Stage 1: System temperature scenario branching (1→2 nodes) - EARLY BRANCHING
- Stage 2: Energy price branching within each system temp scenario (2→6 nodes)
- Stage 3+: Energy price transitions within system temperature scenarios (6 nodes)

# Arguments
- `T::Int`: Number of model years
- `energy_transitions::Matrix{Float64}`: Energy state transition probabilities (3×3)
- `initial_energy_dist::Vector{Float64}`: Initial energy distribution (1×3)
- `temp_probs::Vector{Float64}`: System temperature scenario branching probabilities (1×2)

# Returns
- Vector of transition matrices
"""
function build_transition_matrices(T::Int, energy_transitions::Matrix{Float64},
    initial_energy_dist::Vector{Float64},
    temp_probs::Vector{Float64})

    transition_matrices = Array{Float64,2}[]

    for stage in 1:(2*T)
        if stage == 1
            # Stage 1: System temperature scenario branching at root (1 node → 2 nodes)
            # This makes first investments aware of DH system temperature regime
            push!(transition_matrices, reshape(temp_probs, 1, 2))

        elseif stage == 2
            # Stage 2: First energy branching (2 temp → 6 energy×temp nodes)
            # 2×6 matrix: each system temperature scenario branches to 3 energy states
            M = zeros(2, 6)
            for temp in 1:2
                # Node ordering: (e1,t1), (e2,t1), (e3,t1), (e1,t2), (e2,t2), (e3,t2)
                cols = (temp - 1) * 3 .+ (1:3)
                M[temp, cols] = initial_energy_dist
            end
            push!(transition_matrices, M)

        elseif isodd(stage)
            # Odd stages > 2: opr → inv (energy stays same within each temp scenario)
            push!(transition_matrices, Matrix{Float64}(I, 6, 6))

        else
            # Even stages > 2: inv → opr (energy transitions within each temp scenario)
            # 6×6 block diagonal: energy transitions don't cross system temperature scenarios
            M = zeros(6, 6)
            for temp in 1:2
                rows = (temp - 1) * 3 .+ (1:3)
                cols = (temp - 1) * 3 .+ (1:3)
                M[rows, cols] = energy_transitions
            end
            push!(transition_matrices, M)
        end
    end

    return transition_matrices
end

"""
    build_sddp_model(params::ModelParameters, data::ProcessedData)

Build the SDDP model for capacity expansion optimization.

# Arguments
- `params::ModelParameters`: Model parameters structure
- `data::ProcessedData`: Processed data structure

# Returns
- SDDP.PolicyGraph: The constructed SDDP model
"""
function build_sddp_model(params::ModelParameters, data::ProcessedData)
    println("Building SDDP model...")

    # Build transition matrices
    # Build transition matrices using loaded parameters
    temp_probs = [params.temp_scenario_probabilities[i] for i in 1:2]  # Convert Dict to Vector
    transition_matrices = build_transition_matrices(
        params.T,
        params.energy_transitions,
        params.initial_energy_dist,
        temp_probs
    )

    # Helper function to decode markov state into (energy_state, temp_scenario)
    function decode_markov_state(t::Int, markov_state::Int)
        if t == 1
            # Stage 1: System temperature branching only (2 nodes)
            return 1, markov_state  # energy_state=1 (default), temp_scenario∈{1,2}
        else
            # Stages 2+: 6 states representing (energy, temp) combinations
            # Node ordering: (e1,t1), (e2,t1), (e3,t1), (e1,t2), (e2,t2), (e3,t2)
            temp_scenario = div(markov_state - 1, 3) + 1
            energy_state = mod(markov_state - 1, 3) + 1
            return energy_state, temp_scenario
        end
    end

    model = SDDP.MarkovianPolicyGraph(
        transition_matrices=transition_matrices,
        sense=:Min,
        lower_bound=-1e8,
        optimizer=Gurobi.Optimizer
    ) do sp, node
        t, markov_state = node
        energy_state, temp_scenario = decode_markov_state(t, markov_state)

        # Variables
        @variables(sp, begin
            # Investment decisions for technologies
            0 <= u_expansion_tech[tech in params.technologies] <= params.c_max_additional_capacity[tech]
            # Investment decision for storage
            0 <= u_expansion_storage <= params.storage_params[:max_capacity]

            # Production variables
            u_production[tech in params.technologies, week=1:data.n_weeks, hour=1:data.hours_per_week] >= 0

            # Storage variables
            0 <= u_charge[week=1:data.n_weeks, hour=1:data.hours_per_week]
            0 <= u_discharge[week=1:data.n_weeks, hour=1:data.hours_per_week]
            0 <= u_level[week=1:data.n_weeks, hour=1:data.hours_per_week]

            # Unmet demand
            0 <= u_unmet[week=1:data.n_weeks, hour=1:data.hours_per_week]
        end)

        # Create vintage capacity variables for NEW investments only (no stage 0)
        # Exclude last investment stage (no vintage for investments that won't be used)
        last_inv_stage = 2 * params.T - 1
        vintage_stages = filter(s -> s != last_inv_stage, params.investment_stages)

        cap_vintage_tech = Dict()
        cap_vintage_stor = Dict()

        # Create state variables only for NEW investment stages (excluding stage 0)
        for stage in vintage_stages
            cap_vintage_tech[stage] = @variable(sp, [tech in params.technologies],
                SDDP.State, (initial_value = 0),
                lower_bound = 0, upper_bound = params.c_max_additional_capacity[tech],
                base_name = "cap_vintage_tech_$(stage)")
        end

        # Storage vintages (also excluding stage 0)
        @variable(
            sp,
            cap_vintage_stor_state[s in vintage_stages], SDDP.State,
            (initial_value = 0.0),
            lower_bound = 0,
            upper_bound = params.storage_params[:max_capacity],
            base_name = "cap_vintage_stor_$(s)"
        )

        for stage in vintage_stages
            cap_vintage_stor[stage] = cap_vintage_stor_state[stage]
        end

        ################### Investment Stage ###################
        if t % 2 == 1
            # Update vintage capacities for technologies and storage
            # Iterate over all vintage_stages (all are NEW investments, no stage 0)
            for stage in vintage_stages
                if stage == t
                    # Current investment stage - add new expansions
                    @constraint(sp, [tech in params.technologies],
                        cap_vintage_tech[stage][tech].out == cap_vintage_tech[stage][tech].in + u_expansion_tech[tech])
                    @constraint(sp,
                        cap_vintage_stor[stage].out == cap_vintage_stor[stage].in + u_expansion_storage)
                else
                    # Other stages - no change
                    @constraint(sp, [tech in params.technologies],
                        cap_vintage_tech[stage][tech].out == cap_vintage_tech[stage][tech].in)
                    @constraint(sp,
                        cap_vintage_stor[stage].out == cap_vintage_stor[stage].in)
                end
            end

            model_year = Int(ceil(t / 2))
            next_model_year = min(model_year + 1, params.T)

            # Compute alive capacities from NEW investments for next model year
            capacity_alive_next_stage = Dict{Symbol,JuMP.GenericAffExpr{Float64,VariableRef}}()
            for tech in params.technologies
                capacity_expr = 0.0
                # Sum NEW investment vintages only
                for stage in vintage_stages
                    if is_alive(stage, t + 2, params.c_lifetime_new, tech)
                        capacity_expr += cap_vintage_tech[stage][tech].out
                    end
                end

                # Add existing capacity from retirement schedule
                if haskey(params.c_existing_capacity_schedule, tech)
                    existing_cap = params.c_existing_capacity_schedule[tech][next_model_year]
                    capacity_expr += existing_cap
                end

                capacity_alive_next_stage[tech] = capacity_expr
            end

            # Compute alive storage capacity in next model year (NEW investments only)
            storage_cap_next_stage = 0.0
            for stage in vintage_stages
                if is_storage_alive(stage, t + 2, Int(params.storage_params[:lifetime]))
                    storage_cap_next_stage += cap_vintage_stor[stage].out
                end
            end
            # Add existing storage capacity from retirement schedule (consistent with technologies)
            if haskey(params.storage_params, :existing_capacity_schedule)
                storage_cap_next_stage += params.storage_params[:existing_capacity_schedule][next_model_year]
            else
                # Fallback: assume initial capacity does not retire (documented assumption)
                storage_cap_next_stage += params.storage_params[:initial_capacity]
            end

            # Multi-year look-ahead capacity limit constraints
            # Apply constraints for ALL future years, not just next year
            # This ensures the model plans investments accounting for future retirements
            if t < 2 * params.T - 1
                # Loop through all future years from current investment decision
                for future_year in (model_year + 1):params.T
                    future_stage = 2 * future_year  # Operational stage for that year

                    # Calculate capacity alive at future_year for each technology
                    capacity_alive_future = Dict{Symbol,JuMP.GenericAffExpr{Float64,VariableRef}}()

                    for tech in params.technologies
                        capacity_expr = 0.0

                        # Sum NEW investment vintages that will be alive at future_year
                        for stage in vintage_stages
                            if is_alive(stage, future_stage, params.c_lifetime_new, tech)
                                # Use .out for current stage (t), .in for future stages
                                if stage == t
                                    capacity_expr += cap_vintage_tech[stage][tech].out
                                else
                                    capacity_expr += cap_vintage_tech[stage][tech].in
                                end
                            end
                        end

                        # Add existing capacity from retirement schedule at future_year
                        if haskey(params.c_existing_capacity_schedule, tech)
                            existing_cap = params.c_existing_capacity_schedule[tech][future_year]
                            capacity_expr += existing_cap
                        end

                        capacity_alive_future[tech] = capacity_expr
                    end

                    # Apply capacity limits for future_year
                    for tech in params.technologies
                        limit = params.c_capacity_limits[tech][future_year]
                        if isfinite(limit)
                            @constraint(sp, capacity_alive_future[tech] <= limit)
                        end
                    end

                    # Calculate storage capacity alive at future_year
                    storage_cap_future = 0.0
                    for stage in vintage_stages
                        if is_storage_alive(stage, future_stage, Int(params.storage_params[:lifetime]))
                            # Use .out for current stage, .in for future stages
                            if stage == t
                                storage_cap_future += cap_vintage_stor[stage].out
                            else
                                storage_cap_future += cap_vintage_stor[stage].in
                            end
                        end
                    end
                    # Add existing storage capacity from retirement schedule at future_year
                    if haskey(params.storage_params, :existing_capacity_schedule)
                        storage_cap_future += params.storage_params[:existing_capacity_schedule][future_year]
                    else
                        # Fallback: use initial capacity as constant (legacy behavior)
                        storage_cap_future += params.storage_params[:initial_capacity]
                    end

                    # Storage capacity limit constraint for future_year
                    storage_limit = params.storage_capacity_limits[future_year]
                    if isfinite(storage_limit)
                        @constraint(sp, storage_cap_future <= storage_limit)
                    end
                end
            end

            # Investment objective
            local df = discount_factor(t, params.T_years, params.discount_rate)

            # Investment costs
            expr_invest = sum(params.c_investment_cost[tech] * u_expansion_tech[tech] for tech in params.technologies)
            expr_invest += params.storage_params[:capacity_cost] * u_expansion_storage

            # Fixed O&M costs for NEW investment technologies
            expr_fix_om = 0.0
            for tech in params.technologies
                for stage in vintage_stages
                    if is_alive(stage, t, params.c_lifetime_new, tech)
                        expr_fix_om += params.c_opex_fixed[tech] * cap_vintage_tech[stage][tech].in
                    end
                end

                # Add fixed O&M for existing capacity from retirement schedule
                if haskey(params.c_existing_capacity_schedule, tech)
                    existing_cap = params.c_existing_capacity_schedule[tech][model_year]
                    expr_fix_om += params.c_opex_fixed[tech] * existing_cap
                end
            end

            # Fixed O&M for NEW investment storage
            for stage in vintage_stages
                if is_storage_alive(stage, t, Int(params.storage_params[:lifetime]))
                    expr_fix_om += params.storage_params[:fixed_om] * cap_vintage_stor[stage].in
                end
            end
            # Add fixed O&M for existing storage
            expr_fix_om += params.storage_params[:fixed_om] * params.storage_params[:initial_capacity]

            expr_fix_om *= params.T_years
            @stageobjective(sp, df * (expr_invest + expr_fix_om))

            ################### Operational Stage ###################
        else
            model_year = Int(ceil(t / 2))

            # Get deterministic carrier prices and carbon price based on Markovian states
            carrier_prices = params.energy_price_map[energy_state][model_year]
            carbon_price = params.carbon_trajectory[model_year]  # Single net-zero trajectory

            # Compute alive capacities from NEW investments
            capacity_alive = Dict{Symbol,Union{Float64,JuMP.GenericAffExpr{Float64,VariableRef}}}()
            for tech in params.technologies
                capacity_expr = 0.0
                # Sum NEW investment vintages
                for stage in vintage_stages
                    if is_alive(stage, t, params.c_lifetime_new, tech)
                        capacity_expr += cap_vintage_tech[stage][tech].in
                    end
                end

                # Add existing capacity from retirement schedule
                if haskey(params.c_existing_capacity_schedule, tech)
                    existing_cap = params.c_existing_capacity_schedule[tech][model_year]
                    capacity_expr += existing_cap
                end

                capacity_alive[tech] = capacity_expr
            end

            # Compute alive storage capacity from NEW investments
            storage_cap = 0.0
            for stage in vintage_stages
                if is_storage_alive(stage, t, Int(params.storage_params[:lifetime]))
                    storage_cap += cap_vintage_stor[stage].in
                end
            end
            # Add existing storage from retirement schedule
            if haskey(params.storage_params, :existing_capacity_schedule)
                storage_cap += params.storage_params[:existing_capacity_schedule][model_year]
            else
                # Fallback to constant for backward compatibility
                storage_cap += params.storage_params[:initial_capacity]
            end

            # Demand balance constraints (deterministic demand = base_annual_demand)
            # Form: production + discharge - charge + unmet = base_demand
            for week in 1:data.n_weeks
                @constraint(sp, [hour in 1:data.hours_per_week],
                    sum(u_production[tech, week, hour] for tech in params.technologies) +
                    u_discharge[week, hour] - u_charge[week, hour] + u_unmet[week, hour] ==
                    data.scaled_weeks[week][hour]
                )
            end

            # Structural operational constraints (same for all demand realizations)
            for week in 1:data.n_weeks
                # Technology capacity constraints
                @constraint(sp, [tech in params.technologies, hour in 1:data.hours_per_week],
                    u_production[tech, week, hour] <= capacity_alive[tech]
                )

                # Storage rate constraints
                @constraint(sp, [hour in 1:data.hours_per_week],
                    u_charge[week, hour] <= params.storage_params[:max_charge_rate] * storage_cap
                )

                @constraint(sp, [hour in 1:data.hours_per_week],
                    u_discharge[week, hour] <= params.storage_params[:max_discharge_rate] * storage_cap
                )

                # Storage capacity constraint
                @constraint(sp, [hour in 1:data.hours_per_week],
                    u_level[week, hour] <= storage_cap
                )

                # Storage dynamics
                for hour in 1:data.hours_per_week
                    if hour == 1
                        # First hour: start empty
                        @constraint(sp,
                            u_level[week, hour] ==
                            params.storage_params[:efficiency] * u_charge[week, hour] - u_discharge[week, hour] / params.storage_params[:efficiency]
                        )
                    else
                        # Subsequent hours: include previous level and losses
                        @constraint(sp,
                            u_level[week, hour] ==
                            u_level[week, hour-1] * (1 - params.storage_params[:loss_rate] / 24) +
                            params.storage_params[:efficiency] * u_charge[week, hour] - u_discharge[week, hour] / params.storage_params[:efficiency]
                        )
                    end
                end

                # End-of-week constraint: storage must be nearly empty
                @constraint(sp, u_level[week, data.hours_per_week] <= 0.01 * storage_cap)
            end

            # State updates for capacity vintages (deterministic - same for all demand realizations)
            for stage in vintage_stages
                @constraint(sp, [tech in params.technologies],
                    cap_vintage_tech[stage][tech].out == cap_vintage_tech[stage][tech].in)
                @constraint(sp,
                    cap_vintage_stor[stage].out == cap_vintage_stor[stage].in)
            end

            # Salvage value calculation (deterministic, NEW investments only)
            local salvage = 0.0
            if t == params.T * 2
                # Technology salvage (NEW investments only)
                for tech in params.technologies
                    for stage in vintage_stages
                        stage_year = Int(ceil(stage / 2))

                        if (model_year - stage_year) < params.c_lifetime_new[tech]
                            remaining_life = params.c_lifetime_new[tech] - (model_year - stage_year)
                            salvage += params.c_investment_cost[tech] * cap_vintage_tech[stage][tech].in *
                                       (remaining_life / params.c_lifetime_new[tech])
                        end
                    end
                    # Note: Existing capacity has no salvage value (already built)
                end

                # Storage salvage (NEW investments only)
                for stage in vintage_stages
                    stage_year = Int(ceil(stage / 2))
                    if (model_year - stage_year) < params.storage_params[:lifetime]
                        remaining_life = params.storage_params[:lifetime] - (model_year - stage_year)
                        salvage += params.storage_params[:capacity_cost] * cap_vintage_stor[stage].in *
                                   (remaining_life / params.storage_params[:lifetime])
                    end
                end
                # Note: Existing storage has no salvage value
            end

            # Operational objective with deterministic energy and carbon prices and demand
            local df = discount_factor(t, params.T_years, params.discount_rate)

            # Select electricity price scenario based on energy_state (correlated with gas prices)
            # energy_state: 1=high, 2=medium, 3=low (matches Excel energy_price_map)
            local scenario = energy_state == 1 ? :high : (energy_state == 2 ? :medium : :low)
            local purch_elec_price = data.purch_elec_prices[model_year][scenario]
            local sale_elec_price = data.sale_elec_prices[model_year][scenario]

            # Apply temperature-dependent COP multipliers to HeatPump efficiency
            temp_scenario_symbol = params.temp_scenarios[temp_scenario]
            cop_multiplier = params.temp_cop_multipliers[temp_scenario_symbol]

            # Create adjusted efficiencies (temperature and time-varying)
            efficiency_th_adjusted = Dict{Symbol, Float64}()
            for tech in params.technologies
                # Start with base efficiency
                base_eff = params.c_efficiency_th[tech]

                # Apply Waste_CHP time-varying efficiency if applicable
                if tech == :Waste_CHP && params.waste_chp_efficiency_schedule[model_year] > 0.0
                    base_eff = params.waste_chp_efficiency_schedule[model_year]
                end

                # Apply temperature-dependent COP multiplier for heat pumps
                if occursin("HeatPump", string(tech))
                    efficiency_th_adjusted[tech] = base_eff * cop_multiplier
                else
                    efficiency_th_adjusted[tech] = base_eff
                end
            end

            # Waste fuel availability constraint (annual limit on waste input)
            if :Waste_CHP in params.technologies && params.waste_availability[model_year] > 0
                waste_eff = efficiency_th_adjusted[:Waste_CHP]
                @constraint(sp,
                    sum(data.week_weights_normalized[week] *
                        sum(u_production[:Waste_CHP, week, hour] / waste_eff
                            for hour in 1:data.hours_per_week)
                        for week in 1:data.n_weeks)
                    <= params.waste_availability[model_year]
                )
            end

            local expr_annual_cost = 0.0

            for week in 1:data.n_weeks
                week_cost = 0.0
                for hour in 1:data.hours_per_week
                    # Technology production costs
                    for tech in params.technologies
                        carrier = params.c_energy_carrier[tech]

                        # Get fuel cost based on carrier type
                        if carrier == :elec
                            fuel_cost = purch_elec_price[week, hour]
                        else
                            # Use carrier price from energy price map
                            fuel_cost = carrier_prices[carrier]
                        end

                        # Get emission factor (time-varying for electricity, static for others)
                        emission_factor = (carrier == :elec) ? params.elec_emission_factors[model_year] : params.c_emission_fac[carrier]

                        # Use temperature-adjusted efficiency
                        tech_cost = params.c_opex_var[tech] * u_production[tech, week, hour] +
                                    (fuel_cost + carbon_price * emission_factor) *
                                    (u_production[tech, week, hour] / efficiency_th_adjusted[tech]) -
                                    params.c_efficiency_el[tech] * sale_elec_price[week, hour] *
                                    (u_production[tech, week, hour] / efficiency_th_adjusted[tech])
                        week_cost += tech_cost
                    end

                    # Storage operational cost (only on discharge)
                    week_cost += params.storage_params[:variable_om] * u_discharge[week, hour]

                    # Unmet demand penalty
                    week_cost += params.c_penalty * u_unmet[week, hour]
                end

                # Apply week weight
                expr_annual_cost += data.week_weights_normalized[week] * week_cost
            end

            @stageobjective(sp, df * (params.T_years * expr_annual_cost - salvage * params.salvage_fraction))
        end
    end

    return model
end
