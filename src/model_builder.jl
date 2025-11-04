"""
SDDP model construction for District Heating Capacity Expansion
"""

using SDDP, Gurobi, LinearAlgebra

include("helper_functions.jl")

"""
    build_transition_matrices(T::Int, energy_transitions, initial_energy_dist, carbon_probs)

Build Markovian transition matrices for the policy graph with:
- Stage 1-2: Deterministic (1 node each)
- Stage 3: Carbon policy branching (1→3 nodes)
- Stage 4+: Energy price transitions within carbon scenarios (9 nodes)

# Arguments
- `T::Int`: Number of model years
- `energy_transitions::Matrix{Float64}`: Energy state transition probabilities (3×3)
- `initial_energy_dist::Vector{Float64}`: Initial energy distribution (1×3)
- `carbon_probs::Vector{Float64}`: Carbon policy branching probabilities (1×3)

# Returns
- Vector of transition matrices
"""
function build_transition_matrices(T::Int, energy_transitions::Matrix{Float64},
    initial_energy_dist::Vector{Float64},
    carbon_probs::Vector{Float64})

    transition_matrices = Array{Float64,2}[]

    for stage in 1:(2*T)
        if stage == 1
            # Stage 1: Deterministic root → first investment stage
            push!(transition_matrices, reshape([1.0], 1, 1))

        elseif stage == 2
            # Stage 2: Deterministic inv → opr (year 1)
            push!(transition_matrices, reshape([1.0], 1, 1))

        elseif stage == 3
            # Stage 3: Carbon policy branching (1 node → 3 nodes)
            push!(transition_matrices, reshape(carbon_probs, 1, 3))

        elseif stage == 4
            # Stage 4: First energy branching (3 carbon → 9 energy×carbon nodes)
            # 3×9 matrix: each carbon scenario branches to 3 energy states
            M = zeros(3, 9)
            for c in 1:3
                # Node ordering: (e1,c1), (e2,c1), (e3,c1), (e1,c2), (e2,c2), (e3,c2), (e1,c3), (e2,c3), (e3,c3)
                cols = (c - 1) * 3 .+ (1:3)
                M[c, cols] = initial_energy_dist
            end
            push!(transition_matrices, M)

        elseif isodd(stage)
            # Odd stages > 4: opr → inv (energy stays same within each carbon scenario)
            push!(transition_matrices, Matrix{Float64}(I, 9, 9))

        else
            # Even stages > 4: inv → opr (energy transitions within each carbon scenario)
            # 9×9 block diagonal: energy transitions don't cross carbon scenarios
            M = zeros(9, 9)
            for c in 1:3
                rows = (c - 1) * 3 .+ (1:3)
                cols = (c - 1) * 3 .+ (1:3)
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
    carbon_probs = [params.carbon_probabilities[i] for i in 1:3]  # Convert Dict to Vector
    transition_matrices = build_transition_matrices(
        params.T,
        params.energy_transitions,
        params.initial_energy_dist,
        carbon_probs
    )

    # Helper function to decode markov state into (energy_state, carbon_scenario)
    function decode_markov_state(t::Int, markov_state::Int)
        if t <= 2
            # Stages 1-2: Single deterministic state
            return 1, 1  # energy_state=1 (medium), carbon_scenario=1
        elseif t == 3
            # Stage 3: 3 carbon scenarios, energy not yet branched
            return 1, markov_state  # energy_state=1, carbon_scenario∈{1,2,3}
        else
            # Stages 4+: 9 states representing (energy, carbon) combinations
            # Node ordering: (e1,c1), (e2,c1), (e3,c1), (e1,c2), (e2,c2), (e3,c2), (e1,c3), (e2,c3), (e3,c3)
            carbon_scenario = div(markov_state - 1, 3) + 1
            energy_state = mod(markov_state - 1, 3) + 1
            return energy_state, carbon_scenario
        end
    end

    model = SDDP.MarkovianPolicyGraph(
        transition_matrices=transition_matrices,
        sense=:Min,
        lower_bound=0.0,
        optimizer=Gurobi.Optimizer
    ) do sp, node
        t, markov_state = node
        energy_state, carbon_scenario = decode_markov_state(t, markov_state)

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

        # Create vintage capacity variables for technologies
        # Exclude last investment stage (no vintage for investments that won't be used)
        last_inv_stage = 2 * params.T - 1
        vintage_stages = filter(s -> s != last_inv_stage, params.investment_stages)

        cap_vintage_tech = Dict()
        cap_vintage_stor = Dict()

        for stage in vintage_stages
            if stage == 0
                # Initial capacities
                cap_vintage_tech[stage] = @variable(sp, [tech in params.technologies],
                    SDDP.State, (initial_value = params.c_initial_capacity[tech]),
                    lower_bound = 0, upper_bound = 3000,
                    base_name = "cap_vintage_tech_$(stage)")
            else
                # New investment capacities
                cap_vintage_tech[stage] = @variable(sp, [tech in params.technologies],
                    SDDP.State, (initial_value = 0),
                    lower_bound = 0, upper_bound = params.c_max_additional_capacity[tech],
                    base_name = "cap_vintage_tech_$(stage)")
            end
        end

        @variable(
            sp,
            cap_vintage_stor_state[s in vintage_stages], SDDP.State,
            (initial_value = s == 0 ? params.storage_params[:initial_capacity] : 0.0),
            lower_bound = 0,
            upper_bound = params.storage_params[:max_capacity],
            base_name = "cap_vintage_stor_$(s)"
        )

        for stage in vintage_stages
            cap_vintage_stor[stage] = cap_vintage_stor_state[stage]
        end

        # Initial capacities stay constant
        @constraint(sp, [tech in params.technologies],
            cap_vintage_tech[0][tech].out == cap_vintage_tech[0][tech].in)
        @constraint(sp, cap_vintage_stor[0].out == cap_vintage_stor[0].in)

        ################### Investment Stage ###################
        if t % 2 == 1
            # Update vintage capacities for technologies and storage
            # Only iterate over vintage_stages (excludes last investment stage)
            for stage in vintage_stages[2:end]  # Skip stage 0, iterate over investment stages with vintages
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
            # Compute alive capacities for technologies in next model year
            capacity_alive_next_stage = Dict{Symbol,JuMP.GenericAffExpr{Float64,VariableRef}}()
            for tech in params.technologies
                capacity_expr = 0.0
                for stage in vintage_stages
                    lifetime_dict = (stage == 0) ? params.c_lifetime_initial : params.c_lifetime_new
                    if is_alive(stage, t + 2, lifetime_dict, tech)
                        capacity_expr += cap_vintage_tech[stage][tech].out
                    end
                end
                capacity_alive_next_stage[tech] = capacity_expr
            end

            # Compute alive storage capacity in next model year
            storage_cap_next_stage = 0.0
            for stage in vintage_stages
                if is_storage_alive(stage, t + 2, Int(params.storage_params[:lifetime]))
                    storage_cap_next_stage += cap_vintage_stor[stage].out
                end
            end

            # Capacity limit constraints for next model year
            # Only apply limits before the last investment stage (no need to constrain stage 7)
            # This reduces constraints and improves numerical stability
            if t < 2 * params.T - 1
                next_model_year = min(model_year + 1, params.T)  # Bound by final year
                for tech in params.technologies
                    limit = params.c_capacity_limits[tech][next_model_year]
                    if isfinite(limit)
                        @constraint(sp, capacity_alive_next_stage[tech] <= limit)
                    end
                end

                # Storage capacity limit constraint for next model year
                storage_limit = params.storage_capacity_limits[next_model_year]
                if isfinite(storage_limit)
                    @constraint(sp, storage_cap_next_stage <= storage_limit)
                end
            end

            # Investment objective
            local df = discount_factor(t, params.T_years, params.discount_rate)

            # Investment costs
            expr_invest = sum(params.c_investment_cost[tech] * u_expansion_tech[tech] for tech in params.technologies)
            expr_invest += params.storage_params[:capacity_cost] * u_expansion_storage

            # Fixed O&M costs for technologies
            expr_fix_om = 0.0
            for tech in params.technologies
                for stage in vintage_stages
                    lifetime_dict = (stage == 0) ? params.c_lifetime_initial : params.c_lifetime_new
                    if is_alive(stage, t, lifetime_dict, tech)
                        expr_fix_om += params.c_opex_fixed[tech] * cap_vintage_tech[stage][tech].in
                    end
                end
            end

            # Fixed O&M for storage
            for stage in vintage_stages
                if is_storage_alive(stage, t, Int(params.storage_params[:lifetime]))
                    expr_fix_om += params.storage_params[:fixed_om] * cap_vintage_stor[stage].in
                end
            end

            expr_fix_om *= params.T_years
            @stageobjective(sp, df * (expr_invest + expr_fix_om))

            ################### Operational Stage ###################
        else
            model_year = Int(ceil(t / 2))

            # Get deterministic carrier prices and carbon prices based on Markovian states
            carrier_prices = params.energy_price_map[energy_state][model_year]
            carbon_price = params.carbon_trajectories[carbon_scenario][model_year]

            # Compute alive capacities for technologies
            capacity_alive = Dict{Symbol,JuMP.GenericAffExpr{Float64,VariableRef}}()
            for tech in params.technologies
                capacity_expr = 0.0
                for stage in vintage_stages
                    lifetime_dict = (stage == 0) ? params.c_lifetime_initial : params.c_lifetime_new
                    if is_alive(stage, t, lifetime_dict, tech)
                        capacity_expr += cap_vintage_tech[stage][tech].in
                    end
                end
                capacity_alive[tech] = capacity_expr
            end

            # Compute alive storage capacity
            storage_cap = 0.0
            for stage in vintage_stages
                if is_storage_alive(stage, t, Int(params.storage_params[:lifetime]))
                    storage_cap += cap_vintage_stor[stage].in
                end
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
            for stage in vintage_stages[2:end]
                @constraint(sp, [tech in params.technologies],
                    cap_vintage_tech[stage][tech].out == cap_vintage_tech[stage][tech].in)
                @constraint(sp,
                    cap_vintage_stor[stage].out == cap_vintage_stor[stage].in)
            end

            # Salvage value calculation (deterministic)
            local salvage = 0.0
            if t == params.T * 2
                # Technology salvage
                for tech in params.technologies
                    for stage in vintage_stages
                        stage_year = (stage == 0) ? 0 : Int(ceil(stage / 2))
                        lifetime_dict = (stage == 0) ? params.c_lifetime_initial : params.c_lifetime_new

                        if (model_year - stage_year) < lifetime_dict[tech]
                            remaining_life = lifetime_dict[tech] - (model_year - stage_year)
                            salvage += params.c_investment_cost[tech] * cap_vintage_tech[stage][tech].in *
                                       (remaining_life / lifetime_dict[tech])
                        end
                    end
                end

                # Storage salvage
                for stage in vintage_stages
                    stage_year = (stage == 0) ? 0 : Int(ceil(stage / 2))
                    if (model_year - stage_year) < params.storage_params[:lifetime]
                        remaining_life = params.storage_params[:lifetime] - (model_year - stage_year)
                        salvage += params.storage_params[:capacity_cost] * cap_vintage_stor[stage].in *
                                   (remaining_life / params.storage_params[:lifetime])
                    end
                end
            end

            # Operational objective with deterministic energy and carbon prices and demand
            local df = discount_factor(t, params.T_years, params.discount_rate)

            # Select electricity price scenario based on energy_state (correlated with gas prices)
            # energy_state: 1=low, 2=medium, 3=high
            local scenario = energy_state == 1 ? :low : (energy_state == 2 ? :medium : :high)
            local purch_elec_price = data.purch_elec_prices[model_year][scenario]
            local sale_elec_price = data.sale_elec_prices[model_year][scenario]

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

                        tech_cost = params.c_opex_var[tech] * u_production[tech, week, hour] +
                                    (fuel_cost + carbon_price * emission_factor) *
                                    (u_production[tech, week, hour] / params.c_efficiency_th[tech]) -
                                    params.c_efficiency_el[tech] * sale_elec_price[week, hour] *
                                    (u_production[tech, week, hour] / params.c_efficiency_th[tech])
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
