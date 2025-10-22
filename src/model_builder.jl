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

            # State variable for demand multiplier
            0 <= x_demand_mult, SDDP.State, (initial_value = 1.0)
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
            @constraint(sp, x_demand_mult.out == x_demand_mult.in)

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

            # Get deterministic energy and carbon prices based on Markovian states
            nat_gas_price = params.energy_price_map[energy_state]
            carbon_price = params.carbon_trajectories[carbon_scenario][model_year]

            # Build state transition constraint for cumulative demand multiplier
            # Pattern: x_demand_mult.out = ε_t × x_demand_mult.in
            # Implemented as: x_demand_mult.out - ε_t × x_demand_mult.in = 0
            #
            # IMPORTANT: Constraint built ONCE here with placeholder coefficient.
            # Actual coefficient (-ε_t) is set inside SDDP.parameterize block below.
            # The negative sign is required because JuMP normalizes to: x.out + (-ε) × x.in = 0
            #
            # ALTERNATIVE APPROACH (from SDDP.jl ARMA tutorial, see https://sddp.dev/stable/tutorial/arma/):
            # Could use a noise variable instead of coefficient manipulation:
            #   @variable(sp, ω)
            #   @constraint(sp, x_demand_mult.out == ε * x_demand_mult.in + ω)
            #   SDDP.parameterize(...) do noise
            #       JuMP.fix(ω, noise)
            #   end
            demand_mult_transition = @constraint(sp, x_demand_mult.out == x_demand_mult.in)

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

            # Build demand balance constraints including the state variable
            # Form: production + discharge - charge + unmet = base_demand * x_demand_mult.out
            # where x_demand_mult.out = ε_t * x_demand_mult.in (set via coefficient in parameterize)
            # Rewritten as: production + discharge - charge + unmet - base_demand * x_demand_mult.out = 0
            # The base_demand coefficients are fixed; only the state transition gets updated in parameterize
            for week in 1:data.n_weeks
                @constraint(sp, [hour in 1:data.hours_per_week],
                    sum(u_production[tech, week, hour] for tech in params.technologies) +
                    u_discharge[week, hour] - u_charge[week, hour] + u_unmet[week, hour] -
                    data.scaled_weeks[week][hour] * x_demand_mult.out == 0.0
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

            # Stage-wise independent demand uncertainty using loaded parameters
            demand_multipliers = t == 2 ? [1.0] : params.demand_multipliers  # Deterministic first year
            demand_probabilities = t == 2 ? [1.0] : params.demand_probabilities

            SDDP.parameterize(sp, demand_multipliers, demand_probabilities) do demand_mult_sample
                # Update ONLY the state transition constraint coefficient
                # State transition: x_demand_mult.out = demand_mult_sample * x_demand_mult.in
                # Implemented as: x_demand_mult.out - demand_mult_sample * x_demand_mult.in = 0
                # Set the coefficient of x_demand_mult.in to -demand_mult_sample
                JuMP.set_normalized_coefficient(demand_mult_transition, x_demand_mult.in, -demand_mult_sample)

                # Operational objective with deterministic energy and carbon prices
                local df = discount_factor(t, params.T_years, params.discount_rate)
                local purch_elec_price = model_year <= 2 ? data.purch_elec_price_2030_weeks : data.purch_elec_price_2050_weeks
                local sale_elec_price = model_year <= 2 ? data.sale_elec_price_2030_weeks : data.sale_elec_price_2050_weeks

                local expr_annual_cost = 0.0

                for week in 1:data.n_weeks
                    week_cost = 0.0
                    for hour in 1:data.hours_per_week
                        # Technology production costs
                        for tech in params.technologies
                            fuel_cost = 0.0
                            if params.c_energy_carrier[tech] == :nat_gas
                                fuel_cost = nat_gas_price
                            elseif params.c_energy_carrier[tech] == :elec
                                fuel_cost = purch_elec_price[week, hour]
                            end

                            tech_cost = params.c_opex_var[tech] * u_production[tech, week, hour] +
                                        (fuel_cost + carbon_price * params.c_emission_fac[params.c_energy_carrier[tech]]) *
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
            end  # End of SDDP.parameterize for demand
        end
    end

    return model
end
