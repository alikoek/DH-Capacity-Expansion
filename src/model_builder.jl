"""
SDDP model construction for District Heating Capacity Expansion
"""

using SDDP, Gurobi, LinearAlgebra, Distributions

include("helper_functions.jl")

"""
    build_transition_matrices(T::Int)

Build Markovian transition matrices for the policy graph.

# Arguments
- `T::Int`: Number of model years

# Returns
- Vector of transition matrices
"""
function build_transition_matrices(T::Int)
    I = Diagonal(ones(3))
    transition_matrices = Array{Float64,2}[
        [1.0]',
        [1.0]',
        [0.3 0.5 0.2],
        I,
        [0.3 0.5 0.2; 0.3 0.5 0.2; 0.3 0.5 0.2],
        I,
        [0.3 0.5 0.2; 0.3 0.5 0.2; 0.3 0.5 0.2],
        I,
    ]
    return transition_matrices
end

"""
    build_price_distribution(params::ModelParameters)

Build the energy price distribution and scenarios.

# Arguments
- `params::ModelParameters`: Model parameters

# Returns
- Tuple of (price_values, price_probabilities_normalized)
"""
function build_price_distribution(params::ModelParameters)
    μ_normal = log(params.mean_price^2 / sqrt(params.price_volatility^2 + params.mean_price^2))
    σ_normal = sqrt(log(1 + (params.price_volatility / params.mean_price)^2))
    price_distribution = LogNormal(μ_normal, σ_normal)

    price_quantiles = range(0.05, 0.95; length=params.num_price_scenarios)
    price_values = quantile.(price_distribution, price_quantiles)
    price_probabilities = pdf(price_distribution, price_values)
    price_probabilities_normalized = price_probabilities / sum(price_probabilities)

    return price_values, price_probabilities_normalized
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

    # Build transition matrices and price distribution
    transition_matrices = build_transition_matrices(params.T)
    price_values, price_probabilities_normalized = build_price_distribution(params)

    model = SDDP.MarkovianPolicyGraph(
        transition_matrices=transition_matrices,
        sense=:Min,
        lower_bound=0.0,
        optimizer=Gurobi.Optimizer
    ) do sp, node
        t, demand_state = node

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
        cap_vintage_tech = Dict()
        cap_vintage_stor = Dict()

        for stage in params.investment_stages
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
            cap_vintage_stor_state[s in params.investment_stages], SDDP.State,
            (initial_value = s == 0 ? params.storage_params[:initial_capacity] : 0.0),
            lower_bound = 0,
            upper_bound = params.storage_params[:max_capacity],
            base_name = "cap_vintage_stor_$(s)"
        )

        for stage in params.investment_stages
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
            for stage in params.investment_stages[2:end]
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
                for stage in params.investment_stages
                    lifetime_dict = (stage == 0) ? params.c_lifetime_initial : params.c_lifetime_new
                    if is_alive(stage, t, lifetime_dict, tech)
                        expr_fix_om += params.c_opex_fixed[tech] * cap_vintage_tech[stage][tech].in
                    end
                end
            end

            # Fixed O&M for storage
            for stage in params.investment_stages
                if is_storage_alive(stage, t, Int(params.storage_params[:lifetime]))
                    expr_fix_om += params.storage_params[:fixed_om] * cap_vintage_stor[stage].in
                end
            end

            expr_fix_om *= params.T_years
            @stageobjective(sp, df * (expr_invest + expr_fix_om))

        ################### Operational Stage ###################
        else
            model_year = Int(ceil(t / 2))

            # Update demand multiplier
            if t == 2
                new_demand_mult = 1.0
            else
                new_demand_mult = params.demand_multipliers[demand_state] * x_demand_mult.in
            end

            # Compute alive capacities for technologies
            capacity_alive = Dict{Symbol,JuMP.GenericAffExpr{Float64,VariableRef}}()
            for tech in params.technologies
                capacity_expr = 0.0
                for stage in params.investment_stages
                    lifetime_dict = (stage == 0) ? params.c_lifetime_initial : params.c_lifetime_new
                    if is_alive(stage, t, lifetime_dict, tech)
                        capacity_expr += cap_vintage_tech[stage][tech].in
                    end
                end
                capacity_alive[tech] = capacity_expr
            end

            # Compute alive storage capacity
            storage_cap = 0.0
            for stage in params.investment_stages
                if is_storage_alive(stage, t, Int(params.storage_params[:lifetime]))
                    storage_cap += cap_vintage_stor[stage].in
                end
            end

            # Operational constraints for each representative week
            for week in 1:data.n_weeks
                # Demand balance: production + discharge - charge + unmet = demand
                @constraint(sp, [hour in 1:data.hours_per_week],
                    sum(u_production[tech, week, hour] for tech in params.technologies) +
                    u_discharge[week, hour] - u_charge[week, hour] + u_unmet[week, hour] ==
                    data.scaled_weeks[week][hour] * new_demand_mult
                )

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
                            params.storage_params[:efficiency] * u_charge[week, hour] - u_discharge[week, hour]
                        )
                    else
                        # Subsequent hours: include previous level and losses
                        @constraint(sp,
                            u_level[week, hour] ==
                            u_level[week, hour-1] * (1 - params.storage_params[:loss_rate]/24) +
                            params.storage_params[:efficiency] * u_charge[week, hour] - u_discharge[week, hour]
                        )
                    end
                end

                # End-of-week constraint: storage must be nearly empty
                @constraint(sp, u_level[week, data.hours_per_week] <= 0.01 * storage_cap)
            end

            # State updates
            @constraint(sp, x_demand_mult.out == new_demand_mult)

            for stage in params.investment_stages[2:end]
                @constraint(sp, [tech in params.technologies],
                    cap_vintage_tech[stage][tech].out == cap_vintage_tech[stage][tech].in)
                @constraint(sp,
                    cap_vintage_stor[stage].out == cap_vintage_stor[stage].in)
            end

            # Salvage value calculation
            local salvage = 0.0
            if t == params.T * 2
                # Technology salvage
                for tech in params.technologies
                    for stage in params.investment_stages
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
                for stage in params.investment_stages
                    stage_year = (stage == 0) ? 0 : Int(ceil(stage / 2))
                    if (model_year - stage_year) < params.storage_params[:lifetime]
                        remaining_life = params.storage_params[:lifetime] - (model_year - stage_year)
                        salvage += params.storage_params[:capacity_cost] * cap_vintage_stor[stage].in *
                                  (remaining_life / params.storage_params[:lifetime])
                    end
                end
            end

            # Operational objective with random energy prices
            SDDP.parameterize(sp, price_values, price_probabilities_normalized) do ω
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
                                fuel_cost = ω
                            elseif params.c_energy_carrier[tech] == :elec
                                fuel_cost = purch_elec_price[week, hour]
                            end

                            tech_cost = params.c_opex_var[tech] * u_production[tech, week, hour] +
                                       (fuel_cost + params.c_carbon_price[model_year] * params.c_emission_fac[params.c_energy_carrier[tech]]) *
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
    end

    return model
end
