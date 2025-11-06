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
# function build_transition_matrices(T::Int)
#     I = Diagonal(ones(3))
#     transition_matrices = Array{Float64,2}[
#         [1.0]',
#         [1.0]',
#         [0.3 0.5 0.2],
#         I,
#         [0.3 0.5 0.2; 0.3 0.5 0.2; 0.3 0.5 0.2],
#         I,
#         [0.3 0.5 0.2; 0.3 0.5 0.2; 0.3 0.5 0.2],
#         I,
#     ]
#     return transition_matrices
# end

function build_transition_matrices_new(stage2year_phase, state2keys, keys2state, p_policy, p_price)
    n_stages = length(state2keys)
    # Create a 2x2 array of zeros
    println("hello")

    B = zeros(n_stages)
    for i in 1:n_stages    # rows
        policy1, price1 = state2keys[i]
        if price1 == "mid"
            B[i] = p_policy[p_policy.policy .== policy1, :probability][1]
        end
    end

    # Matrix 9 stages on 9 stages - from investment to 
    A = zeros(n_stages, n_stages)
    for i in 1:size(A, 1)    # rows
        for j in 1:size(A, 2)    # columns
            policy1, price1 = state2keys[i]
            policy2, price2 = state2keys[j]

            if policy1 != policy2
                A[i,j] = 0
            else
                price_proba = p_price[p_price[!, "price"] .== price1, price2] 
                A[i, j] =  first(price_proba)
            end
        end
    end
    
    I = Diagonal(ones(n_stages))
    
    transition_matrices = Array{Float64,2}[]
    for i in 1:length(stage2year_phase)
        year, phase = stage2year_phase[i]
        
        if i == 1
            push!(transition_matrices, [1.0]')
        elseif i == 2
            push!(transition_matrices, [1.0]')
        elseif i == 3
            push!(transition_matrices, B')
        elseif i == 4
            push!(transition_matrices, A)
        elseif phase == "investment"
            push!(transition_matrices, I)
        else
            push!(transition_matrices, A)
        end
    end
    # for matrix in transition_matrices
    #     println("Matrix:")
    #     for row in eachrow(matrix)
    #         println(row)
    #     end
    # end
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
function build_distribution(errors)
    error_distribution = Dict()
    for row in eachrow(errors)
        year, price, ts_type = first(row["year"]),first(row["scenario_price"]),first(row["ts_type"])
        RMSE = first(row["RMSE"])
        μ_normal = 0
        σ_normal = RMSE

        price_distribution = Normal(μ_normal, σ_normal)
        quantiles = range(0.05, 0.95; length=5)
        values = quantile.(price_distribution, quantiles)
        probabilities = pdf(price_distribution, values)
        probabilities_normalized = probabilities / sum(probabilities)
        error_distribution[year, price, ts_type] = Dict(
            :probabilities => probabilities_normalized,
            :values => values,
        )
    end
    # print(error_distribution)
    return error_distribution
    # return price_values, price_probabilities_normalized
end

function build_dictionnaries(policies_transition, prices_transition, rep_years)
    state2keys = Dict()
    keys2state = Dict()
    i = 0
    for policy in eachrow(policies_transition)
        nest = Dict()
        for price in eachrow(prices_transition)  # Fixed: use prices_transition
            nest[price["price"]] = i
            i = i+1
            state2keys[i] = (policy["policy"], price["price"])
        end
        keys2state[policy["policy"]] = copy(nest)
    end
    i = 1
    stage2year_phase = Dict()
    year_phase2stage = Dict()
    for i in 1:length(rep_years)
        year = rep_years[i]
        stage2year_phase[2*i-1] = (year,"investment")
        stage2year_phase[2*i] = (year,"operations")

        year_phase2stage[year,"investment"] = 2*i-1
        year_phase2stage[year,"operations"] = 2*i
    end
    return sort(state2keys), sort(keys2state), sort(stage2year_phase), sort(year_phase2stage)
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
    println("  Using $(length(data.rep_years)) typical years")
    println("  Using $(length(data.week_indexes)) typical weeks per year")
    println("  Using $(length(data.hour_indexes)) typical hours per week")

    state2keys, keys2state, stage2year_phase, year_phase2stage = build_dictionnaries(params.policy_proba_df, params.price_proba_df, data.rep_years)

    # Build transition matrices and price distribution
    transition_matrices = build_transition_matrices_new(stage2year_phase, state2keys, keys2state, params.policy_proba_df, params.price_proba_df)
    error_distribution = build_distribution(data.errors)
    
    tech_type = keys(params.tech_dict)
    stor_type = keys(params.stor_dict)


    stor_initial = Dict((stor, lives) => 0 
               for stor in stor_type 
               for lives in 1:(params.stor_dict[stor]["lifetime_new"] + 1))

    tech_initial = Dict((tech, lives) => 0 
        for tech in tech_type 
        for lives in 1:(params.tech_dict[tech]["lifetime_new"] + 1))
    

    for (key, value) in params.stor_dict
        stor_initial[key,value["lifetime_initial"]] = value["initial_capacity[MW_th]"]
    end

    for (key, value) in params.tech_dict
        tech_initial[key,value["lifetime_initial"]] = value["initial_capacity[MW_th]"]
    end

    model = SDDP.MarkovianPolicyGraph(
        transition_matrices=transition_matrices,
        sense=:Min,
        lower_bound=0.0,
        optimizer=Gurobi.Optimizer
    ) do sp, node
        # Elementary information
        stage, state = node
        year, phase = stage2year_phase[stage]
        if stage <= 3
            policy, price = "usual", "mid"
        else
            policy, price = state2keys[state]
        end

        # to remove
        year_start = first(data.rep_years)
        year_evaluate = year
        discount_rate = 0.02
        exponent = year_evaluate - year_start
        final_stage = maximum(keys(stage2year_phase)) - 1
        local df = 1.0 / (1.0 + discount_rate)^exponent

        ###################################### State variables ######################################
        # States variable definition
        @variables(sp, begin
            0 <= X_stor[stor in stor_type, lives in 1:params.stor_dict[stor]["lifetime_new"] + 1] <= 3000,  SDDP.State, (initial_value = stor_initial[(stor, lives)])
            0 <= X_tech[tech in tech_type, lives in 1:params.tech_dict[tech]["lifetime_new"] + 1] <= 3000,  SDDP.State, (initial_value = tech_initial[(tech, lives)])
            0 <= x_demand_mult,                                                                     SDDP.State, (initial_value = 1.0) 
        end)


        ####### Investment Stage #######
        if phase == "investment"
            ##################### 
            # VARIABLES
            #####################
            @variables(sp, begin
                0 <= U_tech[tech in tech_type] <= params.tech_dict[tech]["max_additional_capacity[MW_th]"]
                0 <= U_stor[stor in stor_type] <= params.stor_dict[stor]["max_additional_capacity[MW_th]"]
            end)

            ##################### 
            # CONSTRAINTS
            #####################
            # New assets - add force no 
            @constraint(sp, Stor_Expnd[stor in stor_type], X_stor[stor,params.stor_dict[stor]["lifetime_new"] + 1].out == U_stor[stor])
            @constraint(sp, Tech_Expnd[tech in tech_type], X_tech[tech,params.tech_dict[tech]["lifetime_new"] + 1].out == U_tech[tech])

            # Exisiting assets do not age in investement (they do in operation)
            @constraint(sp, Stor_Aging[stor in stor_type, live in 1:params.stor_dict[stor]["lifetime_new"]], X_stor[stor, live].out == X_stor[stor,live].in)
            @constraint(sp, Tech_Aging[tech in tech_type, live in 1:params.tech_dict[tech]["lifetime_new"]], X_tech[tech, live].out == X_tech[tech,live].in)

            # Multiplier changes - add constraint
            @constraint(sp, Multi, x_demand_mult.in - x_demand_mult.out== 0)

            # Exisiting assets do not exceed maximal allowed
            @constraint(sp, MaxStor[stor in stor_type], sum(X_stor[stor, live].out for live in 2:params.stor_dict[stor]["lifetime_new"] + 1 ) <= params.stor_dict[stor]["max_capacity[MW_th]"])
            @constraint(sp, MaxTech[tech in tech_type], sum(X_tech[tech, live].out for live in 2:params.tech_dict[tech]["lifetime_new"] + 1 ) <= params.tech_dict[tech]["max_capacity[MW_th]"])


            ##################### 
            # OBJECTIVES
            #####################
            # Objective function
            @expression(sp, Invest,  sum(U_tech[tech] * params.tech_dict[tech]["investment_cost[MSEK/MW_th]"] * 1e6 * min(1,(final_stage - stage)/(2 * params.tech_dict[tech]["lifetime_new"])) for tech in tech_type) + 
                                     sum(U_stor[stor] * params.stor_dict[stor]["investment_cost[MSEK/MW_th]"] * 1e6 * min(1,(final_stage - stage)/(2 * params.stor_dict[stor]["lifetime_new"])) for stor in stor_type))
            
            @stageobjective(sp, df * Invest)

        ####### Operational Stage #######
        else
            @assert (phase == "operations")

            ##################### 
            # VARIABLES
            #####################
            @variables(sp, begin
                0 <= u_production[tech in tech_type, week in data.week_indexes, hour in data.hour_indexes]
                0 <= u_charge[stor in stor_type, week in data.week_indexes, hour in data.hour_indexes]
                0 <= u_discharge[stor in stor_type, week in data.week_indexes, hour in data.hour_indexes]
                0 <= u_level[stor in stor_type, week in data.week_indexes, hour in data.hour_indexes]

                0 <= u_unmet[week in data.week_indexes, hour in data.hour_indexes]
            end)

            ##################### 
            # CONSTRAINTS
            #####################
            # All exisiting assets age one time
            @constraint(sp, Stor_Aging[stor in stor_type, live in 2:params.stor_dict[stor]["lifetime_new"] + 1], X_stor[stor, live - 1].out == X_stor[stor,live].in)
            @constraint(sp, Tech_Aging[tech in tech_type, live in 2:params.tech_dict[tech]["lifetime_new"] + 1], X_tech[tech, live - 1].out == X_tech[tech,live].in)
            
            # Multiplier remains same
            @constraint(sp, Multi, x_demand_mult.in - x_demand_mult.out== 0)
            

            # constraint meeting demand - add demand multiplier
            demand = (week, hour, year, price) -> begin
                local df = data.weeks
                row_mask = (df[!, "typical_week"] .== week) .& 
                        (df[!, "hour"] .== hour) .& 
                        (df[!, "year"] .== year) .& 
                        (df[!, "scenario_price"] .== price)
                local matching_rows = df[row_mask, "Load Profile"]
                isempty(matching_rows) && error("No demand data found for week=$week, hour=$hour, year=$year, price=$price")
                return matching_rows[1]
            end
            @constraint(sp, Demand[week in data.week_indexes, hour in data.hour_indexes], 
                            sum(u_production[tech, week, hour] for tech in tech_type)
                          + sum(u_discharge[stor, week, hour] for stor in stor_type)
                          + u_unmet[week,hour]
                         == sum(u_charge[stor, week, hour]/params.stor_dict[stor]["efficiency_th"] for stor in stor_type)
                          + x_demand_mult.in * demand(week, hour, year, price))

            # constraint storage balance
            @constraint(sp, StorageBal[stor in stor_type, week in data.week_indexes, hour in data.hour_indexes],
                            (1 - params.stor_dict[stor]["loss_rate"]) * u_level[stor, week, hour]
                          + u_charge[stor, week, hour] 
                          - u_discharge[stor, week, hour] 
                         == u_level[stor, week, mod(hour+1, length(data.hour_indexes))])
            
            # constraint storage balance
            @constraint(sp, StorageCap[stor in stor_type, week in data.week_indexes, hour in data.hour_indexes],
                        u_level[stor, week, hour] <= sum(X_stor[stor,lives].in for lives in 1:params.stor_dict[stor]["lifetime_new"]))
            
            # constraint storage discharge
            @constraint(sp, BoundDischarge[stor in stor_type, week in data.week_indexes, hour in data.hour_indexes],
                            u_discharge[stor, week, hour] <=  params.stor_dict[stor]["max_discharge_rate"] * sum(X_stor[stor,lives].in for lives in 1:params.stor_dict[stor]["lifetime_new"]))
            # constraint storage charge
            @constraint(sp, BoundCharge[stor in stor_type, week in data.week_indexes, hour in data.hour_indexes],
                            u_charge[stor, week, hour] <=  params.stor_dict[stor]["max_charge_rate"] * sum(X_stor[stor,lives].in for lives in 1:params.stor_dict[stor]["lifetime_new"]))

            # constraint tech power
            @constraint(sp, BoundProd[tech in tech_type, week in data.week_indexes, hour in data.hour_indexes],
                            u_production[tech, week, hour] <= sum(X_tech[tech,lives].in for lives in 1:params.tech_dict[tech]["lifetime_new"]))


            ##################### 
            # OBJECTIVES
            #####################
            # Operation and maintenance
            @expression(sp, FixOpeMaint, (sum(params.tech_dict[tech]["fixed_om[MSEK/MW_th]"] * 1e6 * sum(X_tech[tech,lives].in for lives in 1:params.tech_dict[tech]["lifetime_new"]) for tech in tech_type)
                                        + sum(params.stor_dict[stor]["fixed_om[MSEK/MW_th]"] * 1e6 * sum(X_stor[stor,lives].in for lives in 1:params.stor_dict[stor]["lifetime_new"]) for stor in stor_type)))
            
            # Cost of primary resources
            dyn_cost = (carrier, week, hour) -> begin
                if carrier == "elec"
                    local df = data.weeks
                    row_mask = (df[!, "typical_week"] .== week) .& 
                                (df[!, "hour"] .== hour) .& 
                                (df[!, "year"] .== year) .& 
                                (df[!, "scenario_price"] .== price)
                    if sum(row_mask) == 0
                        error("No electricity price found for week=$week, hour=$hour, year=$year")
                    end
                    return df[row_mask, "price"][1]
                else
                    local df = params.price_df
                    row_mask = (df[!, "carrier"] .== carrier) .& 
                                (df[!, "year"] .== year) .& 
                                (df[!, "price"] .== price)
                    if sum(row_mask) == 0
                        error("No price found for carrier=$carrier, year=$year, price=$price")
                    end
                    return df[row_mask, "value"][1]
                end
            end

            # Emission of primary ressources
            dyn_CO2 = (carrier, year) -> begin
                if carrier == "elec"
                    @assert carrier == "elec"
                    first(params.elec_CO2_df[(params.elec_CO2_df[!, "year"] .== year) ,  "emission_factor"])
                else
                    params.carrier_dict[Symbol(carrier)]["emission_factor"]
                end
            end
            # technology primary cost
            @expression(sp, TechOpeCost[week in data.week_indexes], sum( dyn_cost(params.tech_dict[tech]["energy_carrier"],week,hour) * u_production[tech, week, hour] / params.tech_dict[tech]["efficiency_th"] for tech in tech_type for hour in data.hour_indexes))

            # revenue primary cost - divided by two because it
            @expression(sp, TechOpeRev[week in data.week_indexes],  sum( dyn_cost("elec",week,hour) * u_production[tech, week, hour] / params.tech_dict[tech]["efficiency_th"] * params.tech_dict[tech]["efficiency_el"] / 2 for tech in tech_type for hour in data.hour_indexes))

            # technology variable operation and maintenance cost
            @expression(sp, VarTechOM[week in data.week_indexes], sum( params.tech_dict[tech]["variable_om[SEK/MWh_th]"] * u_production[tech, week, hour] for tech in tech_type for hour in data.hour_indexes))
            @expression(sp, VarStorOM[week in data.week_indexes], sum( params.stor_dict[stor]["variable_om[SEK/MWh_th]"] * u_discharge[stor, week, hour] for stor in stor_type for hour in data.hour_indexes))

            # CO2 cost
            local carbon_price = first(params.carbon_df[(params.carbon_df[!,"policy"] .== policy) .& (params.carbon_df[!,"year"] .== year),"CO2tax"])
            @expression(sp, CO2ope[week in data.week_indexes], sum( dyn_CO2((params.tech_dict[tech]["energy_carrier"]), year) * carbon_price * u_production[tech, week, hour] / params.tech_dict[tech]["efficiency_th"] for tech in tech_type for hour in data.hour_indexes))
            
            # Unmet demand
            @expression(sp, UnmetCost[week in data.week_indexes], sum(params.config_dict[:c_penalty] * u_unmet[week, hour] for hour in data.hour_indexes))

            # Objective function
            @expression(sp, VarOpeCost, 
                sum(
                first(data.week_weights[(data.week_weights[!,"year"] .== (year)) .& (data.week_weights[!,"scenario_price"] .== price), string(week)]) * 
                (UnmetCost[week] + TechOpeCost[week] - TechOpeRev[week] + VarTechOM[week] + VarStorOM[week] + CO2ope[week]) 
                for week in data.week_indexes
            ))

            @stageobjective(sp, df * params.config_dict[:T_years] * (VarOpeCost + FixOpeMaint))
            # Uncertainty definition
            # Add long-term uncertainty on demand
            # Add short-term uncertainty on demand
            # Add short-term uncertainty on price
            # expr_fix_om += params.c_opex_fixed[tech] * cap_vintage_tech[stage][tech].in
            # SDDP.parameterize(sp, [1/1.1, 1, 1.1], [.2, .6, .2]) do ω
            #     JuMP.set_normalized_coefficient(Multi, x_demand_mult.in, ω)
            # end
        end

    end

    return model
end
