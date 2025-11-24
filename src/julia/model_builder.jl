"""
SDDP model construction for District Heating Capacity Expansion
"""

using SDDP, Gurobi, LinearAlgebra, Distributions

using Plots
using ColorSchemes

include("helper_functions.jl")

"""
    build_transition_matrices(T::Int)

Build Markovian transition matrices for the policy graph.

# Arguments
- `T::Int`: Number of model years

# Returns
- Vector of transition matrices
"""

function build_transition_matrices_new(stage2year_phase, state2keys, keys2state, p_policy, p_price, p_temperature)
    start_price = "mid"
    start_policy = "usual"
    start_temperature = "highT"
    n_states = length(state2keys)
    
    branch_temperature = (stage) -> stage == 1
    branch_carbon = (stage) -> stage == 3
    branch_price = (stage) -> stage2year_phase[stage][2] == "operations"

    # Start Matrix: creating 18 and pointing towards the right stage
    # Matrix to temperature
    A1 = zeros(n_states)
    stage = 1
    for state2 in 1:n_states
        policy2, price2, temperature2 = state2keys[state2]
        if branch_temperature(stage)
            p1 = p_temperature[p_temperature.temp .== temperature2, :probability][1]
        else
            p1 = start_temperature == temperature2
        end

        if branch_carbon(stage)
            p2 = p_policy[p_policy.policy .== policy2, :probability][1]
        else
            p2 = start_policy == policy2
        end
        
        if branch_price(stage)
            p3 = p_price[p_price.price .== start_price, price2][1]
        else
            p3 = start_price == price2
        end
        A1[state2] = p1 * p2 * p3
    end

    function create_matrix(stage)
        C = zeros(n_states,n_states)
        for state1 in 1:n_states
            for state2 in 1:n_states
                policy1, price1, temperature1 = state2keys[state1]
                policy2, price2, temperature2 = state2keys[state2]
                if branch_temperature(stage)
                    p1 = p_temperature[p_temperature.temp .== temperature2, :probability][1]
                else
                    p1 = temperature1 == temperature2
                end

                if branch_carbon(stage)
                    p2 = p_policy[p_policy.policy .== policy2, :probability][1]
                else
                    p2 = policy1 == policy2
                end
                
                if branch_price(stage)
                    p3 = p_price[p_price.price .== price1, price2][1]
                else
                    p3 = price1 == price2
                end
                
                C[state1,state2] = p1 * p2 * p3
            end
        end
        return C
    end
    
    transition_matrices = Array{Float64,2}[]
    for stage in 1:length(stage2year_phase)
        if stage == 1
            push!(transition_matrices, A1')
        else
            push!(transition_matrices, create_matrix(stage))
        end
    end
    # for matrix in transition_matrices
    #     println("Matrix:")
    #     for row in eachrow(matrix)
    #         println(row)
    #     end
    # end
    p = plot_adjacency_matrices(transition_matrices, state2keys,stage2year_phase)
    display(p)
    return transition_matrices
end


"""
plot_adjacency_matrices(model, state2keys)

`model`      :: Vector{Matrix{Float64}}  (adjacency matrices stage→stage+1)
`state2keys` :: OrderedDict{Int,(policy, price, temperature)}
"""
function plot_adjacency_matrices(model::Vector{Matrix{Float64}}, state2keys,stage2year_phase)

    n_stages = length(model)
    p = plot(size=(1200, 800), legend=false,
             xlim=(0, n_stages), ylim=(0, maximum(size.(model,1)) + 1))

    title!("Adjacency Matrix Flow Between Stages")
    xlabel!("Stage")
    ylabel!("Node index")

    # --- Visual encoding dictionaries ---
    T_colors = Dict("highT" => "red", "lowT" => "blue")

    prices = ["low", "mid", "high"]
    price_color = Dict("low" => "yellow", "mid" => "pink", "high" => "purple")

    policies = ["weak", "usual", "strong"]
    shapes = Dict("weak" => :square, "usual" => :pentagon, "strong" => :hexagon)

    # --- 1. Draw transitions (arrows) ---
    for stage in 1:(n_stages-1)
        mat = model[stage]
        n_rows, n_cols = size(mat)

        for i in 1:n_rows
            for j in 1:n_cols
                value = mat[i,j]
                if value > 0
                    grey = RGB((1-value)^2, (1-value)^2, (1-value)^4)

                    plot!([stage-1, stage], [i, j],
                          arrow=true,
                          color=grey,
                          linewidth=1.5 * value)
                end
            end
        end
    end

    # --- 2. Scatter nodes based on state2keys ---
    
    for stage in 1:n_stages
        n_nodes = size(model[stage], 1)
        idx = 1
        for local_idx in 1:n_nodes
            policy, price, temp = state2keys[idx]
            scatter!([stage-1], [local_idx];
                markersize=9,
                markershape=get(shapes, policy, :circle),
                color=get(price_color, price, :black),
                markerstrokecolor=get(T_colors, temp, :black),
                markerstrokewidth=2
            )

            idx += 1
        end
    end

    # --- Stage labels ---
    for stage in 1:(n_stages)
        year,phase = stage2year_phase[stage]
        annotate!(stage-1, 0.0, text("$year \n $phase", 6, :center, :bottom))
    end

    return p
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
    println("  Using $(length(data.period_indexes)) typical periods per year")
    println("  Using $(length(data.hour_indexes)) typical hours per period")

    state2keys, keys2state, stage2year_phase, year_phase2stage = get_encoder_decoder(params.policy_proba_df, params.price_proba_df, params.temperature_proba_df, data.rep_years)

    # Build transition matrices and price distribution
    transition_matrices = build_transition_matrices_new(stage2year_phase, state2keys, keys2state, params.policy_proba_df, params.price_proba_df, params.temperature_proba_df)
    error_distribution = build_distribution(data.errors)
    
    tech_type = keys(params.tech_dict)
    stor_type = keys(params.stor_dict)

    stor_initial = Dict((stor, lives) => 0.0 
               for stor in stor_type 
               for lives in 1:(params.stor_dict[stor]["lifetime_new"] + 1))

    tech_initial = Dict((tech, lives) => 0.0 
        for tech in tech_type 
        for lives in 1:(params.tech_dict[tech]["lifetime_new"] + 1))
    
    for row in eachrow(params.existing_capa_df)
        tech = Symbol(row.technology)
        lives = row.lives
        capacity = row.capacity
        tech_initial[tech, lives] = capacity
    end

    for (key, value) in params.stor_dict
        stor_initial[key,value["lifetime_initial"]] = value["initial_capacity[MW_th]"]
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
        policy, price, temperature = state2keys[state]
        if stage <= 2
            policy, price = "usual", "mid"
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
                0 <= u_production[tech in tech_type, period in data.period_indexes, hour in data.hour_indexes]
                0 <= u_charge[stor in stor_type, period in data.period_indexes, hour in data.hour_indexes]
                0 <= u_discharge[stor in stor_type, period in data.period_indexes, hour in data.hour_indexes]
                0 <= u_level[stor in stor_type, period in data.period_indexes, hour in data.hour_indexes]
                0 <= u_unmet[period in data.period_indexes, hour in data.hour_indexes]
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
            demand = (period, hour, year, price) -> begin
                local df = data.periods
                row_mask = (df[!, "period"] .== period) .& 
                        (df[!, "hour"] .== hour) .& 
                        (df[!, "year"] .== year) .& 
                        (df[!, "scenario_price"] .== price)
                local matching_rows = df[row_mask, "Load Profile"]
                isempty(matching_rows) && error("No demand data found for period=$period, hour=$hour, year=$year, price=$price")
                return matching_rows[1]
            end
            @constraint(sp, Demand[period in data.period_indexes, hour in data.hour_indexes], 
                            sum(u_production[tech, period, hour] for tech in tech_type)
                          + sum(u_discharge[stor, period, hour] for stor in stor_type)
                          + u_unmet[period,hour]
                         == sum(u_charge[stor, period, hour]/params.stor_dict[stor]["efficiency_th"] for stor in stor_type)
                          + x_demand_mult.in * demand(period, hour, year, price))

            # constraint storage balance
            @constraint(sp, StorageBal[stor in stor_type, period in data.period_indexes, hour in data.hour_indexes],
                            (1 - params.stor_dict[stor]["loss_rate"]) * u_level[stor, period, hour]
                          + u_charge[stor, period, hour] 
                          - u_discharge[stor, period, hour] 
                         == u_level[stor, period, mod(hour+1, length(data.hour_indexes))])
            
            # constraint storage balance
            @constraint(sp, StorageCap[stor in stor_type, period in data.period_indexes, hour in data.hour_indexes],
                        u_level[stor, period, hour] <= sum(X_stor[stor,lives].in for lives in 1:params.stor_dict[stor]["lifetime_new"]))
            
            # constraint storage discharge
            @constraint(sp, BoundDischarge[stor in stor_type, period in data.period_indexes, hour in data.hour_indexes],
                            u_discharge[stor, period, hour] <=  params.stor_dict[stor]["max_discharge_rate"] * sum(X_stor[stor,lives].in for lives in 1:params.stor_dict[stor]["lifetime_new"]))
            
                            # constraint storage charge
            @constraint(sp, BoundCharge[stor in stor_type, period in data.period_indexes, hour in data.hour_indexes],
                            u_charge[stor, period, hour] <=  params.stor_dict[stor]["max_charge_rate"] * sum(X_stor[stor,lives].in for lives in 1:params.stor_dict[stor]["lifetime_new"]))

            # constraint tech power
            @constraint(sp, BoundProd[tech in tech_type, period in data.period_indexes, hour in data.hour_indexes],
                            u_production[tech, period, hour] <= sum(X_tech[tech,lives].in for lives in 1:params.tech_dict[tech]["lifetime_new"]))
            
            
            # constraint - waste availability (only for carriers with limits)
            for carrier in String.(keys(params.carrier_dict))
                limitation_rows = params.limit_ressource_df[
                    (params.limit_ressource_df[!, "carrier"] .== carrier) .&
                    (params.limit_ressource_df[!, "year"] .== year),
                    "limitation"
                ]
                
                if !isempty(limitation_rows)
                    @constraint(sp, 
                        sum(
                            first(data.period_weights[
                                (data.period_weights[!, "year"] .== year) .& 
                                (data.period_weights[!, "scenario_price"] .== price), 
                                string(period)
                            ]) *
                            u_production[tech, period, hour] / 
                            first(
                                params.temperature_efficiency_df[
                                    (params.temperature_efficiency_df[!, "technology"] .== string(tech)) .&
                                    (params.temperature_efficiency_df[!, "temperature"] .== temperature) .&
                                    (params.temperature_efficiency_df[!, "year"] .== year),
                                    "efficiency"
                                ]
                            )
                            for period in data.period_indexes 
                            for hour in data.hour_indexes
                            for tech in keys(params.tech_dict) if params.tech_dict[tech]["energy_carrier"] == carrier
                        ) <= first(limitation_rows)
                    )
                end
                # No constraint created if no limitation exists (effectively unbounded)
            end

            ##################### 
            # OBJECTIVES
            #####################
            # Operation and maintenance
            @expression(sp, FixOpeMaint, (sum(params.tech_dict[tech]["fixed_om[MSEK/MW_th]"] * 1e6 * sum(X_tech[tech,lives].in for lives in 1:params.tech_dict[tech]["lifetime_new"]) for tech in tech_type)
                                        + sum(params.stor_dict[stor]["fixed_om[MSEK/MW_th]"] * 1e6 * sum(X_stor[stor,lives].in for lives in 1:params.stor_dict[stor]["lifetime_new"]) for stor in stor_type)))
            
            # Cost of primary resources
            dyn_cost = (carrier, period, hour) -> begin
                if carrier == "elec"
                    local df = data.periods
                    row_mask = (df[!, "period"] .== period) .& 
                                (df[!, "hour"] .== hour) .& 
                                (df[!, "year"] .== year) .& 
                                (df[!, "scenario_price"] .== price)
                    if sum(row_mask) == 0
                        error("No electricity price found for period=$period, hour=$hour, year=$year")
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
            @expression(sp, TechOpeCost[period in data.period_indexes], 
                sum( 
                    dyn_cost(params.tech_dict[tech]["energy_carrier"], period, hour) * 
                    u_production[tech, period, hour] / 
                    first(
                        params.temperature_efficiency_df[
                            (params.temperature_efficiency_df[!, "technology"] .== string(tech)) .&
                            (params.temperature_efficiency_df[!, "temperature"] .== temperature) .&
                            (params.temperature_efficiency_df[!, "year"] .== year),
                            "efficiency"
                        ]
                    )
                    for tech in tech_type 
                    for hour in data.hour_indexes
                )
            )

            # revenue primary cost - divided by two because it makes more sense (Transport & Distribution fees)
            @expression(sp, TechOpeRev[period in data.period_indexes], 
                sum( 
                    dyn_cost("elec",period,hour) / 2 * 
                    u_production[tech, period, hour] * 
                    params.tech_dict[tech]["efficiency_el"] / params.tech_dict[tech]["efficiency_th"]
                    for tech in tech_type 
                    for hour in data.hour_indexes
                )
            )
            
            # technology variable operation and maintenance cost
            @expression(sp, VarTechOM[period in data.period_indexes], sum( params.tech_dict[tech]["variable_om[SEK/MWh_th]"] * u_production[tech, period, hour] for tech in tech_type for hour in data.hour_indexes))
            @expression(sp, VarStorOM[period in data.period_indexes], sum( params.stor_dict[stor]["variable_om[SEK/MWh_th]"] * u_discharge[stor, period, hour] for stor in stor_type for hour in data.hour_indexes))

            # CO2 cost
            local carbon_price = first(params.carbon_df[(params.carbon_df[!,"policy"] .== policy) .& (params.carbon_df[!,"year"] .== year),"CO2tax"])
            @expression(sp, CO2ope[period in data.period_indexes], sum( dyn_CO2((params.tech_dict[tech]["energy_carrier"]), year) * carbon_price * u_production[tech, period, hour] / params.tech_dict[tech]["efficiency_th"] for tech in tech_type for hour in data.hour_indexes))
            
            # Unmet demand
            @expression(sp, UnmetCost[period in data.period_indexes], sum(params.config_dict[:c_penalty] * u_unmet[period, hour] for hour in data.hour_indexes))

            # Objective function
            @expression(sp, VarOpeCost, 
                sum(
                first(data.period_weights[(data.period_weights[!,"year"] .== (year)) .& (data.period_weights[!,"scenario_price"] .== price), string(period)]) * 
                (UnmetCost[period] + TechOpeCost[period] - TechOpeRev[period] + VarTechOM[period] + VarStorOM[period] + CO2ope[period]) 
                for period in data.period_indexes
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

