"""
Model training, simulation, and results export
"""

using SDDP, Random, Statistics

include("helper_functions.jl")

"""
    run_simulation(model, params::ModelParameters, data::ProcessedData;
                  risk_measure=SDDP.CVaR(0.95), iteration_limit=100,
                  n_simulations=400, random_seed=1234)

Train the SDDP model and run simulations.

# Arguments
- `model`: SDDP.PolicyGraph model
- `params::ModelParameters`: Model parameters structure
- `data::ProcessedData`: Processed data structure
- `risk_measure`: Risk measure for training (default: CVaR(0.95))
- `iteration_limit::Int`: Number of training iterations (default: 100)
- `n_simulations::Int`: Number of simulation runs (default: 400)
- `random_seed::Int`: Random seed for reproducibility (default: 1234)

# Returns
- Simulation results
"""
function run_simulation(model, params::ModelParameters, data::ProcessedData;
    risk_measure=SDDP.CVaR(0.95), iteration_limit=100,
    n_simulations=400, random_seed=1234)
    println("Training SDDP model...")
    SDDP.train(model; 
               risk_measure=risk_measure, 
               iteration_limit=iteration_limit, 
               log_frequency=100,
            #    stopping_rules = [SDDP.BoundStalling(10, 1e-4)],
               )

    println("Optimal Cost: ", SDDP.calculate_bound(model))

    # Set seed for reproducibility
    Random.seed!(random_seed)

    println("Running simulations...")
    
    simulation_symbols = [:x_demand_mult, 
                          :X_stor,
                          :X_tech,

                          :U_tech, 
                          :U_stor,

                          :u_production, 
                          :u_charge, 
                          :u_discharge, 
                          :u_level, 
                          :u_unmet]

    # Add custom recorders for vintage capacity state variables
    custom_recorders = Dict{Symbol,Function}()

    for stage in params.investment_stages
        # Record technology vintage capacities
        tech_symbol = Symbol("X_tech_avlb")
        stor_symbol = Symbol("X_stor_avlb")

        custom_recorders[tech_symbol] = (sp) -> begin
            result = Dict{Symbol,Float64}()

            # Search through all variables for matching vintage capacity names
            for var in JuMP.all_variables(sp)
                var_name = JuMP.name(var)
                # Match pattern: cap_vintage_tech_X[Technology]_out
                for tech in params.technologies
                    target_name = "cap_vintage_tech_$(stage)[$(tech)]_out"
                    if var_name == target_name
                        result[tech] = value(var)
                        break
                    end
                end
            end

            # Fill in zeros for any missing technologies
            for tech in params.technologies
                if !haskey(result, tech)
                    result[tech] = 0.0
                end
            end

            return result
        end

        # Record storage vintage capacities
        stor_symbol = Symbol("cap_vintage_stor_$(stage)")
        custom_recorders[stor_symbol] = (sp) -> begin
            # Match pattern: cap_vintage_stor_X[X]_out
            target_name = "cap_vintage_stor_$(stage)[$(stage)]_out"
            for var in JuMP.all_variables(sp)
                if JuMP.name(var) == target_name
                    return value(var)
                end
            end
            return 0.0
        end
    end

    simulations = SDDP.simulate(model, 
                                n_simulations, 
                                simulation_symbols; 
                                # custom_recorders=custom_recorders,
                                skip_undefined_variables=true)

    println("Simulations complete.")

    return simulations
end

"""
    export_results(simulations, params::ModelParameters, data::ProcessedData, output_file::String)

Export detailed simulation results to a text file.

# Arguments
- `simulations`: Simulation results from SDDP.simulate
- `params::ModelParameters`: Model parameters structure
- `data::ProcessedData`: Processed data structure
- `output_file::String`: Path to output file
"""
function export_results(simulations, params::ModelParameters, data::ProcessedData, output_file::String)
    println("Exporting simulation results...")

    # Open output file for writing
    io = open(output_file, "w")

    println("=== SIMULATION RESULTS (Storage with Representative Weeks) ===")
    println(io, "=== SIMULATION RESULTS (Storage with Representative Weeks) ===")

    state2keys, keys2state, stage2year_phase, year_phase2stage = build_dictionnaries(data.policy_transitions, data.price_transitions, data.rep_years)
    
    sim = simulations[1]
    
    println("="^35, " TECHNOLOGIES ", "="^35)
    line_tech = " "^length("       | existing  ")* "|" * join([rpad(tech, 9) for tech in keys(params.tech_dict)], " | ") * " |"
    println(line_tech)
    println("="^80)

    for t in keys(stage2year_phase)
        sp = sim[t]  # Use 1st simulation for detailed output
        if stage2year_phase[t][2] == "investment"  # Investment stages (odd stages)
            current_year = stage2year_phase[t][1]
            year = current_year

            tech_current = "       | existing  | "
            tech_new =     " $year  | installed | "
            tech_out =     "       | shutdown  | "
            
            for tech in keys(params.tech_dict)
                t_install =  sum(sp[:U_tech][tech])
                t_shutdown = sum(sp[:X_tech][tech,1].in)
                t_current =  sum(sp[:X_tech][tech,live].in for live in 1:params.tech_dict[tech]["lifetime_new"])
                
                # Format numbers with 2 decimals and padding for up to 5 digits before decimal
                current_str = lpad(round(t_current, digits=2), 8)  # 8 chars total (5+1+2)
                install_str = lpad(round(t_install, digits=2), 8)
                shutdown_str = lpad(round(t_shutdown, digits=2), 8)
                
                tech_current = tech_current * " " * current_str * " | "
                tech_new = tech_new * "+" * install_str * " | "
                tech_out = tech_out * "-" * shutdown_str * " | "
            end
            println(tech_current)
            println(tech_new)
            println(tech_out)
            println("-"^80)
        end
    end

    println("\n\n\n")
    println("="^35, " STORAGES ", "="^35)
    line_tech = " "^length("       | existing ")*" | " * join([rpad(tech, 8) for tech in keys(params.stor_dict)], " | ")* "|"
    println(line_tech)
    println("="^80)

    for t in keys(stage2year_phase)
        sp = sim[t]  # Use 1st simulation for detailed output
        if stage2year_phase[t][2] == "investment"  # Investment stages (odd stages)
            current_year = stage2year_phase[t][1]
            year = current_year

            tech_current = "       | existing  | "
            tech_new =     " $year  | installed | "
            tech_out =     "       | shutdown  | "
            
            for tech in keys(params.stor_dict)
                t_install =  sum(sp[:U_stor][tech])
                t_shutdown = sum(sp[:X_stor][tech,1].in)
                t_current =  sum(sp[:X_stor][tech,live].in for live in 1:params.stor_dict[tech]["lifetime_new"])
                
                # Format numbers with 2 decimals and padding for up to 5 digits before decimal
                current_str = lpad(round(t_current, digits=2), 8)  # 8 chars total (5+1+2)
                install_str = lpad(round(t_install, digits=2), 8)
                shutdown_str = lpad(round(t_shutdown, digits=2), 8)
                
                tech_current = tech_current * " " * current_str * " | "
                tech_new = tech_new * "+" * install_str * " | "
                tech_out = tech_out * "-" * shutdown_str * " | "
            end
            println(tech_current)
            println(tech_new)
            println(tech_out)
            println("-"^80)
        end
    end
    

    for t in keys(stage2year_phase)
        sp = sim[t]  # Use 1st simulation for detailed output
        if stage2year_phase[t][2] == "operations"  # Investment stages (odd stages)
            current_year = stage2year_phase[t][1]
            year = current_year

            tech_current = "       | existing  | "
            tech_new =     " $year  | installed | "
            tech_out =     "       | shutdown  | "
            
            for tech in keys(params.stor_dict)
                t_install =  sum(sp[:U_stor][tech])
                t_shutdown = sum(sp[:X_stor][tech,1].in)
                t_current =  sum(sp[:X_stor][tech,live].in for live in 1:params.stor_dict[tech]["lifetime_new"])
                
                # Format numbers with 2 decimals and padding for up to 5 digits before decimal
                current_str = lpad(round(t_current, digits=2), 8)  # 8 chars total (5+1+2)
                install_str = lpad(round(t_install, digits=2), 8)
                shutdown_str = lpad(round(t_shutdown, digits=2), 8)
                
                tech_current = tech_current * " " * current_str * " | "
                tech_new = tech_new * "+" * install_str * " | "
                tech_out = tech_out * "-" * shutdown_str * " | "
            end
            println(tech_current)
            println(tech_new)
            println(tech_out)
            println("-"^80)
        end
    end
    #     else  # Operational stages
    #         println("Year $(div(t, 2)) - Operational Stage")
    #         println(io, "Year $(div(t, 2)) - Operational Stage")

    #         # Demand information
    #         println("   Demand Multiplier (in/out) = ", value(sp[:x_demand_mult].in), " / ", value(sp[:x_demand_mult].out))
    #         println(io, "   Demand Multiplier (in/out) = ", value(sp[:x_demand_mult].in), " / ", value(sp[:x_demand_mult].out))

    #         # Calculate total annual demand across all representative weeks
    #         annual_demand = 0.0
    #         for week in 1:data.n_weeks
    #             week_demand = sum(data.scaled_weeks[week]) * value(sp[:x_demand_mult].out)
    #             annual_demand += week_demand * data.week_weights_normalized[week]
    #         end

    #         println("   Annual Demand = ", annual_demand)
    #         println(io, "   Annual Demand = ", annual_demand)

    #         # Calculate storage charge and discharge totals
    #         total_storage_charge = 0.0
    #         total_storage_discharge = 0.0
    #         for week in 1:data.n_weeks
    #             week_charge = sum(value(sp[:u_charge][week, hour]) for hour in 1:data.hours_per_week)
    #             week_discharge = sum(value(sp[:u_discharge][week, hour]) for hour in 1:data.hours_per_week)
    #             total_storage_charge += week_charge * data.week_weights_normalized[week]
    #             total_storage_discharge += week_discharge * data.week_weights_normalized[week]
    #         end

    #         println("    Storage charge total = ", total_storage_charge)
    #         println("    Storage discharge total = ", total_storage_discharge)
    #         println(io, "    Storage charge total = ", total_storage_charge)
    #         println(io, "    Storage discharge total = ", total_storage_discharge)

    #         # Storage level at end
    #         println("    Storage level at end = ", value(sp[:u_level][data.n_weeks, data.hours_per_week]))
    #         println(io, "    Storage level at end = ", value(sp[:u_level][data.n_weeks, data.hours_per_week]))

    #         # Calculate production by technology and unmet demand
    #         production_by_tech = Dict{Symbol, Float64}()
    #         for tech in params.technologies
    #             tech_production = 0.0
    #             for week in 1:data.n_weeks
    #                 week_tech_production = sum(value(sp[:u_production][tech, week, hour]) for hour in 1:data.hours_per_week)
    #                 tech_production += week_tech_production * data.week_weights_normalized[week]
    #             end
    #             production_by_tech[tech] = tech_production
    #         end

    #         # Calculate total unmet demand
    #         total_unmet = 0.0
    #         for week in 1:data.n_weeks
    #             week_unmet = sum(value(sp[:u_unmet][week, hour]) for hour in 1:data.hours_per_week)
    #             total_unmet += week_unmet * data.week_weights_normalized[week]
    #         end

    #         # Print production breakdown
    #         println("  Production by Technology:")
    #         println(io, "  Production by Technology:")
    #         total_production = 0.0
    #         for tech in params.technologies
    #             tech_prod = production_by_tech[tech]
    #             total_production += tech_prod
    #             println("    $tech: $(round(tech_prod, digits=2)) MWh")
    #             println(io, "    $tech: $(round(tech_prod, digits=2)) MWh")
    #         end

    #         println("  Total Production = ", round(total_production, digits=2), " MWh")
    #         println("  Total Unmet Demand = ", round(total_unmet, digits=2), " MWh")
    #         println(io, "  Total Production = ", round(total_production, digits=2), " MWh")
    #         println(io, "  Total Unmet Demand = ", round(total_unmet, digits=2), " MWh")
    #     end
    # end

    println("=== END SIMULATION RESULTS ===")
    println(io, "=== END SIMULATION RESULTS ===")

    # Close the output file
    close(io)

    println("\nDetailed simulation results have been written to \'$output_file\'")
end

"""
    print_summary_statistics(simulations, params::ModelParameters, data::ProcessedData)

Print summary statistics from simulations.

# Arguments
- `simulations`: Simulation results from SDDP.simulate
- `params::ModelParameters`: Model parameters structure
- `data::ProcessedData`: Processed data structure
"""
function print_summary_statistics(simulations, params::ModelParameters, data::ProcessedData)
    println("\n" * "="^80)
    println("SUMMARY STATISTICS")
    println("="^80)

    state2keys, keys2state, stage2year_phase, year_phase2stage = build_dictionnaries(data.policy_transitions, data.price_transitions, data.rep_years)
    
    all_stages = keys(stage2year_phase)
    inv_stages = [state for (state, dept) in stage2year_phase if dept[2] == "investment"]
    ope_stages = [state for (state, dept) in stage2year_phase if dept[2] == "operations"]


    # Calculate key metrics
    total_costs = [sum(simulations[s][t][:stage_objective] for t in all_stages) for s in 1:length(simulations)]
    mean_cost = mean(total_costs)
    std_cost = std(total_costs)
    cvar_95_cost = quantile(total_costs, 0.95)

    println("\nTotal System Cost (€ million):")
    println("  Mean: $(round(mean_cost/1e6, digits=2))")
    println("  Std Dev: $(round(std_cost/1e6, digits=2))")
    println("  CVaR 95%: $(round(cvar_95_cost/1e6, digits=2))")

    # Technology investments summary
    println("\nAverage Technology Investments (MW):")
    for tech in keys(params.tech_dict)
        avg_investment = 0.0
        for sim in 1:length(simulations)
            for t in inv_stages
                avg_investment += value(simulations[sim][t][:U_tech][tech])
            end
        end
        avg_investment /= length(simulations)
        println("  $tech: $(round(avg_investment, digits=1))")
    end

    # Storage investment summary
    println("\nAverage Storage Investments (MW):")
    for stor in keys(params.stor_dict)
        avg_storage = 0.0
        for sim in 1:length(simulations)
            for t in inv_stages
                avg_storage += value(simulations[sim][t][:U_stor][stor])
            end
        end
        avg_storage /= length(simulations)
        println("  Storage: $(round(avg_storage, digits=1)) MWh")
    end

    # Unmet demand analysis
    println("\nUnmet Demand Analysis:")
    for t in ope_stages
        year = stage2year_phase[t][1]

        unmet_values = []
        for sim in 1:length(simulations)
            state = simulations[sim][t][:node_index][2]
            policy, price = state2keys[state]
            total_unmet = 0.0
            for week in data.week_indexes
                week_unmet = sum(value(simulations[sim][t][:u_unmet][week, hour])
                                 for hour in data.hour_indexes)
                total_unmet += week_unmet * first(data.week_weights[(data.week_weights[!,"year"] .== (year)) .& (data.week_weights[!,"scenario_price"] .== price), string(week)])
            end
            push!(unmet_values, total_unmet)
        end

        mean_unmet = mean(unmet_values)
        max_unmet = maximum(unmet_values)

        println("  Year $year:")
        println("    Mean: $(round(mean_unmet, digits=2)) MWh")
        println("    Max: $(round(max_unmet, digits=2)) MWh")
    end

    # Storage utilization
    println("\nStorage Utilization (Year 2050):")
    t = length(keys(stage2year_phase))
    year = stage2year_phase[t][1]
    storage_utilization = []
    for stor in keys(params.stor_dict)
        for sim in 1:length(simulations)
            state = simulations[sim][t][:node_index][2]
            policy, price = state2keys[state]

            total_discharge = 0.0
            for week in data.week_indexes
                week_discharge = sum(value(simulations[sim][t][:u_discharge][stor,week, hour])
                                    for hour in data.hour_indexes)

                total_discharge += week_discharge * first(data.week_weights[(data.week_weights[!,"year"] .== (year)) .& (data.week_weights[!,"scenario_price"] .== price), string(week)])
            end
            push!(storage_utilization, total_discharge)
        end

        if length(storage_utilization) > 0 && maximum(storage_utilization) > 0
            println("  Mean Annual Discharge $(stor): $(round(mean(storage_utilization), digits=1)) MWh")
            println("  Max Annual Discharge $(stor): $(round(maximum(storage_utilization), digits=1)) MWh")
        end
    end
    println("\n" * "="^80)
end
