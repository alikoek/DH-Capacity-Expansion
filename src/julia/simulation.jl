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

    println(io, "=== SIMULATION RESULTS (Storage with Representative Weeks) ===")

    state2keys, keys2state, stage2year_phase, year_phase2stage = build_dictionnaries(data.policy_transitions, data.price_transitions, data.rep_years)
    
    sim = simulations[1]
    
    println(io, "="^35, " TECHNOLOGIES ", "="^35)
    line_tech = " "^length("       | existing  ")* "|" * join([rpad(tech, 9) for tech in keys(params.tech_dict)], " | ") * " |"
    println(io, line_tech)
    println(io, "="^80)
    

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
            println(io, tech_current)
            println(io, tech_new)
            println(io, tech_out)
            println(io, "-"^80)
        end
    end

    println(io, "\n\n")
    println(io, "="^35, " STORAGES ", "="^35)
    line_tech = " "^length("       | existing ")*" | " * join([rpad(tech, 8) for tech in keys(params.stor_dict)], " | ")* "|"
    println(io, line_tech)
    println(io, "="^80)

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
            println(io, tech_current)
            println(io, tech_new)
            println(io, tech_out)
            println(io, "-"^80)
        end
    end

    println(io, "\n\n")
    for t in keys(stage2year_phase)
        sp = sim[t]  # Use 1st simulation for detailed output
        state = sp[:node_index][2]
        policy, price = state2keys[state]
        if stage2year_phase[t][2] == "operations"  # Investment stages (odd stages)
            current_year = stage2year_phase[t][1]
            year = current_year
            
            println(io,"="^29 *" OPERATIONS "*string(year) * " " *"="^29)

            println(io,rpad("Weeks",10) *  "|"*join([(lpad(week,8)) *  "|" for week in data.week_indexes])*"  Total |")

            println(io,"-"^75)

            # Production
            for tech in keys(params.tech_dict)
                tech_line = lpad(tech, 10)*"|"
                for week in data.week_indexes
                    prod = sum(sp[:u_production][tech,week,hour] for hour in data.hour_indexes)/ 1e3
                    tech_line  = tech_line * (lpad(round(prod, digits=2), 8)) * "|"
                end
                prod = sum(sp[:u_production][tech,week,hour] * first(data.week_weights[(data.week_weights[!,"year"] .== (year)) .& (data.week_weights[!,"scenario_price"] .== price), string(week)]) for hour in data.hour_indexes for week in data.week_indexes)/ 1e3
                tech_line  = tech_line * (lpad(round(prod, digits=2), 8)) * "|"
                println(io,tech_line)
            end

            # Storage out
            for stor in keys(params.stor_dict)
                tech_line = lpad(stor, 10)*"|"
                for week in data.week_indexes
                    prod = sum(sp[:u_discharge][stor,week,hour] for hour in data.hour_indexes) / 1e3
                    tech_line  = tech_line * (lpad(round(prod, digits=2), 8)) * "|"
                end
                prod = sum(sp[:u_discharge][stor,week,hour] * first(data.week_weights[(data.week_weights[!,"year"] .== (year)) .& (data.week_weights[!,"scenario_price"] .== price), string(week)]) for hour in data.hour_indexes for week in data.week_indexes)/ 1e3
                tech_line  = tech_line * (lpad(round(prod, digits=2), 8)) * "|"
                println(io,tech_line)
            end

            # Unmet demand
            tech_line = lpad("unmet", 10)*"|"
            for week in data.week_indexes
                prod = sum(sp[:u_unmet][week,hour] for hour in data.hour_indexes)/ 1e3
                tech_line  = tech_line * (lpad(round(prod, digits=2), 8)) * "|"
            end
            prod = sum(sp[:u_unmet][week,hour]  * first(data.week_weights[(data.week_weights[!,"year"] .== (year)) .& (data.week_weights[!,"scenario_price"] .== price), string(week)]) for hour in data.hour_indexes for week in data.week_indexes)/ 1e3
            tech_line  = tech_line * (lpad(round(prod, digits=2), 8)) * "|"
            println(io,tech_line)

            println(io,"-"^75)
            
            # Storage In
            for stor in keys(params.stor_dict)
                tech_line = lpad(stor, 10)*"|"
                for week in data.week_indexes
                    prod = sum(sp[:u_charge][stor,week,hour] for hour in data.hour_indexes) / 1e3
                    tech_line  = tech_line * (lpad(round(prod, digits=2), 8)) * "|"
                end
                prod = sum(sp[:u_charge][stor,week,hour] * first(data.week_weights[(data.week_weights[!,"year"] .== (year)) .& (data.week_weights[!,"scenario_price"] .== price), string(week)]) for hour in data.hour_indexes for week in data.week_indexes)/ 1e3
                tech_line  = tech_line * (lpad(round(prod, digits=2), 8)) * "|"
                println(io,tech_line)
            end

            tech_line = lpad("Demand", 10)*"|"
            for week in data.week_indexes
                prod = sum(data.weeks[(data.weeks[!, "typical_week"] .== week) .& (data.weeks[!, "year"] .== year) .& (data.weeks[!, "scenario_price"] .== price), "Load Profile"])
                data.weeks
                tech_line  = tech_line * (lpad(round(prod / 1e3, digits=2), 8)) * "|"
            end
            prod = sum(sum(data.weeks[(data.weeks[!, "typical_week"] .== week) .& (data.weeks[!, "year"] .== year) .& (data.weeks[!, "scenario_price"] .== price), "Load Profile"]) * first(data.week_weights[(data.week_weights[!,"year"] .== (year)) .& (data.week_weights[!,"scenario_price"] .== price), string(week)]) for week in data.week_indexes)
            
            tech_line  = tech_line * (lpad(round(prod / 1e3, digits=2), 8)) * "|"
            println(io,tech_line)
            # println("-"^70)
        end
    end
    
    # Close the file after writing
    println(io, "=== END SIMULATION RESULTS ===")
    close(io)

    content = read(output_file, String)
    println("File content:")
    println(content)


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
