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

function train_model(model;
    risk_measure=SDDP.CVaR(0.95), iteration_limit=100)
    println("Training SDDP model...")
    SDDP.train(model; 
               risk_measure=risk_measure, 
               iteration_limit=iteration_limit, 
               log_frequency=10,

                parallel_scheme = SDDP.Threaded(),
                # custom_recorders =
                #     Dict{Symbol,Function}(:thread_id => sp -> Threads.threadid())
               stopping_rules = [SDDP.BoundStalling(10, 1e-4)]
            #    dashboard = true
               )

    println("Optimal Cost: ", SDDP.calculate_bound(model))

    # Set seed for reproducibility
end

function run_simulation(model;
    n_simulations=400, random_seed=1234)

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


    simulations = SDDP.simulate(model, 
                                n_simulations, 
                                simulation_symbols; 
                                # custom_recorders=custom_recorders,
                                parallel_scheme = SDDP.Threaded(),
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

    println(io, "=== SIMULATION RESULTS (Storage with Representative Periods) ===")

    state2keys, keys2state, stage2year_phase, year_phase2stage = get_encoder_decoder(params.policy_proba_df, params.price_proba_df, params.temperature_proba_df, data.rep_years)

    sim = simulations[1]
    
    println(io, "="^35, " TECHNOLOGIES ", "="^35)
    line_tech = " "^length("       | existing  ")* "|" * join([rpad(string(tech)[1:min(end,9)], 9) for tech in keys(params.tech_dict)], " | ") * " |"
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

            println(io,rpad("Periods",10) *  "|"*join([(lpad(period,8)) *  "|" for period in data.period_indexes])*"  Total |")

            println(io,"-"^75)

            # Production
            for tech in keys(params.tech_dict)
                tech_line = rpad(string(tech)[1:min(end,10)], 10)*"|"
                for period in data.period_indexes
                    prod = sum(sp[:u_production][tech,period,hour] for hour in data.hour_indexes)/ 1e3
                    tech_line  = tech_line * (lpad(round(prod, digits=2), 8)) * "|"
                end
                prod = sum(sp[:u_production][tech,period,hour] * first(data.period_weights[(data.period_weights[!,"year"] .== (year)) .& (data.period_weights[!,"scenario_price"] .== price), string(period)]) for hour in data.hour_indexes for period in data.period_indexes)/ 1e3
                tech_line  = tech_line * (lpad(round(prod, digits=2), 8)) * "|"
                println(io,tech_line)
            end

            # Storage out
            for stor in keys(params.stor_dict)
                tech_line = lpad(stor, 10)*"|"
                for period in data.period_indexes
                    prod = sum(sp[:u_discharge][stor,period,hour] for hour in data.hour_indexes) / 1e3
                    tech_line  = tech_line * (lpad(round(prod, digits=2), 8)) * "|"
                end
                prod = sum(sp[:u_discharge][stor,period,hour] * first(data.period_weights[(data.period_weights[!,"year"] .== (year)) .& (data.period_weights[!,"scenario_price"] .== price), string(period)]) for hour in data.hour_indexes for period in data.period_indexes)/ 1e3
                tech_line  = tech_line * (lpad(round(prod, digits=2), 8)) * "|"
                println(io,tech_line)
            end

            # Unmet demand
            tech_line = lpad("unmet", 10)*"|"
            for period in data.period_indexes
                prod = sum(sp[:u_unmet][period,hour] for hour in data.hour_indexes)/ 1e3
                tech_line  = tech_line * (lpad(round(prod, digits=2), 8)) * "|"
            end
            prod = sum(sp[:u_unmet][period,hour]  * first(data.period_weights[(data.period_weights[!,"year"] .== (year)) .& (data.period_weights[!,"scenario_price"] .== price), string(period)]) for hour in data.hour_indexes for period in data.period_indexes)/ 1e3
            tech_line  = tech_line * (lpad(round(prod, digits=2), 8)) * "|"
            println(io,tech_line)

            println(io,"-"^75)
            
            # Storage In
            for stor in keys(params.stor_dict)
                tech_line = lpad(stor, 10)*"|"
                for period in data.period_indexes
                    prod = sum(sp[:u_charge][stor,period,hour] for hour in data.hour_indexes) / 1e3
                    tech_line  = tech_line * (lpad(round(prod, digits=2), 8)) * "|"
                end
                prod = sum(sp[:u_charge][stor,period,hour] * first(data.period_weights[(data.period_weights[!,"year"] .== (year)) .& (data.period_weights[!,"scenario_price"] .== price), string(period)]) for hour in data.hour_indexes for period in data.period_indexes)/ 1e3
                tech_line  = tech_line * (lpad(round(prod, digits=2), 8)) * "|"
                println(io,tech_line)
            end

            tech_line = lpad("Demand", 10)*"|"
            for period in data.period_indexes
                prod = sum(data.periods[(data.periods[!, "period"] .== period) .& (data.periods[!, "year"] .== year) .& (data.periods[!, "scenario_price"] .== price), "Load Profile"])
                data.periods
                tech_line  = tech_line * (lpad(round(prod / 1e3, digits=2), 8)) * "|"
            end
            prod = sum(sum(data.periods[(data.periods[!, "period"] .== period) .& (data.periods[!, "year"] .== year) .& (data.periods[!, "scenario_price"] .== price), "Load Profile"]) * first(data.period_weights[(data.period_weights[!,"year"] .== (year)) .& (data.period_weights[!,"scenario_price"] .== price), string(period)]) for period in data.period_indexes)
            
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

    state2keys, keys2state, stage2year_phase, year_phase2stage = get_encoder_decoder(params.policy_proba_df, params.price_proba_df, params.temperature_proba_df, data.rep_years)

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
            for period in data.period_indexes
                period_unmet = sum(value(simulations[sim][t][:u_unmet][period, hour])
                                 for hour in data.hour_indexes)
                total_unmet += period_unmet * first(data.period_weights[(data.period_weights[!,"year"] .== (year)) .& (data.period_weights[!,"scenario_price"] .== price), string(period)])
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
            for period in data.period_indexes
                period_discharge = sum(value(simulations[sim][t][:u_discharge][stor,period, hour])
                                    for hour in data.hour_indexes)

                total_discharge += period_discharge * first(data.period_weights[(data.period_weights[!,"year"] .== (year)) .& (data.period_weights[!,"scenario_price"] .== price), string(period)])
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
