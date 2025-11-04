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
    risk_measure=SDDP.CVaR(0.95), iteration_limit=1000,
    n_simulations=400, random_seed=1234)
    println("Training SDDP model...")

    # Train with automatic convergence detection
    # Cut deletion removes weak cuts with very small coefficients to improve numerical stability
    SDDP.train(
        model;
        risk_measure=risk_measure,
        iteration_limit=iteration_limit,
        log_frequency=50,
        parallel_scheme=SDDP.Threaded(),
        cut_deletion_minimum=100,  # Remove weak cuts after iteration 100
    )

    println("Optimal Cost: ", SDDP.calculate_bound(model))

    # Set seed for reproducibility
    Random.seed!(random_seed)

    println("Running simulations...")
    simulation_symbols = [:u_production, :u_expansion_tech, :u_expansion_storage,
        :u_charge, :u_discharge, :u_level, :u_unmet]

    # Add custom recorders for vintage capacity state variables
    custom_recorders = Dict{Symbol,Function}()

    for stage in params.investment_stages
        # Record technology vintage capacities
        tech_symbol = Symbol("cap_vintage_tech_$(stage)")
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

    simulations = SDDP.simulate(model, n_simulations, simulation_symbols;
        parallel_scheme=SDDP.Threaded(),
        custom_recorders=custom_recorders,
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
function export_results(simulations, params::ModelParameters, data::ProcessedData, output_file::String; verbose::Bool=false)
    println("Exporting simulation results to $output_file...")

    # Helper function: print to console only if verbose
    vprint(args...) = verbose && println(args...)

    # Open output file for writing
    io = open(output_file, "w")

    vprint("=== SIMULATION RESULTS (Storage with Representative Weeks) ===")
    println(io, "=== SIMULATION RESULTS (Storage with Representative Weeks) ===")

    for t in 1:(params.T*2)
        sp = simulations[1][t]  # Use 1st simulation for detailed output

        if t % 2 == 1  # Investment stages (odd stages)
            current_year = div(t + 1, 2)
            vprint("Year $current_year - Investment Stage")
            println(io, "Year $current_year - Investment Stage")

            # Print technology expansions and alive capacities
            for tech in params.technologies
                expansion = value(sp[:u_expansion_tech][tech])
                vprint("  Technology: $tech")
                vprint("    Expansion = ", expansion, " MW")
                println(io, "  Technology: $tech")
                println(io, "    Expansion = ", expansion, " MW")

                # Calculate alive capacity using is_alive() function for validation
                alive_cap = 0.0
                vprint("    Vintage Capacities:")
                println(io, "    Vintage Capacities:")

                for stage in params.investment_stages
                    vintage_symbol = Symbol("cap_vintage_tech_$(stage)")
                    if haskey(sp, vintage_symbol)
                        vintage_data = sp[vintage_symbol]
                        if isa(vintage_data, Dict) && haskey(vintage_data, tech)
                            vintage_cap = vintage_data[tech]

                            if vintage_cap > 0.001  # Only print non-zero vintages
                                # Apply is_alive logic consistently for ALL stages (same as model)
                                lifetime_dict = (stage == 0) ? params.c_lifetime_initial : params.c_lifetime_new
                                alive = is_alive(stage, t, lifetime_dict, tech)

                                status = alive ? "alive" : "retired"
                                vprint("      Stage $stage: $(round(vintage_cap, digits=2)) MW [$status]")
                                println(io, "      Stage $stage: $(round(vintage_cap, digits=2)) MW [$status]")

                                if alive
                                    alive_cap += vintage_cap
                                end
                            end
                        end
                    end
                end

                vprint("    Total Alive Capacity = ", round(alive_cap, digits=2), " MW")
                println(io, "    Total Alive Capacity = ", round(alive_cap, digits=2), " MW")
            end

            # Print storage expansion and alive capacity
            storage_expansion = value(sp[:u_expansion_storage])
            vprint("  Storage:")
            vprint("    Expansion = ", storage_expansion, " MWh")
            println(io, "  Storage:")
            println(io, "    Expansion = ", storage_expansion, " MWh")

            # Calculate alive storage capacity using is_storage_alive() function
            alive_storage = 0.0
            storage_lifetime = Int(params.storage_params[:lifetime])
            vprint("    Vintage Capacities:")
            println(io, "    Vintage Capacities:")

            for stage in params.investment_stages
                vintage_symbol = Symbol("cap_vintage_stor_$(stage)")
                if haskey(sp, vintage_symbol)
                    vintage_cap = sp[vintage_symbol]

                    if vintage_cap > 0.001  # Only print non-zero vintages
                        # Apply is_storage_alive logic consistently for ALL stages (same as model)
                        alive = is_storage_alive(stage, t, storage_lifetime)

                        status = alive ? "alive" : "retired"
                        vprint("      Stage $stage: $(round(vintage_cap, digits=2)) MWh [$status]")
                        println(io, "      Stage $stage: $(round(vintage_cap, digits=2)) MWh [$status]")

                        if alive
                            alive_storage += vintage_cap
                        end
                    end
                end
            end

            vprint("    Total Alive Capacity = ", round(alive_storage, digits=2), " MWh")
            println(io, "    Total Alive Capacity = ", round(alive_storage, digits=2), " MWh")

        else  # Operational stages
            vprint("Year $(div(t, 2)) - Operational Stage")
            println(io, "Year $(div(t, 2)) - Operational Stage")

            # Calculate total annual demand across all representative weeks
            annual_demand = 0.0
            for week in 1:data.n_weeks
                week_demand = sum(data.scaled_weeks[week])
                annual_demand += week_demand * data.week_weights_normalized[week]
            end

            vprint("   Annual Demand = ", annual_demand, " MWh")
            println(io, "   Annual Demand = ", annual_demand, " MWh")

            # Calculate storage charge and discharge totals
            total_storage_charge = 0.0
            total_storage_discharge = 0.0
            for week in 1:data.n_weeks
                week_charge = sum(value(sp[:u_charge][week, hour]) for hour in 1:data.hours_per_week)
                week_discharge = sum(value(sp[:u_discharge][week, hour]) for hour in 1:data.hours_per_week)
                total_storage_charge += week_charge * data.week_weights_normalized[week]
                total_storage_discharge += week_discharge * data.week_weights_normalized[week]
            end

            vprint("    Storage charge total = ", total_storage_charge)
            vprint("    Storage discharge total = ", total_storage_discharge)
            println(io, "    Storage charge total = ", total_storage_charge)
            println(io, "    Storage discharge total = ", total_storage_discharge)

            # Storage level at end
            vprint("    Storage level at end = ", value(sp[:u_level][data.n_weeks, data.hours_per_week]))
            println(io, "    Storage level at end = ", value(sp[:u_level][data.n_weeks, data.hours_per_week]))

            # Calculate production by technology and unmet demand
            production_by_tech = Dict{Symbol,Float64}()
            for tech in params.technologies
                tech_production = 0.0
                for week in 1:data.n_weeks
                    week_tech_production = sum(value(sp[:u_production][tech, week, hour]) for hour in 1:data.hours_per_week)
                    tech_production += week_tech_production * data.week_weights_normalized[week]
                end
                production_by_tech[tech] = tech_production
            end

            # Calculate total unmet demand
            total_unmet = 0.0
            for week in 1:data.n_weeks
                week_unmet = sum(value(sp[:u_unmet][week, hour]) for hour in 1:data.hours_per_week)
                total_unmet += week_unmet * data.week_weights_normalized[week]
            end

            # Print production breakdown
            vprint("  Production by Technology:")
            println(io, "  Production by Technology:")
            total_production = 0.0
            for tech in params.technologies
                tech_prod = production_by_tech[tech]
                total_production += tech_prod
                vprint("    $tech: $(round(tech_prod, digits=2)) MWh")
                println(io, "    $tech: $(round(tech_prod, digits=2)) MWh")
            end

            vprint("  Total Production = ", round(total_production, digits=2), " MWh")
            vprint("  Total Unmet Demand = ", round(total_unmet, digits=2), " MWh")
            println(io, "  Total Production = ", round(total_production, digits=2), " MWh")
            println(io, "  Total Unmet Demand = ", round(total_unmet, digits=2), " MWh")
        end
    end

    vprint("=== END SIMULATION RESULTS ===")
    println(io, "=== END SIMULATION RESULTS ===")

    # Close the output file
    close(io)

    vprint("\nDetailed simulation results have been written to '$output_file'")
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

    # Calculate key metrics
    total_costs = [sum(simulations[s][t][:stage_objective] for t in 1:(params.T*2)) for s in 1:length(simulations)]
    mean_cost = mean(total_costs)
    std_cost = std(total_costs)
    cvar_95_cost = quantile(total_costs, 0.95)

    println("\nTotal System Cost (billion SEK / GSEK):")
    println("  Mean: $(round(mean_cost/1e6, digits=2))")
    println("  Std Dev: $(round(std_cost/1e6, digits=2))")
    println("  CVaR 95%: $(round(cvar_95_cost/1e6, digits=2))")

    # Technology investments summary
    println("\nAverage Technology Investments (MW):")
    for tech in params.technologies
        avg_investment = 0.0
        for sim in 1:length(simulations)
            for t in collect(1:2:(params.T*2))
                avg_investment += value(simulations[sim][t][:u_expansion_tech][tech])
            end
        end
        avg_investment /= length(simulations)
        println("  $tech: $(round(avg_investment, digits=1))")
    end

    # Storage investment summary
    avg_storage = 0.0
    for sim in 1:length(simulations)
        for t in collect(1:2:(params.T*2))
            avg_storage += value(simulations[sim][t][:u_expansion_storage])
        end
    end
    avg_storage /= length(simulations)
    println("  Storage: $(round(avg_storage, digits=1)) MWh")

    # Unmet demand analysis
    println("\nUnmet Demand Analysis:")
    for year_idx in 1:params.T
        t = 2 * year_idx
        year = 2010 + year_idx * 10

        unmet_values = []
        for sim in 1:length(simulations)
            total_unmet = 0.0
            for week in 1:data.n_weeks
                week_unmet = sum(value(simulations[sim][t][:u_unmet][week, hour])
                                 for hour in 1:data.hours_per_week)
                total_unmet += week_unmet * data.week_weights_normalized[week]
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
    t = params.T * 2
    storage_utilization = []
    for sim in 1:length(simulations)
        total_discharge = 0.0
        for week in 1:data.n_weeks
            week_discharge = sum(value(simulations[sim][t][:u_discharge][week, hour])
                                 for hour in 1:data.hours_per_week)
            total_discharge += week_discharge * data.week_weights_normalized[week]
        end
        push!(storage_utilization, total_discharge)
    end

    if length(storage_utilization) > 0 && maximum(storage_utilization) > 0
        println("  Mean Annual Discharge: $(round(mean(storage_utilization), digits=1)) MWh")
        println("  Max Annual Discharge: $(round(maximum(storage_utilization), digits=1)) MWh")
    end

    println("\n" * "="^80)
end
