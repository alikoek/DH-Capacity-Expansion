"""
Script for analyzing previously saved simulation results

This script demonstrates how to load saved simulation results and perform
various analyses without rerunning the optimization model.

Usage:
    julia examples/analyze_saved_results.jl [filepath]

If no filepath is provided, it will list available saved results and
prompt for selection.
"""

# Add the src directory to the load path
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

using DHCapEx
using Statistics
using Plots
using StatsPlots
using DataFrames
using SDDP  # This includes JuMP

##############################################################################
# Configuration
##############################################################################

# Default output directory for analysis results
ANALYSIS_OUTPUT_DIR = joinpath(dirname(@__DIR__), "output", "analysis")

##############################################################################
# Helper Functions
##############################################################################

"""
    analyze_extreme_event_impacts(simulations, params, data)

Analyze the impact of different extreme event scenarios on costs and operations.
"""
function analyze_extreme_event_impacts(simulations, params, data)
    if !params.enable_extreme_events
        println("⚠ Extreme events were not enabled in this simulation")
        return
    end

    println("\n" * "="^80)
    println("EXTREME EVENT IMPACT ANALYSIS")
    println("="^80)

    # First, print the actual extreme events configuration from params
    if params.extreme_events !== nothing
        println("\nConfigured Extreme Events (from Excel):")
        println("-"^40)
        for row in eachrow(params.extreme_events)
            println("  $(row.scenario): demand=$(row.demand_multiplier), elec=$(row.elec_price_multiplier), dc=$(row.dc_availability), prob=$(row.probability)")
        end
    end

    # Extract extreme event information
    extreme_info = extract_extreme_events_info(simulations, params)

    # Get unique scenarios
    extreme_stage = params.apply_to_year * 2
    unique_scenarios = Dict()
    for (sim_idx, sim) in enumerate(simulations)
        if haskey(sim[extreme_stage], :noise_term)
            noise = sim[extreme_stage][:noise_term]
            key = (noise.demand_mult, noise.elec_price_mult, noise.dc_avail)
            if !haskey(unique_scenarios, key)
                unique_scenarios[key] = []
            end
            push!(unique_scenarios[key], sim_idx)
        end
    end

    # Calculate statistics for each scenario
    println("\nCost Impact by Extreme Event Scenario:")
    println("-"^40)

    scenario_stats = Dict()
    all_costs = get_simulation_costs(simulations)

    for (scenario, sim_indices) in unique_scenarios
        scenario_costs = [all_costs[idx] for idx in sim_indices]
        mean_cost = mean(scenario_costs)
        std_cost = std(scenario_costs)
        max_cost = maximum(scenario_costs)
        min_cost = minimum(scenario_costs)

        demand_mult, elec_mult, dc_avail = scenario
        scenario_name = get_scenario_name(demand_mult, elec_mult, dc_avail)

        scenario_stats[scenario_name] = (
            mean = mean_cost,
            std = std_cost,
            max = max_cost,
            min = min_cost,
            count = length(sim_indices)
        )

        println("\n$scenario_name (n=$(length(sim_indices))):")
        println("  Mean cost: $(round(mean_cost/1e6, digits=2)) GSEK")
        println("  Std dev:  $(round(std_cost/1e6, digits=2)) GSEK")
        println("  Range:    [$(round(min_cost/1e6, digits=2)), $(round(max_cost/1e6, digits=2))] GSEK")
    end

    # Add detailed cost breakdown
    println("\n\nDetailed Cost Breakdown by Scenario:")
    println("="^60)
    analyze_cost_components_by_scenario(simulations, params, data, unique_scenarios)

    return scenario_stats
end

"""
    analyze_cost_components_by_scenario(simulations, params, data, scenario_indices)

Analyze cost components for each extreme event scenario to understand why certain
scenarios are cheaper (e.g., Price Surge being cheapest due to CHP revenue).
"""
function analyze_cost_components_by_scenario(simulations, params, data, scenario_indices)
    extreme_stage = params.apply_to_year * 2

    for (scenario, sim_indices) in scenario_indices
        demand_mult, elec_mult, dc_avail = scenario
        scenario_name = get_scenario_name(demand_mult, elec_mult, dc_avail)

        println("\n$scenario_name Scenario Analysis:")
        println("-"^40)

        # Calculate average cost components for this scenario
        investment_costs = Float64[]
        year2_operational_costs = Float64[]
        year2_unmet_costs = Float64[]

        for idx in sim_indices
            sim = simulations[idx]

            # Investment costs (sum of all investment stages)
            inv_cost = 0.0
            for t in 1:2:(params.T*2-1)  # Investment stages
                inv_cost += sim[t][:stage_objective]
            end
            push!(investment_costs, inv_cost)

            # Year 2 operational costs (where extreme events occur)
            op_cost = sim[extreme_stage][:stage_objective]
            push!(year2_operational_costs, op_cost)

            # Unmet demand penalty for year 2
            unmet_penalty = 0.0
            for week in 1:data.n_weeks
                for hour in 1:data.hours_per_week
                    unmet_val = value(sim[extreme_stage][:u_unmet][week, hour])
                    unmet_penalty += unmet_val * params.c_penalty * data.week_weights_normalized[week]
                end
            end
            push!(year2_unmet_costs, unmet_penalty * params.T_years)
        end

        println("  Investment costs: $(round(mean(investment_costs)/1e6, digits=2)) GSEK")
        println("  Year 2 operational: $(round(mean(year2_operational_costs)/1e6, digits=2)) GSEK")
        println("  Year 2 unmet penalty: $(round(mean(year2_unmet_costs)/1e6, digits=2)) GSEK")

        # Estimate electricity revenue impact (for Price Surge scenario)
        if elec_mult > 1.5
            println("  Note: High electricity prices ($(elec_mult)x) increase CHP revenue")
            println("        This can offset higher heat pump costs")
        end

        if demand_mult > 1.1
            println("  Note: Higher demand ($(demand_mult)x) drives unmet demand costs")
        end

        if dc_avail < 1.0
            println("  Note: DC outage reduces available heat recovery capacity")
        end
    end
end

"""
    get_scenario_name(demand_mult, elec_mult, dc_avail)

Get a human-readable name for an extreme event scenario based on its parameters.
"""
function get_scenario_name(demand_mult::Float64, elec_mult::Float64, dc_avail::Float64)
    # Check most specific conditions first
    # DC Outage: dc_avail = 0.0
    if dc_avail < 0.5  # This catches dc_avail = 0.0
        return "DC Outage"
    # Cold Snap: demand_mult = 1.2 (or higher)
    elseif demand_mult > 1.1  # Changed threshold to catch 1.2
        return "Cold Snap"
    # Price Surge: elec_mult = 2.0 (or higher)
    elseif elec_mult > 1.5  # This catches elec_mult = 2.0
        return "Price Surge"
    # Baseline: all multipliers at 1.0
    elseif demand_mult ≈ 1.0 && elec_mult ≈ 1.0 && dc_avail ≈ 1.0
        return "Baseline"
    # Anything else is a combined event
    else
        return "Combined Event"
    end
end

"""
    create_cost_distribution_plots(simulations, params, data; output_dir)

Create violin plots comparing cost distributions across extreme event scenarios.
"""
function create_cost_distribution_plots(simulations, params, data;
                                      output_dir::String=ANALYSIS_OUTPUT_DIR)
    if !params.enable_extreme_events
        println("⚠ Skipping extreme event plots (not enabled)")
        return
    end

    # Ensure output directory exists
    mkpath(output_dir)

    # Prepare data for plotting
    extreme_stage = params.apply_to_year * 2
    costs_by_scenario = Dict{String, Vector{Float64}}()

    for (sim_idx, sim) in enumerate(simulations)
        total_cost = sum(sim[t][:stage_objective] for t in 1:length(sim))

        if haskey(sim[extreme_stage], :noise_term)
            noise = sim[extreme_stage][:noise_term]
            scenario_name = get_scenario_name(noise.demand_mult, noise.elec_price_mult, noise.dc_avail)

            if !haskey(costs_by_scenario, scenario_name)
                costs_by_scenario[scenario_name] = Float64[]
            end
            push!(costs_by_scenario[scenario_name], total_cost / 1e6)  # Convert to GSEK
        end
    end

    # Create violin plot
    scenarios = collect(keys(costs_by_scenario))
    costs_data = [costs_by_scenario[s] for s in scenarios]

    p = violin(scenarios, costs_data,
               label=false,
               title="Cost Distribution by Extreme Event Scenario",
               xlabel="Scenario",
               ylabel="Total System Cost (GSEK)",
               fillalpha=0.75,
               linewidth=2)

    # Add mean markers
    for (i, scenario) in enumerate(scenarios)
        mean_cost = mean(costs_data[i])
        scatter!([i], [mean_cost], color=:red, markersize=6, label=(i==1 ? "Mean" : ""))
    end

    savefig(p, joinpath(output_dir, "extreme_event_cost_distribution.png"))
    println("✓ Saved cost distribution plot to $(joinpath(output_dir, "extreme_event_cost_distribution.png"))")

    return p
end

"""
    analyze_investment_patterns(simulations, params, data)

Analyze investment patterns across simulations and extreme event scenarios.
"""
function analyze_investment_patterns(simulations, params, data)
    println("\n" * "="^80)
    println("INVESTMENT PATTERN ANALYSIS")
    println("="^80)

    # Calculate average investments by technology
    tech_investments = Dict(tech => Float64[] for tech in params.technologies)
    storage_investments = Float64[]

    for sim in simulations
        for stage in params.investment_stages
            for tech in params.technologies
                push!(tech_investments[tech], value(sim[stage][:u_expansion_tech][tech]))
            end
            push!(storage_investments, value(sim[stage][:u_expansion_storage]))
        end
    end

    println("\nAverage Investments Across All Simulations:")
    println("-"^40)
    println("Note: Values show mean ± std deviation of investments per stage")
    println("      (Stages 1, 3, 5, 7 for T=4). High std indicates varying timing/amounts.")
    println("      Actual investments are never negative.\n")

    for tech in params.technologies
        avg_inv = mean(tech_investments[tech])
        std_inv = std(tech_investments[tech])
        min_inv = minimum(tech_investments[tech])
        max_inv = maximum(tech_investments[tech])
        println("$tech:")
        println("  Mean ± Std: $(round(avg_inv, digits=1)) ± $(round(std_inv, digits=1)) MW")
        println("  Range: [$(round(min_inv, digits=1)), $(round(max_inv, digits=1))] MW")
    end

    avg_stor = mean(storage_investments)
    std_stor = std(storage_investments)
    min_stor = minimum(storage_investments)
    max_stor = maximum(storage_investments)
    println("Storage:")
    println("  Mean ± Std: $(round(avg_stor, digits=1)) ± $(round(std_stor, digits=1)) MWh")
    println("  Range: [$(round(min_stor, digits=1)), $(round(max_stor, digits=1))] MWh")

    # If extreme events enabled, compare investments by scenario
    if params.enable_extreme_events
        println("\n\nInvestment Differences by Extreme Event:")
        println("-"^40)
        println("Note: First-stage investments should be identical across scenarios")
        println("(SDDP makes anticipatory decisions knowing all possible futures)")
        println()

        extreme_stage = params.apply_to_year * 2

        # Group simulations by extreme event
        baseline_sims = Int[]
        extreme_sims = Int[]

        for (sim_idx, sim) in enumerate(simulations)
            if haskey(sim[extreme_stage], :noise_term)
                noise = sim[extreme_stage][:noise_term]
                if noise.demand_mult ≈ 1.0 && noise.elec_price_mult ≈ 1.0 && noise.dc_avail ≈ 1.0
                    push!(baseline_sims, sim_idx)
                else
                    push!(extreme_sims, sim_idx)
                end
            end
        end

        # Compare first-stage investments
        if !isempty(baseline_sims) && !isempty(extreme_sims)
            println("\nFirst-stage investment comparison:")
            for tech in params.technologies
                baseline_inv = mean([value(simulations[i][1][:u_expansion_tech][tech]) for i in baseline_sims])
                extreme_inv = mean([value(simulations[i][1][:u_expansion_tech][tech]) for i in extreme_sims])
                diff_pct = 100 * (extreme_inv - baseline_inv) / max(baseline_inv, 0.001)

                println("  $tech:")
                println("    Baseline: $(round(baseline_inv, digits=1)) MW")
                println("    With extremes: $(round(extreme_inv, digits=1)) MW")
                println("    Difference: $(round(diff_pct, digits=1))%")
            end
        end
    end

    return tech_investments, storage_investments
end

"""
    analyze_operational_metrics(simulations, params, data)

Analyze operational metrics like unmet demand, production mix, and storage utilization.
"""
function analyze_operational_metrics(simulations, params, data)
    println("\n" * "="^80)
    println("OPERATIONAL METRICS ANALYSIS")
    println("="^80)

    # Calculate unmet demand statistics
    unmet_demand_total = Float64[]

    for sim in simulations
        total_unmet = 0.0
        for stage in 2:2:(params.T*2)  # Operational stages only
            for week in 1:data.n_weeks
                week_unmet = sum(value(sim[stage][:u_unmet][week, hour]) for hour in 1:data.hours_per_week)
                total_unmet += week_unmet * data.week_weights_normalized[week]
            end
        end
        push!(unmet_demand_total, total_unmet * params.T_years)
    end

    mean_unmet = mean(unmet_demand_total)
    max_unmet = maximum(unmet_demand_total)
    pct_with_unmet = 100 * sum(unmet_demand_total .> 0.001) / length(unmet_demand_total)

    println("\nUnmet Demand Statistics:")
    println("  Mean total: $(round(mean_unmet, digits=2)) MWh")
    println("  Maximum: $(round(max_unmet, digits=2)) MWh")
    println("  Simulations with unmet demand: $(round(pct_with_unmet, digits=1))%")

    # Add breakdown by year/stage
    println("\nUnmet Demand by Year:")
    for year in 1:params.T
        stage = year * 2  # Operational stage
        year_unmet = Float64[]
        for sim in simulations
            unmet = 0.0
            for week in 1:data.n_weeks
                week_unmet = sum(value(sim[stage][:u_unmet][week, hour]) for hour in 1:data.hours_per_week)
                unmet += week_unmet * data.week_weights_normalized[week]
            end
            push!(year_unmet, unmet * params.T_years)
        end
        println("  Year $year: Mean = $(round(mean(year_unmet), digits=2)) MWh, Max = $(round(maximum(year_unmet), digits=2)) MWh")
    end

    # Add breakdown by extreme event scenario
    if params.enable_extreme_events
        println("\nUnmet Demand by Extreme Event (Year $(params.apply_to_year)):")
        extreme_stage = params.apply_to_year * 2

        # Group by scenario
        scenario_unmet = Dict{String, Vector{Float64}}()
        for (sim_idx, sim) in enumerate(simulations)
            if haskey(sim[extreme_stage], :noise_term)
                noise = sim[extreme_stage][:noise_term]
                scenario_name = get_scenario_name(noise.demand_mult, noise.elec_price_mult, noise.dc_avail)

                # Calculate unmet for this specific year
                unmet = 0.0
                for week in 1:data.n_weeks
                    week_unmet = sum(value(sim[extreme_stage][:u_unmet][week, hour]) for hour in 1:data.hours_per_week)
                    unmet += week_unmet * data.week_weights_normalized[week]
                end

                if !haskey(scenario_unmet, scenario_name)
                    scenario_unmet[scenario_name] = Float64[]
                end
                push!(scenario_unmet[scenario_name], unmet * params.T_years)
            end
        end

        for (scenario, unmets) in scenario_unmet
            println("  $scenario: Mean = $(round(mean(unmets), digits=2)) MWh, Max = $(round(maximum(unmets), digits=2)) MWh")
        end
    end

    # Calculate production mix
    println("\nAverage Production Mix (Year 1):")
    production_by_tech = Dict(tech => 0.0 for tech in params.technologies)

    for sim in simulations
        for tech in params.technologies
            for week in 1:data.n_weeks
                week_prod = sum(value(sim[2][:u_production][tech, week, hour]) for hour in 1:data.hours_per_week)
                production_by_tech[tech] += week_prod * data.week_weights_normalized[week]
            end
        end
    end

    total_production = sum(values(production_by_tech))
    for tech in params.technologies
        avg_prod = production_by_tech[tech] / length(simulations)
        pct = 100 * avg_prod / (total_production / length(simulations))
        println("  $tech: $(round(pct, digits=1))%")
    end

    return unmet_demand_total, production_by_tech
end

##############################################################################
# Main Execution
##############################################################################

function main()
    println("="^80)
    println("Post-Processing Analysis of Saved Simulation Results")
    println("="^80)
    println()

    # Check for command-line argument
    if length(ARGS) > 0
        filepath = ARGS[1]
        if !isfile(filepath)
            error("File not found: $filepath")
        end
    else
        # List available files and prompt for selection
        output_dir = joinpath(dirname(@__DIR__), "output")
        saved_results = list_saved_results(output_dir)

        if isempty(saved_results)
            error("No saved simulation results found in $output_dir")
        end

        print("\nEnter the number of the file to analyze (1-$(length(saved_results))): ")
        selection = parse(Int, readline())

        if selection < 1 || selection > length(saved_results)
            error("Invalid selection")
        end

        filepath = saved_results[selection][2]
    end

    # Load the simulation results
    println("\nLoading simulation results...")
    simulations, params, data, metadata = load_simulation_results(filepath)

    # Display basic information
    println("\n" * "="^80)
    println("SIMULATION OVERVIEW")
    println("="^80)
    println("Timestamp: $(metadata["timestamp"])")
    println("Number of simulations: $(metadata["n_simulations"])")
    println("Planning horizon: $(metadata["planning_horizon_years"]) years")
    println("Technologies: $(join(metadata["technologies"], ", "))")
    if metadata["enable_extreme_events"]
        println("Extreme events: ENABLED at year $(metadata["extreme_events_year"])")
    else
        println("Extreme events: DISABLED")
    end

    # Perform analyses
    println("\nPerforming analyses...")

    # 1. Investment pattern analysis
    tech_investments, storage_investments = analyze_investment_patterns(simulations, params, data)

    # 2. Operational metrics
    unmet_demand, production_mix = analyze_operational_metrics(simulations, params, data)

    # 3. Extreme event impact (if applicable)
    if params.enable_extreme_events
        extreme_stats = analyze_extreme_event_impacts(simulations, params, data)

        # Create visualization
        println("\nGenerating extreme event visualizations...")
        create_cost_distribution_plots(simulations, params, data)
    end

    # 4. Cost statistics
    println("\n" * "="^80)
    println("COST STATISTICS")
    println("="^80)

    costs = get_simulation_costs(simulations)
    println("Mean total cost: $(round(mean(costs)/1e6, digits=2)) GSEK")
    println("Std deviation: $(round(std(costs)/1e6, digits=2)) GSEK")
    println("Min cost: $(round(minimum(costs)/1e6, digits=2)) GSEK")
    println("Max cost: $(round(maximum(costs)/1e6, digits=2)) GSEK")
    println("CVaR 95%: $(round(quantile(costs, 0.95)/1e6, digits=2)) GSEK")

    println("\n" * "="^80)
    println("Analysis Complete!")
    println("="^80)

    if params.enable_extreme_events
        println("\nExtreme event plots saved to: $ANALYSIS_OUTPUT_DIR")
    end

    println("\nYou can now:")
    println("1. Modify this script to add custom analyses")
    println("2. Export specific results to CSV/Excel")
    println("3. Create additional visualizations")
    println("4. Filter and compare specific scenarios")
end

# Run the main function
main()