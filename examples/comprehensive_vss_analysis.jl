"""
Comprehensive VSS Analysis Framework

This script provides multiple ways to calculate and demonstrate the value
of the stochastic solution, including:
1. Standard VSS (EEV - RP)
2. Risk-Adjusted VSS with CVaR
3. Conditional VSS by extreme event
4. Comparison with simpler models
5. Solution quality metrics

Usage:
    julia --threads auto examples/comprehensive_vss_analysis.jl [options]

Options (set in script):
    - RUN_STANDARD_VSS: true/false
    - RUN_RISK_ADJUSTED: true/false
    - RUN_CONDITIONAL_VSS: true/false
    - RUN_SIMPLE_MODELS: true/false
    - RUN_SOLUTION_QUALITY: true/false
"""

# Add the src directory to the load path
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

using DHCapEx
using SDDP
using Statistics
using Plots
using StatsPlots
using DataFrames
using Printf
using Dates
using Random

##############################################################################
# Configuration - Choose which analyses to run
##############################################################################

# Analysis options (set to true/false to enable/disable)
RUN_STANDARD_VSS = true      # Standard VSS calculation
RUN_RISK_ADJUSTED = true     # Risk-adjusted VSS with CVaR
RUN_CONDITIONAL_VSS = true   # VSS conditional on extreme events
RUN_SIMPLE_MODELS = false    # Compare with two-stage and single-stage (slower)
RUN_SOLUTION_QUALITY = true  # Non-cost metrics (unmet demand, etc.)

# Model parameters
ITERATION_LIMIT_SDDP = 10    # For SDDP training (increase to 300-500 for production)
ITERATION_LIMIT_DET = 100     # For deterministic training
N_SIMULATIONS = 500           # Monte Carlo simulations (minimum 100, recommend 500-1000)
RANDOM_SEED = 1234
RISK_LEVELS = [0.95]  # CVaR confidence levels to test (add [0.90, 0.99] for production)

# File paths
project_dir = dirname(@__DIR__)
data_dir = joinpath(project_dir, "data")
output_dir = joinpath(project_dir, "output", "vss_analysis")
excel_file = joinpath(data_dir, "model_parameters.xlsx")

# Create output directory
mkpath(output_dir)

##############################################################################
# Helper Functions
##############################################################################

"""
Calculate risk metrics for a cost distribution
"""
function calculate_risk_metrics(costs::Vector{Float64})
    metrics = Dict{String,Float64}()
    metrics["mean"] = mean(costs)
    metrics["std"] = std(costs)
    metrics["min"] = minimum(costs)
    metrics["max"] = maximum(costs)

    for α in RISK_LEVELS
        metrics["cvar_$(Int(100α))"] = quantile(costs, α)
    end

    return metrics
end

"""
Extract solution characteristics (technology mix, timing, etc.)
"""
function extract_solution_characteristics(simulations, params, data)
    characteristics = Dict{String,Any}()

    # Average first-stage investments
    first_stage_inv = Dict{Symbol,Float64}()
    for tech in params.technologies
        investments = [value(sim[1][:u_expansion_tech][tech]) for sim in simulations]
        first_stage_inv[tech] = mean(investments)
    end
    characteristics["first_stage_investments"] = first_stage_inv

    # Storage investment timing
    storage_by_stage = Float64[]
    for stage in params.investment_stages
        storage_inv = [value(sim[stage][:u_expansion_storage]) for sim in simulations]
        push!(storage_by_stage, mean(storage_inv))
    end
    characteristics["storage_timing"] = storage_by_stage

    # Total unmet demand
    total_unmet = Float64[]
    for sim in simulations
        unmet = 0.0
        for t in 2:2:(params.T*2)  # Operational stages
            if haskey(sim[t], :u_unmet)
                # Sum over all weeks and hours
                for week in 1:data.n_weeks
                    for hour in 1:data.hours_per_week
                        unmet += value(sim[t][:u_unmet][week, hour])
                    end
                end
            end
        end
        push!(total_unmet, unmet)
    end
    characteristics["unmet_demand"] = mean(total_unmet)
    characteristics["unmet_demand_std"] = std(total_unmet)
    characteristics["prob_unmet"] = sum(total_unmet .> 0.001) / length(total_unmet)

    return characteristics
end

##############################################################################
# Main Analysis Functions
##############################################################################

"""
Standard VSS calculation: VSS = EEV - RP
"""
function calculate_standard_vss(params, data)
    println("\n" * "="^80)
    println("STANDARD VSS CALCULATION")
    println("="^80)

    # Build and solve SDDP model (RP)
    println("\n1. Training SDDP model (Recourse Problem)...")
    println("   Building model...")
    sddp_model = build_sddp_model(params, data)
    println("   Training with $ITERATION_LIMIT_SDDP iterations...")
    SDDP.train(sddp_model;
        risk_measure=SDDP.Expectation(),
        iteration_limit=ITERATION_LIMIT_SDDP,
        parallel_scheme=SDDP.Threaded(),
        print_level=1,
        log_frequency=10
    )
    println("   ✓ SDDP training complete")

    println("\n2. Running SDDP simulations...")
    println("   Running $N_SIMULATIONS Monte Carlo simulations...")
    # CRITICAL: Set random seed for reproducibility and fair VSS comparison
    # SDDP and EEV must be evaluated on IDENTICAL scenarios
    Random.seed!(RANDOM_SEED)
    sddp_simulations = SDDP.simulate(sddp_model, N_SIMULATIONS,
        [:u_production, :u_expansion_tech, :u_expansion_storage, :u_unmet])
    println("   ✓ SDDP simulations complete")

    println("   Calculating costs...")
    sddp_costs = get_simulation_costs(sddp_simulations)
    RP = mean(sddp_costs)
    println("   ✓ RP (SDDP mean cost): $(round(RP/1e6, digits=2)) GSEK")

    # Build and solve deterministic model (EV)
    println("\n3. Building deterministic (EV) model...")
    ev_model, ev_variables = build_deterministic_model(params, data)
    println("   ✓ Deterministic model solved")

    println("\n4. Extracting EV investments...")
    ev_investments = extract_ev_investments(ev_model, ev_variables, params)
    println("   ✓ EV investments extracted")

    # Evaluate EV policy (EEV)
    println("\n5. Evaluating EV policy under uncertainty...")
    eev_simulations, eev_mean, eev_std = evaluate_ev_policy(sddp_model, ev_investments, params, N_SIMULATIONS;
        random_seed=RANDOM_SEED)
    println("   ✓ EV policy evaluation complete")

    println("   Calculating EEV costs...")
    eev_costs = get_simulation_costs(eev_simulations)
    EEV = mean(eev_costs)
    println("   ✓ EEV (EV policy cost): $(round(EEV/1e6, digits=2)) GSEK")

    # Calculate VSS
    VSS = EEV - RP
    relative_VSS = 100 * VSS / RP

    # Store results
    results = Dict(
        "RP" => RP,
        "EEV" => EEV,
        "VSS" => VSS,
        "VSS_percent" => relative_VSS,
        "sddp_costs" => sddp_costs,
        "eev_costs" => eev_costs,
        "sddp_simulations" => sddp_simulations,
        "eev_simulations" => eev_simulations
    )

    println("\n" * "-"^40)
    println("Standard VSS Results:")
    println("  RP (SDDP mean cost): $(round(RP/1e6, digits=2)) GSEK")
    println("  EEV (EV policy cost): $(round(EEV/1e6, digits=2)) GSEK")
    println("  VSS: $(round(VSS/1e3, digits=2)) MSEK ($(round(relative_VSS, digits=2))%)")

    return results
end

"""
Risk-Adjusted VSS: Compare tail performance
"""
function calculate_risk_adjusted_vss(standard_results)
    println("\n" * "="^80)
    println("RISK-ADJUSTED VSS ANALYSIS")
    println("="^80)

    sddp_costs = standard_results["sddp_costs"]
    eev_costs = standard_results["eev_costs"]

    results = Dict{String,Any}()

    for α in RISK_LEVELS
        # Calculate VaR (Value-at-Risk = α-quantile, e.g., 95th percentile)
        sddp_var = quantile(sddp_costs, α)
        eev_var = quantile(eev_costs, α)

        # Calculate CVaR (Conditional Value-at-Risk = tail mean above VaR threshold)
        sddp_tail = sddp_costs[sddp_costs.>=sddp_var]
        eev_tail = eev_costs[eev_costs.>=eev_var]

        sddp_cvar = mean(sddp_tail)  # This is the actual CVaR
        eev_cvar = mean(eev_tail)

        # Risk-adjusted VSS using CVaR (tail mean)
        ravss_cvar = eev_cvar - sddp_cvar
        ravss_var = eev_var - sddp_var  # Alternative: VSS at VaR threshold
        relative_ravss_cvar = 100 * ravss_cvar / sddp_cvar

        results["cvar_$(Int(100α))"] = Dict(
            "sddp_var" => sddp_var,        # VaR (α-quantile)
            "eev_var" => eev_var,
            "sddp_cvar" => sddp_cvar,      # CVaR (tail mean)
            "eev_cvar" => eev_cvar,
            "ravss_cvar" => ravss_cvar,    # VSS using CVaR
            "ravss_var" => ravss_var,      # VSS using VaR
            "ravss_percent" => relative_ravss_cvar
        )

        println("\nRisk Analysis at α=$(Int(100α))%:")
        println("  VaR ($(Int(100α))th percentile):")
        println("    SDDP: $(round(sddp_var/1e6, digits=2)) GSEK")
        println("    EV Policy: $(round(eev_var/1e6, digits=2)) GSEK")
        println("  CVaR (tail mean above VaR):")
        println("    SDDP: $(round(sddp_cvar/1e6, digits=2)) GSEK")
        println("  EV Policy CVaR: $(round(eev_cvar/1e6, digits=2)) GSEK")
        println("  Risk-Adjusted VSS (CVaR): $(round(ravss_cvar/1e3, digits=2)) MSEK ($(round(relative_ravss_cvar, digits=2))%)")
        println("  Risk-Adjusted VSS (VaR): $(round(ravss_var/1e3, digits=2)) MSEK")
    end

    # Create distribution plot
    p = violin([1], standard_results["sddp_costs"] / 1e6, label="SDDP", fillalpha=0.7, color=:blue)
    violin!([2], standard_results["eev_costs"] / 1e6, label="EV Policy", fillalpha=0.7, color=:red)

    # Add CVaR lines
    for α in RISK_LEVELS
        sddp_cvar = results["cvar_$(Int(100α))"]["sddp_cvar"] / 1e6
        eev_cvar = results["cvar_$(Int(100α))"]["eev_cvar"] / 1e6
        hline!([sddp_cvar], color=:blue, linestyle=:dash, label=(α == RISK_LEVELS[1] ? "CVaR levels" : ""))
        hline!([eev_cvar], color=:red, linestyle=:dash, label="")
    end

    xlabel!("Policy")
    ylabel!("Total Cost (GSEK)")
    title!("Cost Distribution Comparison")
    xticks!([1, 2], ["SDDP", "EV Policy"])

    savefig(p, joinpath(output_dir, "risk_adjusted_vss.png"))
    println("\nSaved risk-adjusted VSS plot to $(joinpath(output_dir, "risk_adjusted_vss.png"))")

    return results
end

"""
Conditional VSS by extreme event scenario
"""
function calculate_conditional_vss(standard_results, params)
    println("\n" * "="^80)
    println("CONDITIONAL VSS BY EXTREME EVENT")
    println("="^80)

    if !params.enable_extreme_events
        println("Extreme events not enabled, skipping conditional analysis")
        return Dict()
    end

    sddp_sims = standard_results["sddp_simulations"]
    eev_sims = standard_results["eev_simulations"]
    sddp_costs = standard_results["sddp_costs"]  # Pre-calculated costs
    eev_costs = standard_results["eev_costs"]    # Pre-calculated costs
    extreme_stage = params.apply_to_year * 2

    # Get unique extreme event scenarios
    scenarios = Dict{String,Tuple{Float64,Float64,Float64}}()
    for sim in sddp_sims
        if haskey(sim[extreme_stage], :noise_term)
            noise = sim[extreme_stage][:noise_term]
            key = get_scenario_name(noise.demand_mult, noise.elec_price_mult, noise.dc_avail)
            scenarios[key] = (noise.demand_mult, noise.elec_price_mult, noise.dc_avail)
        end
    end

    results = Dict{String,Any}()

    for (scenario_name, (d_mult, e_mult, dc_avail)) in scenarios
        # Filter simulations by scenario
        sddp_filtered = filter_simulations_by_extreme_event(sddp_sims, params, d_mult, e_mult, dc_avail)
        eev_filtered = filter_simulations_by_extreme_event(eev_sims, params, d_mult, e_mult, dc_avail)

        if !isempty(sddp_filtered) && !isempty(eev_filtered)
            # Get costs for filtered simulations using pre-calculated cost arrays
            sddp_scenario_costs = sddp_costs[sddp_filtered]
            eev_scenario_costs = eev_costs[eev_filtered]

            mean_sddp = mean(sddp_scenario_costs)
            mean_eev = mean(eev_scenario_costs)
            conditional_vss = mean_eev - mean_sddp
            relative_vss = 100 * conditional_vss / mean_sddp

            results[scenario_name] = Dict(
                "sddp_mean" => mean_sddp,
                "eev_mean" => mean_eev,
                "conditional_vss" => conditional_vss,
                "vss_percent" => relative_vss,
                "n_sddp" => length(sddp_filtered),
                "n_eev" => length(eev_filtered)
            )

            println("\n$scenario_name:")
            println("  SDDP mean: $(round(mean_sddp/1e6, digits=2)) GSEK (n=$(length(sddp_filtered)))")
            println("  EV policy mean: $(round(mean_eev/1e6, digits=2)) GSEK (n=$(length(eev_filtered)))")
            println("  Conditional VSS: $(round(conditional_vss/1e3, digits=2)) MSEK ($(round(relative_vss, digits=2))%)")
        end
    end

    # Create bar plot of conditional VSS
    if !isempty(results)
        scenarios = collect(keys(results))
        vss_values = [results[s]["conditional_vss"] / 1e3 for s in scenarios]
        vss_percents = [results[s]["vss_percent"] for s in scenarios]

        p = bar(scenarios, vss_values,
            label="VSS (MSEK)",
            ylabel="VSS (MSEK)",
            xlabel="Extreme Event Scenario",
            title="Conditional VSS by Extreme Event",
            rotation=45,
            fillalpha=0.7)

        # Add percentage labels
        for (i, (v, pct)) in enumerate(zip(vss_values, vss_percents))
            annotate!(i, v + maximum(vss_values) * 0.02, text("$(round(pct, digits=1))%", 8))
        end

        savefig(p, joinpath(output_dir, "conditional_vss.png"))
        println("\nSaved conditional VSS plot to $(joinpath(output_dir, "conditional_vss.png"))")
    end

    return results
end

"""
Helper function to get scenario name from multipliers
"""
function get_scenario_name(demand_mult::Float64, elec_mult::Float64, dc_avail::Float64)
    if demand_mult ≈ 1.0 && elec_mult ≈ 1.0 && dc_avail ≈ 1.0
        return "Baseline"
    elseif demand_mult > 1.2
        return "Cold Snap"
    elseif elec_mult > 1.5
        return "Price Surge"
    elseif dc_avail < 0.5
        return "DC Outage"
    else
        return "Combined"
    end
end

"""
Compare solution quality metrics beyond cost
"""
function analyze_solution_quality(standard_results, params, data)
    println("\n" * "="^80)
    println("SOLUTION QUALITY METRICS")
    println("="^80)

    sddp_chars = extract_solution_characteristics(standard_results["sddp_simulations"], params, data)
    eev_chars = extract_solution_characteristics(standard_results["eev_simulations"], params, data)

    println("\nFirst-Stage Investment Comparison:")
    println("-"^40)
    for tech in params.technologies
        sddp_inv = sddp_chars["first_stage_investments"][tech]
        eev_inv = eev_chars["first_stage_investments"][tech]
        diff = sddp_inv - eev_inv
        println("$tech:")
        println("  SDDP: $(round(sddp_inv, digits=1)) MW")
        println("  EV: $(round(eev_inv, digits=1)) MW")
        println("  Difference: $(round(diff, digits=1)) MW")
    end

    println("\nStorage Investment Timing:")
    println("-"^40)
    for (i, stage) in enumerate(params.investment_stages)
        sddp_stor = sddp_chars["storage_timing"][i]
        eev_stor = eev_chars["storage_timing"][i]
        println("Stage $stage:")
        println("  SDDP: $(round(sddp_stor, digits=1)) MWh")
        println("  EV: $(round(eev_stor, digits=1)) MWh")
    end

    println("\nOperational Performance:")
    println("-"^40)
    println("Unmet Demand:")
    println("  SDDP: $(round(sddp_chars["unmet_demand"], digits=2)) ± $(round(sddp_chars["unmet_demand_std"], digits=2)) MWh")
    println("  EV: $(round(eev_chars["unmet_demand"], digits=2)) ± $(round(eev_chars["unmet_demand_std"], digits=2)) MWh")
    println("Probability of Unmet Demand:")
    println("  SDDP: $(round(100*sddp_chars["prob_unmet"], digits=1))%")
    println("  EV: $(round(100*eev_chars["prob_unmet"], digits=1))%")

    results = Dict(
        "sddp" => sddp_chars,
        "eev" => eev_chars
    )

    return results
end

"""
Export comprehensive results to file
"""
function export_vss_results(all_results, output_file)
    open(output_file, "w") do io
        println(io, "="^80)
        println(io, "COMPREHENSIVE VSS ANALYSIS RESULTS")
        println(io, "="^80)
        println(io, "Generated: $(Dates.now())")
        println(io)

        if haskey(all_results, "standard")
            println(io, "\n" * "="^60)
            println(io, "STANDARD VSS")
            println(io, "="^60)
            std_res = all_results["standard"]
            println(io, "RP (SDDP): $(round(std_res["RP"]/1e6, digits=3)) GSEK")
            println(io, "EEV (EV Policy): $(round(std_res["EEV"]/1e6, digits=3)) GSEK")
            println(io, "VSS: $(round(std_res["VSS"]/1e3, digits=3)) MSEK")
            println(io, "VSS %: $(round(std_res["VSS_percent"], digits=2))%")
        end

        if haskey(all_results, "risk_adjusted")
            println(io, "\n" * "="^60)
            println(io, "RISK-ADJUSTED VSS")
            println(io, "="^60)
            for α in RISK_LEVELS
                res = all_results["risk_adjusted"]["cvar_$(Int(100α))"]
                println(io, "\nCVaR $(Int(100α))%:")
                println(io, "  SDDP CVaR: $(round(res["sddp_cvar"]/1e6, digits=3)) GSEK")
                println(io, "  EV CVaR: $(round(res["eev_cvar"]/1e6, digits=3)) GSEK")
                println(io, "  Risk-Adjusted VSS: $(round(res["ravss_cvar"]/1e3, digits=3)) MSEK ($(round(res["ravss_percent"], digits=2))%)")
            end
        end

        if haskey(all_results, "conditional")
            println(io, "\n" * "="^60)
            println(io, "CONDITIONAL VSS BY EXTREME EVENT")
            println(io, "="^60)
            for (scenario, res) in all_results["conditional"]
                println(io, "\n$scenario:")
                println(io, "  VSS: $(round(res["conditional_vss"]/1e3, digits=3)) MSEK ($(round(res["vss_percent"], digits=2))%)")
                println(io, "  Sample size: SDDP=$(res["n_sddp"]), EV=$(res["n_eev"])")
            end
        end

        if haskey(all_results, "solution_quality")
            println(io, "\n" * "="^60)
            println(io, "SOLUTION QUALITY METRICS")
            println(io, "="^60)
            sq = all_results["solution_quality"]
            println(io, "\nUnmet Demand:")
            println(io, "  SDDP: $(round(sq["sddp"]["unmet_demand"], digits=2)) MWh")
            println(io, "  EV: $(round(sq["eev"]["unmet_demand"], digits=2)) MWh")
            println(io, "\nProbability of Unmet Demand:")
            println(io, "  SDDP: $(round(100*sq["sddp"]["prob_unmet"], digits=1))%")
            println(io, "  EV: $(round(100*sq["eev"]["prob_unmet"], digits=1))%")
        end

        println(io, "\n" * "="^80)
        println(io, "END OF REPORT")
        println(io, "="^80)
    end

    println("\nResults exported to: $output_file")
end

##############################################################################
# Main Execution
##############################################################################

function main()
    println("="^80)
    println("COMPREHENSIVE VSS ANALYSIS")
    println("="^80)
    println("\nConfiguration:")
    println("  SDDP iterations: $ITERATION_LIMIT_SDDP")
    println("  Simulations: $N_SIMULATIONS")
    println("  Risk levels: $(join(RISK_LEVELS, ", "))")
    println("\nAnalyses to run:")
    println("  Standard VSS: $RUN_STANDARD_VSS")
    println("  Risk-Adjusted: $RUN_RISK_ADJUSTED")
    println("  Conditional VSS: $RUN_CONDITIONAL_VSS")
    println("  Simple Models: $RUN_SIMPLE_MODELS")
    println("  Solution Quality: $RUN_SOLUTION_QUALITY")

    # Load parameters and data
    println("\nLoading parameters and data...")
    params = load_parameters(excel_file)
    data = load_all_data(params, data_dir)

    # Store all results
    all_results = Dict{String,Any}()

    # 1. Standard VSS (required for other analyses)
    if RUN_STANDARD_VSS || RUN_RISK_ADJUSTED || RUN_CONDITIONAL_VSS || RUN_SOLUTION_QUALITY
        println("\n>>> Starting Standard VSS Analysis...")
        standard_results = calculate_standard_vss(params, data)
        all_results["standard"] = standard_results

        # Save intermediate results
        println("\nSaving simulation data...")
        save_simulation_results(standard_results["sddp_simulations"], params, data,
            joinpath(output_dir, "sddp_simulations.jld2"))
        save_simulation_results(standard_results["eev_simulations"], params, data,
            joinpath(output_dir, "eev_simulations.jld2"))
        println("✓ Standard VSS analysis complete")
    end

    # 2. Risk-Adjusted VSS
    if RUN_RISK_ADJUSTED && haskey(all_results, "standard")
        println("\n>>> Starting Risk-Adjusted VSS Analysis...")
        risk_results = calculate_risk_adjusted_vss(all_results["standard"])
        all_results["risk_adjusted"] = risk_results
        println("✓ Risk-Adjusted VSS analysis complete")
    end

    # 3. Conditional VSS
    if RUN_CONDITIONAL_VSS && haskey(all_results, "standard")
        println("\n>>> Starting Conditional VSS Analysis...")
        conditional_results = calculate_conditional_vss(all_results["standard"], params)
        all_results["conditional"] = conditional_results
        println("✓ Conditional VSS analysis complete")
    end

    # 4. Solution Quality
    if RUN_SOLUTION_QUALITY && haskey(all_results, "standard")
        println("\n>>> Starting Solution Quality Analysis...")
        quality_results = analyze_solution_quality(all_results["standard"], params, data)
        all_results["solution_quality"] = quality_results
        println("✓ Solution quality analysis complete")
    end

    # 5. Export all results
    export_vss_results(all_results, joinpath(output_dir, "vss_comprehensive_report.txt"))

    # Summary
    println("\n" * "="^80)
    println("ANALYSIS COMPLETE")
    println("="^80)

    if haskey(all_results, "standard")
        println("\nKey Findings:")
        println("  Standard VSS: $(round(all_results["standard"]["VSS"]/1e3, digits=2)) MSEK ($(round(all_results["standard"]["VSS_percent"], digits=2))%)")

        if haskey(all_results, "risk_adjusted")
            ravss_95 = all_results["risk_adjusted"]["cvar_95"]["ravss_cvar"]
            ravss_95_pct = all_results["risk_adjusted"]["cvar_95"]["ravss_percent"]
            println("  Risk-Adjusted VSS (CVaR 95%): $(round(ravss_95/1e3, digits=2)) MSEK ($(round(ravss_95_pct, digits=2))%)")
        end

        if haskey(all_results, "conditional") && !isempty(all_results["conditional"])
            max_cond_vss = maximum(v["conditional_vss"] for v in values(all_results["conditional"]))
            max_scenario = [k for (k, v) in all_results["conditional"] if v["conditional_vss"] == max_cond_vss][1]
            println("  Max Conditional VSS: $(round(max_cond_vss/1e3, digits=2)) MSEK ($max_scenario)")
        end
    end

    println("\nOutputs saved to: $output_dir")
    println("  - vss_comprehensive_report.txt")
    println("  - risk_adjusted_vss.png")
    println("  - conditional_vss.png")
    println("  - sddp_simulations.jld2")
    println("  - eev_simulations.jld2")
end

# Run the analysis
main()