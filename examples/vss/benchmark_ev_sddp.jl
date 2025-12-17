"""
Benchmark: SDDP vs Expected-Value SDDP Model (VSS Calculation)

This script calculates the proper Value of Stochastic Solution (VSS) using:
- Stochastic SDDP model for the Recourse Problem (RP)
- SDDP-based EV model for extracting deterministic investments
- evaluate_ev_policy() for evaluating EV investments under uncertainty (EEV)

VSS = EEV - RP

Key difference from benchmark_expected_value.jl:
- Uses SDDP-based EV model (from ev_model_integrated.jl) instead of JuMP deterministic model
- Investments extracted from simulations, not JuMP variables
- Both models use same SDDP infrastructure for consistency

Usage:
    cd examples
    julia --threads auto benchmark_ev_sddp.jl
"""

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

using DHCapEx
using SDDP
using Statistics
using Plots
using StatsPlots
using Dates
using Random

println("="^70)
println("BENCHMARK: SDDP vs Expected-Value SDDP Model")
println("="^70)

##############################################################################
# CONFIGURATION - Adjust these for testing vs production
##############################################################################

# Training configuration
const ITERATION_LIMIT_SDDP = 500   # Stochastic model (higher for extreme events)
const ITERATION_LIMIT_EV = 100    # EV model needs fewer iterations (deterministic)

# Simulation configuration
const N_SIMULATIONS = 500          # For SDDP (RP) and EEV evaluation - MUST MATCH for valid comparison
const N_SIMULATIONS_EV = 10       # For EV model only (deterministic - all results identical)

# Random seed for reproducibility (CRITICAL for valid VSS comparison)
const RANDOM_SEED = 12345

# Branching structure
const LATE_TEMP_BRANCHING = true  # true = late branching (default), false = early branching

# Output configuration
const OUTPUT_DIR = joinpath(@__DIR__, "..", "output")

##############################################################################
# Load Parameters and Data
##############################################################################

println("\n1. Loading model parameters...")
params = load_parameters(joinpath(@__DIR__, "..", "data", "model_parameters.xlsx"))

println("2. Loading and processing data...")
data = load_all_data(params, joinpath(@__DIR__, "..", "data"))

println("\n" * "="^70)
println("MODEL CONFIGURATION")
println("="^70)
println("Planning horizon: T=$(params.T) years ($(2*params.T) stages)")
println("Temperature scenarios: $(length(params.temp_scenarios))")
println("Energy states: $(size(params.energy_transitions, 1))")
println("Extreme events: $(params.enable_extreme_events ? "ENABLED at year $(params.apply_to_year)" : "DISABLED")")
println("="^70)

##############################################################################
# PART 1: Build and Train Stochastic SDDP Model (Recourse Problem)
##############################################################################

println("\n" * "="^70)
println("PART 1: STOCHASTIC SOLUTION (SDDP)")
println("="^70)

println("\n3. Building SDDP model...")
branching_str = LATE_TEMP_BRANCHING ? "LATE" : "EARLY"
println("   Temperature branching: $branching_str")
sddp_model = build_sddp_model(params, data; late_temp_branching=LATE_TEMP_BRANCHING)
println("   Nodes: $(length(sddp_model.nodes))")

println("\n4. Training SDDP model ($ITERATION_LIMIT_SDDP iterations)...")
sddp_start_time = time()
SDDP.train(sddp_model;
    risk_measure=SDDP.Expectation(),  # Risk-neutral for standard VSS
    iteration_limit=ITERATION_LIMIT_SDDP,
    print_level=1,
    log_frequency=25,
    run_numerical_stability_report=false
)
sddp_training_time = time() - sddp_start_time

println("\n5. Running SDDP simulations ($N_SIMULATIONS scenarios)...")
sim_start_time = time()

# CRITICAL: Set random seed for reproducibility
Random.seed!(RANDOM_SEED)

sddp_simulations = SDDP.simulate(sddp_model, N_SIMULATIONS,
    [:u_expansion_tech, :u_expansion_storage, :u_unmet];
    parallel_scheme=SDDP.Threaded()
)
sim_time = time() - sim_start_time

# Calculate RP (Recourse Problem cost)
sddp_costs = get_simulation_costs(sddp_simulations)
RP = mean(sddp_costs)
sddp_std = std(sddp_costs)
sddp_lower_bound = SDDP.calculate_bound(sddp_model)

println("\n" * "="^70)
println("SDDP RESULTS (Recourse Problem - RP)")
println("="^70)
println("Training time:      $(round(sddp_training_time, digits=2)) seconds")
println("Simulation time:    $(round(sim_time, digits=2)) seconds")
println()
println("Lower bound:        $(round(sddp_lower_bound / 1e6, digits=3)) MSEK")
println("Mean cost (RP):     $(round(RP / 1e6, digits=3)) MSEK")
println("Std deviation:      $(round(sddp_std / 1e6, digits=3)) MSEK")
println("="^70)

# Extract SDDP first-stage investments for comparison
println("\nFirst-stage investments (SDDP - Stochastic):")
sddp_investments = Dict{Symbol,Float64}()
for tech in params.technologies
    inv_values = [sim[1][:u_expansion_tech][tech] for sim in sddp_simulations]
    avg_inv = mean(inv_values)
    sddp_investments[tech] = avg_inv
    if avg_inv > 0.1
        println("  $tech: $(round(avg_inv, digits=2)) MW (std: $(round(std(inv_values), digits=2)))")
    end
end

##############################################################################
# PART 2: Build and Train EV SDDP Model
##############################################################################

println("\n" * "="^70)
println("PART 2: EXPECTED-VALUE SDDP MODEL")
println("="^70)

println("\n6. Building EV SDDP model (integrated approach)...")
ev_build_start = time()
ev_model, ev_params, ev_data = build_ev_sddp_model_integrated(params, data)
ev_build_time = time() - ev_build_start

# Verify EV model structure
println("\n7. Verifying EV model structure...")
is_linear = verify_ev_model_structure(ev_model, sddp_model)
if !is_linear
    @warn "EV model is not linear! This is unexpected."
end

println("\n8. Training EV SDDP model ($ITERATION_LIMIT_EV iterations)...")
ev_train_start = time()
SDDP.train(ev_model;
    risk_measure=SDDP.Expectation(),
    iteration_limit=ITERATION_LIMIT_EV,
    print_level=1,
    log_frequency=25,
    run_numerical_stability_report=false
)
ev_train_time = time() - ev_train_start

println("\n9. Simulating EV model ($N_SIMULATIONS_EV scenarios - deterministic, all identical)...")
ev_simulations = SDDP.simulate(ev_model, N_SIMULATIONS_EV,
    [:u_expansion_tech, :u_expansion_storage, :u_unmet];
    parallel_scheme=SDDP.Threaded()
)

# Verify deterministic behavior (all simulations should be identical)
ev_costs = get_simulation_costs(ev_simulations)
ev_cost_mean = mean(ev_costs)
ev_cost_std = std(ev_costs)

println("\n" * "="^70)
println("EV SDDP MODEL RESULTS")
println("="^70)
println("Build time:         $(round(ev_build_time, digits=2)) seconds")
println("Training time:      $(round(ev_train_time, digits=2)) seconds")
println()
println("EV objective:       $(round(ev_cost_mean / 1e6, digits=3)) MSEK")
println("Std deviation:      $(round(ev_cost_std / 1e6, digits=6)) MSEK (should be ~0)")

if ev_cost_std > 1e-3
    @warn "EV model has non-zero variance ($(round(ev_cost_std, digits=6))) - unexpected for deterministic model!"
else
    println("EV model is deterministic (near-zero variance)")
end
println("="^70)

##############################################################################
# PART 3: Extract EV Investments and Evaluate Under Uncertainty (EEV)
##############################################################################

println("\n" * "="^70)
println("PART 3: EVALUATING EV POLICY UNDER UNCERTAINTY (EEV)")
println("="^70)

println("\n10. Extracting EV investment decisions...")
ev_investments = extract_ev_investments_from_simulations(ev_simulations, params)

println("\nEV investments by stage:")
for stage in sort(collect(keys(ev_investments)))
    println("\n  Stage $stage (Year $(Int(ceil(stage/2)))):")
    invs = ev_investments[stage]
    for tech in params.technologies
        if invs[:tech][tech] > 0.1
            println("    $tech: $(round(invs[:tech][tech], digits=2)) MW")
        end
    end
    if invs[:storage] > 0.1
        println("    Storage: $(round(invs[:storage], digits=2)) MWh")
    end
end

println("\n11. Evaluating EV policy under uncertainty ($N_SIMULATIONS scenarios)...")
println("    Using SAME random seed ($RANDOM_SEED) for identical scenarios...")

eev_simulations, EEV, eev_std = evaluate_ev_policy(
    sddp_model, ev_investments, params, N_SIMULATIONS;
    verbose=false,
    random_seed=RANDOM_SEED  # CRITICAL: Same seed ensures identical scenarios
)

println("\n" * "="^70)
println("EEV RESULTS (EV policy under uncertainty)")
println("="^70)
println("Mean cost (EEV):    $(round(EEV / 1e6, digits=3)) MSEK")
println("Std deviation:      $(round(eev_std / 1e6, digits=3)) MSEK")
println("="^70)

##############################################################################
# PART 4: Calculate VSS
##############################################################################

println("\n" * "="^70)
println("PART 4: VALUE OF STOCHASTIC SOLUTION (VSS)")
println("="^70)

# VSS = EEV - RP
VSS = EEV - RP
VSS_percent = (VSS / RP) * 100

println("\nNote: Both policies evaluated on IDENTICAL scenarios (fixed seed)")
println("\nCOST COMPARISON:")
println("  RP (SDDP optimized):              $(round(RP / 1e6, digits=3)) MSEK")
println("  SDDP std deviation:               $(round(sddp_std / 1e6, digits=3)) MSEK")
println()
println("  EV objective (perfect foresight): $(round(ev_cost_mean / 1e6, digits=3)) MSEK")
println()
println("  EEV (EV policy under uncertainty): $(round(EEV / 1e6, digits=3)) MSEK")
println("  EEV std deviation:                $(round(eev_std / 1e6, digits=3)) MSEK")
println()
println("VALUE OF STOCHASTIC SOLUTION:")
println("  VSS = EEV - RP:                   $(round(VSS / 1e6, digits=3)) MSEK")
println("  VSS as % of RP:                   $(round(VSS_percent, digits=2))%")

# Interpretation
if VSS > 0
    println("\nPOSITIVE VSS: Stochastic optimization provides value!")
    println("  Interpretation:")
    println("    - Using deterministic (EV) investments under uncertainty costs $(round(VSS / 1e6, digits=2)) MSEK more")
    println("    - SDDP saves $(round(VSS_percent, digits=1))% by hedging against uncertainty")
    println("    - EV model underestimates true cost by $(round((EEV - ev_cost_mean) / 1e6, digits=2)) MSEK")
elseif VSS > -0.01 * RP
    println("\nNEAR-ZERO VSS: Limited benefit from stochastic modeling")
    println("  Interpretation:")
    println("    - SDDP and EV policies perform similarly under uncertainty")
    println("    - Uncertainty impact is low for this problem")
else
    println("\nNEGATIVE VSS: SDDP costs more than using EV decisions!")
    println("  Interpretation:")
    println("    - This is unexpected and may indicate:")
    println("      - SDDP convergence issues (try more iterations)")
    println("      - Implementation bugs")
    println("      - Numerical instability")
end

##############################################################################
# PART 5: Investment Strategy Comparison
##############################################################################

println("\n" * "="^70)
println("PART 5: INVESTMENT STRATEGY COMPARISON")
println("="^70)

# First-stage investments (from ev_investments)
ev_first_stage = ev_investments[1][:tech]

println("\nFIRST-STAGE INVESTMENT DIFFERENCES:")
println("Technology              SDDP (MW)    EV (MW)    Difference")
println("-"^65)
for tech in params.technologies
    sddp_inv = sddp_investments[tech]
    ev_inv = ev_first_stage[tech]
    diff = sddp_inv - ev_inv
    if abs(sddp_inv) > 0.1 || abs(ev_inv) > 0.1
        println("$(rpad(String(tech), 20))  $(lpad(round(sddp_inv, digits=1), 10))  $(lpad(round(ev_inv, digits=1), 10))  $(lpad(round(diff, digits=1), 10))")
    end
end

# Identify key differences
println("\nKEY STRATEGY DIFFERENCES:")
hedging_techs = Symbol[]
for tech in params.technologies
    sddp_inv = sddp_investments[tech]
    ev_inv = ev_first_stage[tech]
    if abs(sddp_inv - ev_inv) > 5.0  # > 5 MW difference
        if sddp_inv > ev_inv
            println("  - SDDP invests MORE in $tech: $(round(sddp_inv - ev_inv, digits=1)) MW more")
        else
            println("  - SDDP invests LESS in $tech: $(round(ev_inv - sddp_inv, digits=1)) MW less")
        end
        push!(hedging_techs, tech)
    end
end

if isempty(hedging_techs)
    println("  - Investment strategies are very similar")
end

##############################################################################
# PART 6: Visualization
##############################################################################

println("\n" * "="^70)
println("PART 6: CREATING VISUALIZATIONS")
println("="^70)

# Plot 1: Cost distribution comparison
eev_costs = get_simulation_costs(eev_simulations)

p1 = violin([1], sddp_costs / 1e6, label="SDDP", fillalpha=0.7, color=:steelblue)
violin!([2], eev_costs / 1e6, label="EV Policy", fillalpha=0.7, color=:coral)
xlabel!("Policy")
ylabel!("Total Cost (MSEK)")
title!("Cost Distribution: SDDP vs EV Policy")
xticks!([1, 2], ["SDDP\n(RP)", "EV Policy\n(EEV)"])

# Add mean lines
hline!([RP / 1e6], color=:steelblue, linestyle=:dash, label="RP mean", linewidth=2)
hline!([EEV / 1e6], color=:coral, linestyle=:dash, label="EEV mean", linewidth=2)

# Plot 2: First-stage investment comparison
techs_to_plot = filter(t -> sddp_investments[t] > 0.1 || ev_first_stage[t] > 0.1, params.technologies)
if !isempty(techs_to_plot)
    tech_names = [String(t) for t in techs_to_plot]
    sddp_invs = [sddp_investments[t] for t in techs_to_plot]
    ev_invs = [ev_first_stage[t] for t in techs_to_plot]

    x = 1:length(techs_to_plot)
    p2 = groupedbar([sddp_invs ev_invs],
        bar_position=:dodge,
        xlabel="Technology",
        ylabel="Investment (MW)",
        title="First-Stage Investments",
        label=["SDDP" "EV Model"],
        xticks=(x, tech_names),
        xrotation=45,
        legend=:topright,
        color=[:steelblue :coral],
        alpha=0.8
    )
else
    p2 = plot(title="No significant first-stage investments", legend=false)
end

# Combine plots
combined = plot(p1, p2, layout=(1, 2), size=(1200, 500),
    plot_title="VSS Analysis: SDDP vs Expected-Value Model")

output_plot = joinpath(OUTPUT_DIR, "benchmark_ev_sddp_comparison.png")
savefig(combined, output_plot)
println("Comparison plot saved to: $output_plot")

##############################################################################
# PART 7: Save Results
##############################################################################

println("\n" * "="^70)
println("PART 7: SAVING RESULTS")
println("="^70)

# Save simulation results for future analysis
println("\nSaving simulation data...")
sddp_file = save_simulation_results_auto(sddp_simulations, params, data; output_dir=OUTPUT_DIR)
println("  SDDP results saved to: $(basename(sddp_file))")

eev_file = joinpath(OUTPUT_DIR, "eev_simulations_$(Dates.format(now(), "yyyy_mm_dd_HHMM")).jld2")
save_simulation_results(eev_simulations, params, data, eev_file)
println("  EEV results saved to: $(basename(eev_file))")

# Save detailed text report
println("\nSaving detailed report...")
output_file = joinpath(OUTPUT_DIR, "benchmark_ev_sddp_summary.txt")
open(output_file, "w") do io
    println(io, "="^70)
    println(io, "BENCHMARK: SDDP vs Expected-Value SDDP Model")
    println(io, "="^70)
    println(io, "\nGenerated: $(Dates.now())")
    println(io, "\nMODEL CONFIGURATION:")
    println(io, "  Planning horizon: T=$(params.T) years")
    println(io, "  Total stages: $(2*params.T)")
    println(io, "  Temperature scenarios: $(length(params.temp_scenarios))")
    println(io, "  Energy price states: $(size(params.energy_transitions, 1))")
    println(io, "  Extreme events: $(params.enable_extreme_events ? "ENABLED at year $(params.apply_to_year)" : "DISABLED")")
    println(io)
    println(io, "\nTRAINING CONFIGURATION:")
    println(io, "  SDDP iterations: $ITERATION_LIMIT_SDDP")
    println(io, "  EV iterations: $ITERATION_LIMIT_EV")
    println(io, "  SDDP/EEV simulations: $N_SIMULATIONS")
    println(io, "  EV simulations: $N_SIMULATIONS_EV")
    println(io, "  Random seed: $RANDOM_SEED")
    println(io)
    println(io, "\nRESULTS:")
    println(io, "  SDDP lower bound:               $(round(sddp_lower_bound / 1e6, digits=3)) MSEK")
    println(io, "  SDDP cost (RP):                 $(round(RP / 1e6, digits=3)) MSEK")
    println(io, "  EV cost (perfect foresight):    $(round(ev_cost_mean / 1e6, digits=3)) MSEK")
    println(io, "  EEV (EV policy under uncert.):  $(round(EEV / 1e6, digits=3)) MSEK")
    println(io)
    println(io, "  VSS = EEV - RP:                 $(round(VSS / 1e6, digits=3)) MSEK")
    println(io, "  VSS as % of RP:                 $(round(VSS_percent, digits=2))%")
    println(io)
    println(io, "\nCOMPUTATION TIME:")
    println(io, "  SDDP training:   $(round(sddp_training_time, digits=2)) seconds")
    println(io, "  SDDP simulation: $(round(sim_time, digits=2)) seconds")
    println(io, "  EV build:        $(round(ev_build_time, digits=2)) seconds")
    println(io, "  EV training:     $(round(ev_train_time, digits=2)) seconds")
    println(io)
    println(io, "\nFIRST-STAGE INVESTMENTS:")
    println(io, "  Technology              SDDP (MW)    EV (MW)    Difference")
    println(io, "  " * "-"^63)
    for tech in params.technologies
        if sddp_investments[tech] > 0.1 || ev_first_stage[tech] > 0.1
            sddp_inv = sddp_investments[tech]
            ev_inv = ev_first_stage[tech]
            diff = sddp_inv - ev_inv
            println(io, "  $(rpad(String(tech), 20))  $(lpad(round(sddp_inv, digits=1), 10))  $(lpad(round(ev_inv, digits=1), 10))  $(lpad(round(diff, digits=1), 10))")
        end
    end
    println(io)
    println(io, "\nINTERPRETATION:")
    if VSS > 0
        println(io, "  - POSITIVE VSS: Stochastic optimization provides value")
        println(io, "  - Using deterministic (EV) investments under uncertainty costs $(round(VSS / 1e6, digits=2)) MSEK more")
        println(io, "  - SDDP saves $(round(VSS_percent, digits=1))% by hedging against uncertainty")
    elseif VSS > -0.01 * RP
        println(io, "  - VSS near zero suggests limited benefit from stochastic modeling")
    else
        println(io, "  - NEGATIVE VSS: Unexpected result, check convergence")
    end
    println(io, "="^70)
end
println("Report saved to: $output_file")

println("\n" * "="^70)
println("BENCHMARK COMPLETE")
println("="^70)
println("\nSUMMARY:")
println("  - SDDP cost (RP):           $(round(RP / 1e6, digits=2)) MSEK")
println("  - EV cost (foresight):      $(round(ev_cost_mean / 1e6, digits=2)) MSEK")
println("  - EEV (EV under uncert.):   $(round(EEV / 1e6, digits=2)) MSEK")
println("  - VSS = EEV - RP:           $(round(VSS / 1e6, digits=2)) MSEK ($(round(VSS_percent, digits=1))%)")
println("\nOutputs saved to: $OUTPUT_DIR")
println("="^70)
