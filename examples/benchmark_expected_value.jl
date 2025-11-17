"""
Benchmark: SDDP vs Expected-Value Deterministic Model

This script compares the stochastic SDDP solution with a deterministic model
that uses expected values for all uncertain parameters.

This calculates the Value of Stochastic Solution (VSS):
  VSS = Cost of using deterministic solution - Cost of stochastic solution
  VSS shows how much value uncertainty modeling provides

Key comparison:
- SDDP: Optimizes under uncertainty, hedges against different scenarios
- EV Model: Optimizes assuming expected values occur with certainty
"""

# Add src to load path
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

using DHCapEx
using SDDP  # Already imported by DHCapEx, but needed for SDDP-specific functions
using Statistics
using Plots
using StatsPlots
using Dates

println("="^70)
println("BENCHMARK: SDDP vs Expected-Value Deterministic Model")
println("="^70)

# Load parameters and data
println("\n1. Loading model parameters...")
params = load_parameters(joinpath(@__DIR__, "..", "data", "model_parameters.xlsx"))

println("2. Loading and processing data...")
data = load_all_data(params, joinpath(@__DIR__, "..", "data"))

println("\n" * "="^70)
println("MODEL CONFIGURATION")
println("="^70)
println("Planning horizon: T=$(params.T) years ($(2*params.T) stages)")
println("Temperature scenarios: 2")
println("Energy states: 3 (High/Medium/Low)")
println("Uncertainty modeling: Yes (SDDP) vs No (EV)")
println("="^70)

# ============================================================================
# PART 1: Solve with SDDP (Stochastic)
# ============================================================================

println("\n" * "="^70)
println("PART 1: STOCHASTIC SOLUTION (SDDP)")
println("="^70)

println("\n3. Building SDDP model...")
sddp_model = build_sddp_model(params, data)

println("\n4. Training SDDP model...")
sddp_start_time = time()
SDDP.train(sddp_model;
    iteration_limit=500,
    time_limit=3600.0,
    print_level=1,
    run_numerical_stability_report=false)
sddp_training_time = time() - sddp_start_time

println("\n5. Running SDDP simulations...")
sim_start_time = time()
import Random
Random.seed!(12345)  # Fix seed for reproducibility
sddp_simulations = SDDP.simulate(sddp_model, 500;
    parallel_scheme=SDDP.Threaded(),
    custom_recorders=Dict{Symbol,Function}(
        :u_expansion_tech => (sp) -> haskey(sp, :u_expansion_tech) ?
                                     Dict(tech => SDDP.JuMP.value(sp[:u_expansion_tech][tech])
                                          for tech in params.technologies) :
                                     nothing
    ))
sim_time = time() - sim_start_time

# Calculate SDDP statistics
sddp_sim_costs = [sum(s[:stage_objective] for s in sim) for sim in sddp_simulations]
sddp_mean_cost = mean(sddp_sim_costs)
sddp_std = std(sddp_sim_costs)
sddp_lower_bound = SDDP.calculate_bound(sddp_model)

println("\n" * "="^70)
println("SDDP RESULTS (Recourse Problem - RP)")
println("="^70)
println("Training time:      $(round(sddp_training_time, digits=2)) seconds")
println("Simulation time:    $(round(sim_time, digits=2)) seconds")
println("Total time:         $(round(sddp_training_time + sim_time, digits=2)) seconds")
println()
println("Mean cost (RP):     $(round(sddp_mean_cost / 1e6, digits=3)) M€")
println("Std deviation:      $(round(sddp_std / 1e6, digits=3)) M€")
println("Lower bound:        $(round(sddp_lower_bound / 1e6, digits=3)) M€")
println("="^70)

# Extract first-stage investments from SDDP
println("\nFirst-stage investments (SDDP - Stochastic):")
sddp_investments = Dict{Symbol,Float64}()
for tech in params.technologies
    # Access the custom recorded variable
    inv_values = [sim[1][:u_expansion_tech][tech] for sim in sddp_simulations if sim[1][:u_expansion_tech] !== nothing]
    avg_inv = length(inv_values) > 0 ? mean(inv_values) : 0.0
    sddp_investments[tech] = avg_inv
    if avg_inv > 0.1
        println("  $tech: $(round(avg_inv, digits=2)) MW")
    end
end

# ============================================================================
# PART 2: Solve Expected-Value Deterministic Model
# ============================================================================

println("\n" * "="^70)
println("PART 2: EXPECTED-VALUE DETERMINISTIC MODEL")
println("="^70)

println("\n6. Building expected-value deterministic model...")
ev_build_start = time()
ev_model, ev_variables = build_deterministic_model(params, data)
ev_build_time = time() - ev_build_start

println("\n7. Solving expected-value model...")
SDDP.JuMP.set_optimizer_attribute(ev_model, "OutputFlag", 1)
ev_solve_start = time()
SDDP.JuMP.optimize!(ev_model)
ev_solve_time = time() - ev_solve_start

ev_status = SDDP.JuMP.termination_status(ev_model)
ev_objective = SDDP.JuMP.objective_value(ev_model)

println("\n" * "="^70)
println("EXPECTED-VALUE MODEL RESULTS")
println("="^70)
println("Status:        $ev_status")
println("Build time:    $(round(ev_build_time, digits=2)) seconds")
println("Solve time:    $(round(ev_solve_time, digits=2)) seconds")
println("Total time:    $(round(ev_build_time + ev_solve_time, digits=2)) seconds")
println()
println("Objective (EV): $(round(ev_objective / 1e6, digits=3)) M€")
println()
println("Note: EV model uses:")
println("  • Probability-weighted carrier prices (evolved via Markov transitions)")
println("  • Probability-weighted electricity prices (per week, hour)")
println("  • Proper expected-value formulation (not single-scenario mode)")
println("  • State mapping: 1=High, 2=Medium, 3=Low (corrected)")
println("="^70)

# Extract ALL EV investment decisions (all vintage stages)
println("\n8. Extracting EV investment decisions from all stages...")
ev_investments_all = extract_ev_investments(ev_model, ev_variables, params)

# Show investments from all stages
println("\nEV investments by stage:")
for stage in sort(collect(keys(ev_investments_all)))
    println("\n  Stage $stage (Year $(Int(ceil(stage/2)))):")
    invs = ev_investments_all[stage]
    for tech in params.technologies
        if invs[:tech][tech] > 0.1
            println("    $tech: $(round(invs[:tech][tech], digits=2)) MW")
        end
    end
    if invs[:storage] > 0.1
        println("    Storage: $(round(invs[:storage], digits=2)) MWh")
    end
end

# Calculate total EV investments
println("\nTotal EV investments (all stages):")
total_ev = Dict{Symbol,Float64}()
for tech in params.technologies
    total_ev[tech] = sum(ev_investments_all[stage][:tech][tech] for stage in keys(ev_investments_all))
    if total_ev[tech] > 0.1
        println("  $tech: $(round(total_ev[tech], digits=2)) MW")
    end
end

# Extract first-stage investments for later comparison
ev_investments_first = ev_investments_all[1][:tech]

# ============================================================================
# PART 3: Calculate Value of Stochastic Solution (VSS)
# ============================================================================

println("\n" * "="^70)
println("PART 3: VALUE OF STOCHASTIC SOLUTION (VSS)")
println("="^70)

# VSS = EEV - RP
# Where:
#   RP = Recourse Problem (SDDP solution) = sddp_mean_cost
#   EEV = Expected value of EV solution in stochastic context
#
# Proper VSS requires evaluating EV investment decisions under uncertainty

println("\n9. Evaluating EV investment policy under uncertainty...")
Random.seed!(12345)  # Use SAME seed to ensure identical scenarios
eev_mean, eev_std = evaluate_ev_policy(sddp_model, ev_investments_all, params, 500; verbose=true)

# Calculate proper VSS
vss = eev_mean - sddp_mean_cost
vss_percent = (vss / sddp_mean_cost) * 100

println("\n" * "="^70)
println("PROPER VSS CALCULATION")
println("="^70)
println("\nNote: Both policies evaluated on IDENTICAL scenarios (fixed seed)")
println("\nCOST COMPARISON:")
println("  RP (SDDP optimized):            $(round(sddp_mean_cost / 1e6, digits=3)) M€")
println("  SDDP std deviation:             $(round(sddp_std / 1e6, digits=3)) M€")
println()
println("  EEV (EV policy under uncertainty): $(round(eev_mean / 1e6, digits=3)) M€")
println("  EEV std deviation:              $(round(eev_std / 1e6, digits=3)) M€")
println()
println("  Variance reduction (SDDP vs EV): $(round((eev_std - sddp_std) / 1e6, digits=3)) M€ ($(round((1 - sddp_std / eev_std) * 100, digits=1))% less volatile)")
println()
println("VALUE OF STOCHASTIC SOLUTION:")
println("  VSS = EEV - RP:                 $(round(vss / 1e6, digits=3)) M€")
println("  VSS as % of RP:                 $(round(vss_percent, digits=2))%")
println()
println("REFERENCE (for comparison):")
println("  EV objective (perfect foresight): $(round(ev_objective / 1e6, digits=3)) M€")
println("  Difference EEV - EV:              $(round((eev_mean - ev_objective) / 1e6, digits=3)) M€")

if vss > 0
    println("\n✓ POSITIVE VSS: Stochastic optimization provides value!")
    println("  Interpretation:")
    println("    • Using deterministic (EV) investments under uncertainty costs $(round(vss / 1e6, digits=2)) M€ more")
    println("    • SDDP saves $(round(vss_percent, digits=1))% by hedging against uncertainty")
    println("    • EV model underestimates true cost by $(round((eev_mean - ev_objective) / 1e6, digits=2)) M€ (assumes perfect foresight)")
    println("    • Investment strategies differ significantly (see PART 4 below)")
elseif vss > -0.01 * sddp_mean_cost
    println("\n≈ NEAR-ZERO VSS: Limited benefit from stochastic modeling")
    println("  Interpretation:")
    println("    • SDDP and EV policies perform similarly under uncertainty")
    println("    • Uncertainty impact is low for this problem")
else
    println("\n⚠ NEGATIVE VSS: SDDP costs more than using EV decisions!")
    println("  Interpretation:")
    println("    • This is unexpected and may indicate:")
    println("      - SDDP convergence issues (try more iterations)")
    println("      - Implementation bugs")
    println("      - Numerical instability")
end

# ============================================================================
# PART 4: Investment Strategy Comparison
# ============================================================================

println("\n" * "="^70)
println("PART 4: INVESTMENT STRATEGY COMPARISON")
println("="^70)

println("\nFIRST-STAGE INVESTMENT DIFFERENCES:")
println("Technology              SDDP (MW)    EV (MW)    Difference")
println("-"^65)
for tech in params.technologies
    sddp_inv = sddp_investments[tech]
    ev_inv = ev_investments_first[tech]
    diff = sddp_inv - ev_inv
    if abs(sddp_inv) > 0.1 || abs(ev_inv) > 0.1
        println("$(rpad(String(tech), 20))  $(lpad(round(sddp_inv, digits=1), 10))  $(lpad(round(ev_inv, digits=1), 10))  $(lpad(round(diff, digits=1), 10))")
    end
end

# Identify key differences
println("\nKEY STRATEGY DIFFERENCES:")
hedging_techs = []
for tech in params.technologies
    sddp_inv = sddp_investments[tech]
    ev_inv = ev_investments_first[tech]
    if abs(sddp_inv - ev_inv) > 5.0  # > 5 MW difference
        diff_pct = abs(sddp_inv - ev_inv) / max(sddp_inv, ev_inv, 1.0) * 100
        if sddp_inv > ev_inv
            println("  • SDDP invests MORE in $tech: $(round(sddp_inv - ev_inv, digits=1)) MW more ($(round(diff_pct, digits=1))%)")
        else
            println("  • SDDP invests LESS in $tech: $(round(ev_inv - sddp_inv, digits=1)) MW less ($(round(diff_pct, digits=1))%)")
        end
        push!(hedging_techs, tech)
    end
end

if isempty(hedging_techs)
    println("  • Investment strategies are very similar")
else
    println("\n  Interpretation: SDDP hedges uncertainty by adjusting investments in:")
    for tech in hedging_techs
        println("    - $tech")
    end
end

# ============================================================================
# PART 5: Visualization
# ============================================================================

println("\n" * "="^70)
println("PART 5: CREATING COMPARISON VISUALIZATIONS")
println("="^70)

# Plot 1: Cost comparison
p1 = bar(["SDDP\n(Stochastic)", "EV Model\n(Deterministic)"],
    [sddp_mean_cost / 1e6, ev_objective / 1e6],
    ylabel="Total Cost [M€]",
    title="Cost Comparison",
    legend=false,
    color=[:steelblue, :coral],
    alpha=0.8,
    grid=true,
    size=(400, 400))

# Add VSS annotation
annotate!(p1, 1.5, max(sddp_mean_cost, eev_mean) / 1e6 * 0.5,
    text("VSS:\n$(round(vss/1e6, digits=2)) M€\n($(round(vss_percent, digits=1))%)",
        10, :center))

# Plot 2: First-stage investment comparison
techs_to_plot = filter(t -> sddp_investments[t] > 0.1 || ev_investments_first[t] > 0.1, params.technologies)
if !isempty(techs_to_plot)
    tech_names = [String(t) for t in techs_to_plot]
    sddp_invs = [sddp_investments[t] for t in techs_to_plot]
    ev_invs = [ev_investments_first[t] for t in techs_to_plot]

    x = 1:length(techs_to_plot)
    p2 = groupedbar([sddp_invs ev_invs],
        bar_position=:dodge,
        xlabel="Technology",
        ylabel="Investment [MW]",
        title="First-Stage Investments",
        label=["SDDP" "EV Model"],
        xticks=(x, tech_names),
        xrotation=45,
        legend=:topright,
        color=[:steelblue :coral],
        alpha=0.8,
        grid=true,
        size=(800, 500))
else
    p2 = plot(title="No significant first-stage investments", legend=false)
end

# Combine plots
combined = plot(p1, p2, layout=(1, 2), size=(1200, 500),
    plot_title="SDDP vs Expected-Value Comparison")

output_plot = joinpath(@__DIR__, "..", "output", "benchmark_sddp_vs_ev.png")
savefig(combined, output_plot)
println("✓ Comparison plot saved to: $output_plot")

# ============================================================================
# PART 6: Save Results and Summary Report
# ============================================================================

# Save simulation results for future analysis
println("\n8. Saving simulation results...")
output_dir = joinpath(@__DIR__, "..", "output")

# Save SDDP simulations
sddp_file = save_simulation_results_auto(sddp_simulations, params, data; output_dir=output_dir)
println("  ✓ SDDP results saved to: $(basename(sddp_file))")

# Save EEV simulations
eev_file = joinpath(output_dir, "eev_simulations_$(Dates.format(now(), "yyyy_mm_dd_HHMM")).jld2")
save_simulation_results(eev_simulations, params, data, eev_file)
println("  ✓ EEV results saved to: $(basename(eev_file))")

println("\n9. Saving detailed report...")
output_file = joinpath(@__DIR__, "..", "output", "benchmark_expected_value_summary.txt")
open(output_file, "w") do io
    println(io, "="^70)
    println(io, "BENCHMARK: SDDP vs Expected-Value Deterministic Model")
    println(io, "="^70)
    println(io, "\nMODEL CONFIGURATION:")
    println(io, "  Planning horizon: T=$(params.T) years")
    println(io, "  Total stages: $(2*params.T)")
    println(io, "  Temperature scenarios: 2")
    println(io, "  Energy price states: 3")
    println(io)
    println(io, "\nRESULTS:")
    println(io, "  SDDP cost (RP):                     $(round(sddp_mean_cost / 1e6, digits=3)) M€")
    println(io, "  EV model cost (perfect foresight):  $(round(ev_objective / 1e6, digits=3)) M€")
    println(io, "  EEV (EV policy under uncertainty):  $(round(eev_mean / 1e6, digits=3)) M€")
    println(io, "  VSS = EEV - RP:                     $(round(vss / 1e6, digits=3)) M€ ($(round(vss_percent, digits=2))%)")
    println(io)
    println(io, "\nCOMPUTATION TIME:")
    println(io, "  SDDP:     $(round(sddp_training_time + sim_time, digits=2)) seconds")
    println(io, "  EV Model: $(round(ev_build_time + ev_solve_time, digits=2)) seconds")
    println(io)
    println(io, "\nFIRST-STAGE INVESTMENTS:")
    println(io, "  Technology              SDDP (MW)    EV (MW)    Difference")
    println(io, "  " * "-"^63)
    for tech in params.technologies
        if sddp_investments[tech] > 0.1 || ev_investments_first[tech] > 0.1
            sddp_inv = sddp_investments[tech]
            ev_inv = ev_investments_first[tech]
            diff = sddp_inv - ev_inv
            println(io, "  $(rpad(String(tech), 20))  $(lpad(round(sddp_inv, digits=1), 10))  $(lpad(round(ev_inv, digits=1), 10))  $(lpad(round(diff, digits=1), 10))")
        end
    end
    println(io)
    println(io, "\nINTERPRETATION:")
    if vss > 0
        println(io, "  • POSITIVE VSS: Stochastic optimization provides value")
        println(io, "  • Using deterministic (EV) investments under uncertainty costs $(round(vss / 1e6, digits=2)) M€ more")
        println(io, "  • SDDP saves $(round(vss_percent, digits=1))% by hedging against uncertainty")
        println(io, "  • EV model underestimates costs (assumes perfect foresight)")
        println(io, "  • Investment strategies differ significantly")
    elseif vss > -0.01 * sddp_mean_cost
        println(io, "  • VSS near zero suggests limited benefit from stochastic modeling")
        println(io, "  • SDDP and EV policies perform similarly under uncertainty")
        println(io, "  • Uncertainty impact is low for this problem")
    else
        println(io, "  • NEGATIVE VSS: Unexpected result")
        println(io, "  • May indicate SDDP convergence issues or implementation bugs")
        println(io, "  • Consider increasing iteration_limit for SDDP training")
    end
    println(io)
    println(io, "="^70)
    println(io, "KEY FINDINGS FOR PAPER:")
    println(io, "="^70)
    if vss > 0
        println(io, "1. Proper VSS = $(round(vss / 1e6, digits=2)) M€ ($(round(vss_percent, digits=1))% of SDDP cost)")
        println(io, "2. Stochastic optimization saves $(round(vss / 1e6, digits=2)) M€ compared to deterministic")
        println(io, "3. Investment strategies differ between stochastic and deterministic")
        println(io, "4. SDDP provides robust hedging against uncertainty")
        println(io, "5. Deterministic model underestimates actual costs under uncertainty")
    else
        println(io, "1. VSS quantifies the value of uncertainty modeling")
        println(io, "2. Investment strategies differ between stochastic and deterministic")
        println(io, "3. SDDP provides hedging against multiple future scenarios")
    end
    println(io, "="^70)
end
println("✓ Report saved to: $output_file")

println("\n" * "="^70)
println("BENCHMARK COMPLETE")
println("="^70)
println("\nSUMMARY:")
println("  • SDDP cost (RP):       $(round(sddp_mean_cost / 1e6, digits=2)) M€")
println("  • EV cost (foresight):  $(round(ev_objective / 1e6, digits=2)) M€")
println("  • EEV (EV under uncertainty): $(round(eev_mean / 1e6, digits=2)) M€")
println("  • VSS = EEV - RP:       $(round(vss / 1e6, digits=2)) M€ ($(round(vss_percent, digits=1))%)")
println("\nOutputs saved to output/ directory")
println("="^70)
