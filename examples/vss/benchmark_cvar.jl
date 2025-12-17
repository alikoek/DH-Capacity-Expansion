"""
Benchmark: Risk-Averse VSS with CVaR

Compares:
- SDDP trained with CVaR(0.95) risk measure
- EV model (deterministic)

Reports both:
- Standard VSS (mean-based)
- Risk-adjusted VSS (CVaR-based)
"""

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

using DHCapEx
using SDDP
using Statistics
using Dates
using Random
using Plots
using StatsPlots

println("="^70)
println("BENCHMARK: Risk-Averse VSS with CVaR(0.95)")
println("="^70)

##############################################################################
# CONFIGURATION
##############################################################################

const ITERATION_LIMIT_SDDP = 250
const ITERATION_LIMIT_EV = 100
const N_SIMULATIONS = 500
const N_SIMULATIONS_EV = 10
const RANDOM_SEED = 12345
const CVAR_ALPHA = 0.95  # Focus on worst 5% of outcomes

# Branching structure
const LATE_TEMP_BRANCHING = true  # true = late branching (default), false = early branching

const OUTPUT_DIR = joinpath(@__DIR__, "..", "output")

##############################################################################
# Helper function to calculate CVaR
##############################################################################
function calculate_cvar(costs::Vector{Float64}, alpha::Float64)
    # CVaR at alpha level = expected value of worst (1-alpha) fraction
    sorted_costs = sort(costs, rev=true)  # Descending for costs (worst = highest)
    n_tail = max(1, Int(ceil((1 - alpha) * length(costs))))
    return mean(sorted_costs[1:n_tail])
end

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
println("Risk measure: CVaR($CVAR_ALPHA) - worst $(round((1-CVAR_ALPHA)*100, digits=0))%")
println("="^70)

##############################################################################
# PART 1: Risk-Averse SDDP
##############################################################################

println("\n" * "="^70)
println("PART 1: RISK-AVERSE SDDP (CVaR)")
println("="^70)

println("\n3. Building SDDP model...")
branching_str = LATE_TEMP_BRANCHING ? "LATE" : "EARLY"
println("   Temperature branching: $branching_str")
sddp_model = build_sddp_model(params, data; late_temp_branching=LATE_TEMP_BRANCHING)
println("   Nodes: $(length(sddp_model.nodes))")

println("\n4. Training SDDP model with CVaR($CVAR_ALPHA) ($ITERATION_LIMIT_SDDP iterations)...")
sddp_start_time = time()
SDDP.train(sddp_model;
    risk_measure=SDDP.CVaR(CVAR_ALPHA),  # Risk-averse!
    iteration_limit=ITERATION_LIMIT_SDDP,
    print_level=1,
    log_frequency=25,
    run_numerical_stability_report=true,  # Monitor coefficient ranges
    cut_deletion_minimum=100,  # Prevent cut accumulation causing numerical instability
    parallel_scheme=SDDP.Threaded()  # Enable multithreaded training
)
sddp_training_time = time() - sddp_start_time

println("\n5. Running SDDP simulations ($N_SIMULATIONS scenarios)...")
Random.seed!(RANDOM_SEED)
sddp_simulations = SDDP.simulate(sddp_model, N_SIMULATIONS,
    [:u_expansion_tech, :u_expansion_storage, :u_unmet];
    parallel_scheme=SDDP.Threaded()
)

# Calculate costs
sddp_costs = get_simulation_costs(sddp_simulations)
RP_mean = mean(sddp_costs)
RP_cvar = calculate_cvar(sddp_costs, CVAR_ALPHA)

println("\n" * "="^70)
println("SDDP RESULTS (Risk-Averse Policy)")
println("="^70)
println("Training time:      $(round(sddp_training_time, digits=2)) seconds")
println("Mean cost (RP):     $(round(RP_mean/1e6, digits=3)) MSEK")
println("CVaR cost (RP):     $(round(RP_cvar/1e6, digits=3)) MSEK")
println("Std deviation:      $(round(std(sddp_costs)/1e6, digits=3)) MSEK")
println("="^70)

##############################################################################
# PART 2: Expected-Value Model
##############################################################################

println("\n" * "="^70)
println("PART 2: EXPECTED-VALUE MODEL")
println("="^70)

println("\n6. Building EV SDDP model...")
ev_model, ev_params, ev_data = build_ev_sddp_model_integrated(params, data)

println("\n7. Training EV model ($ITERATION_LIMIT_EV iterations)...")
SDDP.train(ev_model;
    iteration_limit=ITERATION_LIMIT_EV,
    print_level=1,
    log_frequency=25,
    run_numerical_stability_report=false
)

println("\n8. Simulating EV model ($N_SIMULATIONS_EV scenarios)...")
ev_simulations = SDDP.simulate(ev_model, N_SIMULATIONS_EV,
    [:u_expansion_tech, :u_expansion_storage];
    parallel_scheme=SDDP.Threaded()
)

ev_costs = get_simulation_costs(ev_simulations)
EV_cost = mean(ev_costs)

println("\n" * "="^70)
println("EV MODEL RESULTS")
println("="^70)
println("EV objective:       $(round(EV_cost/1e6, digits=3)) MSEK")
println("="^70)

##############################################################################
# PART 3: EEV (EV Policy Under Uncertainty)
##############################################################################

println("\n" * "="^70)
println("PART 3: EVALUATING EV POLICY UNDER UNCERTAINTY (EEV)")
println("="^70)

println("\n9. Extracting EV investments...")
ev_investments = extract_ev_investments_from_simulations(ev_simulations, params)

println("\n10. Evaluating EV policy under uncertainty ($N_SIMULATIONS scenarios)...")
eev_simulations, _ = evaluate_ev_policy(
    sddp_model, ev_investments, params, N_SIMULATIONS;
    random_seed=RANDOM_SEED
)

eev_costs = get_simulation_costs(eev_simulations)
EEV_mean = mean(eev_costs)
EEV_cvar = calculate_cvar(eev_costs, CVAR_ALPHA)

println("\n" * "="^70)
println("EEV RESULTS")
println("="^70)
println("Mean cost (EEV):    $(round(EEV_mean/1e6, digits=3)) MSEK")
println("CVaR cost (EEV):    $(round(EEV_cvar/1e6, digits=3)) MSEK")
println("Std deviation:      $(round(std(eev_costs)/1e6, digits=3)) MSEK")
println("="^70)

##############################################################################
# PART 4: VSS Calculations
##############################################################################

println("\n" * "="^70)
println("VALUE OF STOCHASTIC SOLUTION (VSS)")
println("="^70)

# Standard VSS (mean-based)
VSS_mean = EEV_mean - RP_mean
VSS_mean_pct = 100 * VSS_mean / RP_mean

# Risk-adjusted VSS (CVaR-based)
VSS_cvar = EEV_cvar - RP_cvar
VSS_cvar_pct = 100 * VSS_cvar / RP_cvar

println("\nMEAN-BASED VSS (Standard):")
println("  RP (mean):              $(round(RP_mean/1e6, digits=3)) MSEK")
println("  EEV (mean):             $(round(EEV_mean/1e6, digits=3)) MSEK")
println("  VSS = EEV - RP:         $(round(VSS_mean/1e6, digits=3)) MSEK ($(round(VSS_mean_pct, digits=2))%)")

println("\nCVaR-BASED VSS (Risk-Adjusted):")
println("  RP (CVaR):              $(round(RP_cvar/1e6, digits=3)) MSEK")
println("  EEV (CVaR):             $(round(EEV_cvar/1e6, digits=3)) MSEK")
println("  VSS_CVaR = EEV - RP:    $(round(VSS_cvar/1e6, digits=3)) MSEK ($(round(VSS_cvar_pct, digits=2))%)")

println("\n" * "="^70)
println("INTERPRETATION")
println("="^70)
println("- Mean-based VSS shows value under average scenarios")
println("- CVaR-based VSS shows value in worst $(round((1-CVAR_ALPHA)*100, digits=0))% scenarios")
if VSS_cvar > VSS_mean
    println("- Risk-averse VSS is HIGHER: SDDP provides more value in tail scenarios")
else
    println("- Risk-averse VSS is similar to mean-based VSS")
end

##############################################################################
# PART 5: Visualizations
##############################################################################

println("\n" * "="^70)
println("PART 5: CREATING VISUALIZATIONS")
println("="^70)

# Extract first-stage investments for comparison
sddp_investments = Dict{Symbol,Float64}()
for tech in params.technologies
    inv_values = [sim[1][:u_expansion_tech][tech] for sim in sddp_simulations]
    sddp_investments[tech] = mean(inv_values)
end

ev_investments_first = ev_investments[1][:tech]

# Plot 1: Cost distribution comparison (violin plot)
p1 = violin([1], sddp_costs / 1e6, label="CVaR-SDDP", fillalpha=0.7, color=:steelblue)
violin!([2], eev_costs / 1e6, label="EV Policy", fillalpha=0.7, color=:coral)
xlabel!("Policy")
ylabel!("Total Cost (MSEK)")
title!("Cost Distribution: CVaR-SDDP vs EV Policy")
xticks!([1, 2], ["CVaR-SDDP\n(RP)", "EV Policy\n(EEV)"])

# Add mean and CVaR lines
hline!([RP_mean / 1e6], color=:steelblue, linestyle=:dash, label="RP mean", linewidth=2)
hline!([RP_cvar / 1e6], color=:steelblue, linestyle=:dot, label="RP CVaR", linewidth=2)
hline!([EEV_mean / 1e6], color=:coral, linestyle=:dash, label="EEV mean", linewidth=2)
hline!([EEV_cvar / 1e6], color=:coral, linestyle=:dot, label="EEV CVaR", linewidth=2)

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
        ylabel="Investment (MW)",
        title="First-Stage Investments",
        label=["CVaR-SDDP" "EV Model"],
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
    plot_title="Risk-Averse VSS Analysis: CVaR-SDDP vs EV Model")

output_plot = joinpath(OUTPUT_DIR, "benchmark_cvar_comparison.png")
savefig(combined, output_plot)
println("Comparison plot saved to: $output_plot")

##############################################################################
# PART 6: Save Results
##############################################################################

println("\n" * "="^70)
println("PART 6: SAVING RESULTS")
println("="^70)

# Save simulation results for future analysis
println("\nSaving simulation data...")
sddp_file = joinpath(OUTPUT_DIR, "cvar_sddp_simulations_$(Dates.format(now(), "yyyy_mm_dd_HHMM")).jld2")
save_simulation_results(sddp_simulations, params, data, sddp_file)
println("  CVaR-SDDP results saved to: $(basename(sddp_file))")

eev_file = joinpath(OUTPUT_DIR, "cvar_eev_simulations_$(Dates.format(now(), "yyyy_mm_dd_HHMM")).jld2")
save_simulation_results(eev_simulations, params, data, eev_file)
println("  EEV results saved to: $(basename(eev_file))")

# Save detailed text report
println("\nSaving detailed report...")
report_path = joinpath(OUTPUT_DIR, "benchmark_cvar_summary.txt")
open(report_path, "w") do io
    println(io, "="^70)
    println(io, "RISK-AVERSE VSS BENCHMARK RESULTS")
    println(io, "="^70)
    println(io, "Generated: $(Dates.now())")
    println(io, "\nCONFIGURATION:")
    println(io, "  T: $(params.T) years")
    println(io, "  Risk measure: CVaR($CVAR_ALPHA)")
    println(io, "  SDDP iterations: $ITERATION_LIMIT_SDDP")
    println(io, "  Simulations: $N_SIMULATIONS")
    println(io, "\nRESULTS:")
    println(io, "  RP (mean):     $(round(RP_mean/1e6, digits=3)) MSEK")
    println(io, "  RP (CVaR):     $(round(RP_cvar/1e6, digits=3)) MSEK")
    println(io, "  EV:            $(round(EV_cost/1e6, digits=3)) MSEK")
    println(io, "  EEV (mean):    $(round(EEV_mean/1e6, digits=3)) MSEK")
    println(io, "  EEV (CVaR):    $(round(EEV_cvar/1e6, digits=3)) MSEK")
    println(io, "\nVSS:")
    println(io, "  VSS (mean):    $(round(VSS_mean/1e6, digits=3)) MSEK ($(round(VSS_mean_pct, digits=2))%)")
    println(io, "  VSS (CVaR):    $(round(VSS_cvar/1e6, digits=3)) MSEK ($(round(VSS_cvar_pct, digits=2))%)")
    println(io, "\nFIRST-STAGE INVESTMENTS:")
    println(io, "  Technology              CVaR-SDDP (MW)    EV (MW)    Difference")
    println(io, "  " * "-"^63)
    for tech in params.technologies
        if sddp_investments[tech] > 0.1 || ev_investments_first[tech] > 0.1
            sddp_inv = sddp_investments[tech]
            ev_inv = ev_investments_first[tech]
            diff = sddp_inv - ev_inv
            println(io, "  $(rpad(String(tech), 20))  $(lpad(round(sddp_inv, digits=1), 14))  $(lpad(round(ev_inv, digits=1), 10))  $(lpad(round(diff, digits=1), 10))")
        end
    end
    println(io, "\nINTERPRETATION:")
    if VSS_cvar > VSS_mean
        println(io, "  - CVaR-based VSS is HIGHER than mean-based VSS")
        println(io, "  - Risk-averse SDDP provides $(round((VSS_cvar - VSS_mean)/1e6, digits=2)) MSEK extra value in tail scenarios")
    else
        println(io, "  - CVaR-based VSS is similar to or lower than mean-based VSS")
    end
    println(io, "="^70)
end
println("Report saved to: $report_path")

println("\n" * "="^70)
println("BENCHMARK COMPLETE")
println("="^70)
println("\nOutputs saved to: $OUTPUT_DIR")
println("  - benchmark_cvar_comparison.png (visualization)")
println("  - benchmark_cvar_summary.txt (text report)")
println("  - cvar_sddp_simulations_*.jld2 (SDDP simulation data)")
println("  - cvar_eev_simulations_*.jld2 (EEV simulation data)")
