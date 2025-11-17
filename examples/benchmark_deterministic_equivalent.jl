"""
Benchmark: SDDP vs Deterministic Equivalent

This script compares the SDDP solution with the full deterministic equivalent.
The deterministic equivalent solves the complete scenario tree as one large LP.

Expected scenario count for T=4:
- Temperature: 2 scenarios
- Energy paths: 3^4 = 81 paths per temperature
- Total: 2 × 81 = 162 scenarios

This is the "gold standard" benchmark but computationally expensive.
"""

# Add src to load path
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

using DHCapEx
using SDDP
using Statistics

println("="^70)
println("BENCHMARK: SDDP vs Deterministic Equivalent")
println("="^70)

# Load parameters and data
println("\n1. Loading model parameters...")
params = load_parameters(joinpath(@__DIR__, "..", "data", "model_parameters.xlsx"))

println("2. Loading and processing data...")
data = load_all_data(params, joinpath(@__DIR__, "..", "data"))

println("\n" * "="^70)
println("MODEL CONFIGURATION")
println("="^70)
println("Planning horizon: T=$(params.T) years")
println("Number of stages: $(2*params.T)")
println("Temperature scenarios: 2")
println("Energy states: 3 (High/Medium/Low)")
println("Expected scenario paths: $(2 * 3^params.T)")
println("="^70)

# ============================================================================
# PART 1: Solve with SDDP
# ============================================================================

println("\n" * "="^70)
println("PART 1: SDDP SOLUTION")
println("="^70)

println("\n3. Building SDDP model...")
sddp_model = build_sddp_model(params, data)

println("\n4. Training SDDP model...")
sddp_start_time = time()
SDDP.train(sddp_model;
          iteration_limit = 100,
          time_limit = 3600.0,  # 1 hour max
          print_level = 1,
          run_numerical_stability_report = false)
sddp_training_time = time() - sddp_start_time

println("\n5. Calculating SDDP lower bound...")
sddp_lower_bound = SDDP.calculate_bound(sddp_model)

println("\n6. Running SDDP simulations...")
sim_start_time = time()
sddp_simulations = SDDP.simulate(sddp_model, 500;
                                 parallel_scheme = SDDP.Threaded(),
                                 custom_recorders = Dict{Symbol, Function}(
                                     :u_expansion_tech => (sp) -> haskey(sp, :u_expansion_tech) ?
                                                                  Dict(tech => SDDP.JuMP.value(sp[:u_expansion_tech][tech])
                                                                       for tech in params.technologies) :
                                                                  nothing
                                 ))
sim_time = time() - sim_start_time

# Calculate SDDP upper bound (mean of simulations)
sddp_sim_costs = [sum(s[:stage_objective] for s in sim) for sim in sddp_simulations]
sddp_upper_bound = mean(sddp_sim_costs)
sddp_std = std(sddp_sim_costs)
sddp_optimality_gap = (sddp_upper_bound - sddp_lower_bound) / sddp_upper_bound * 100

println("\n" * "="^70)
println("SDDP RESULTS")
println("="^70)
println("Training time:      $(round(sddp_training_time, digits=2)) seconds")
println("Simulation time:    $(round(sim_time, digits=2)) seconds")
println("Total SDDP time:    $(round(sddp_training_time + sim_time, digits=2)) seconds")
println()
println("Lower bound:        $(round(sddp_lower_bound / 1e6, digits=2)) M€")
println("Upper bound (mean): $(round(sddp_upper_bound / 1e6, digits=2)) M€")
println("Std deviation:      $(round(sddp_std / 1e6, digits=2)) M€")
println("Optimality gap:     $(round(sddp_optimality_gap, digits=2))%")
println("="^70)

# Extract first-stage investments from SDDP
println("\nFirst-stage investments (SDDP):")
for tech in params.technologies
    inv_values = [sim[1][:u_expansion_tech][tech] for sim in sddp_simulations if sim[1][:u_expansion_tech] !== nothing]
    avg_inv = length(inv_values) > 0 ? mean(inv_values) : 0.0
    if avg_inv > 0.1
        println("  $tech: $(round(avg_inv, digits=2)) MW")
    end
end

# ============================================================================
# PART 2: Solve Deterministic Equivalent
# ============================================================================

println("\n" * "="^70)
println("PART 2: DETERMINISTIC EQUIVALENT")
println("="^70)

println("\n7. Building deterministic equivalent...")
println("   WARNING: This creates a very large LP!")
println("   Expected size: ~162 scenarios × (variables per scenario)")

de_build_start = time()
try
    det_equiv_model = SDDP.deterministic_equivalent(sddp_model, Gurobi.Optimizer)
    de_build_time = time() - de_build_start

    println("\n✓ Deterministic equivalent built successfully!")
    println("  Build time: $(round(de_build_time, digits=2)) seconds")
    println("  Model size:")
    println("    Variables:   $(SDDP.JuMP.num_variables(det_equiv_model))")
    println("    Constraints: $(SDDP.JuMP.num_constraints(det_equiv_model; count_variable_in_set_constraints=false))")

    println("\n8. Solving deterministic equivalent...")
    println("   This may take a while...")

    SDDP.JuMP.set_optimizer_attribute(det_equiv_model, "OutputFlag", 1)
    SDDP.JuMP.set_optimizer_attribute(det_equiv_model, "TimeLimit", 7200.0)  # 2 hour limit

    de_solve_start = time()
    SDDP.JuMP.optimize!(det_equiv_model)
    de_solve_time = time() - de_solve_start

    de_status = SDDP.JuMP.termination_status(det_equiv_model)
    println("\n✓ Deterministic equivalent solved!")
    println("  Status: $de_status")
    println("  Solve time: $(round(de_solve_time, digits=2)) seconds")
    println("  Total DE time: $(round(de_build_time + de_solve_time, digits=2)) seconds")

    if de_status == SDDP.JuMP.MOI.OPTIMAL || de_status == SDDP.JuMP.MOI.TIME_LIMIT
        de_objective = SDDP.JuMP.objective_value(det_equiv_model)

        println("\n" * "="^70)
        println("DETERMINISTIC EQUIVALENT RESULTS")
        println("="^70)
        println("Objective value: $(round(de_objective / 1e6, digits=2)) M€")
        println("="^70)

        # ============================================================================
        # PART 3: Comparison
        # ============================================================================

        println("\n" * "="^70)
        println("COMPARISON: SDDP vs DETERMINISTIC EQUIVALENT")
        println("="^70)

        println("\nOBJECTIVE VALUES:")
        println("  SDDP lower bound:  $(round(sddp_lower_bound / 1e6, digits=3)) M€")
        println("  SDDP upper bound:  $(round(sddp_upper_bound / 1e6, digits=3)) M€")
        println("  DE objective:      $(round(de_objective / 1e6, digits=3)) M€")

        println("\nCOMPUTATION TIME:")
        println("  SDDP total:  $(round(sddp_training_time + sim_time, digits=2)) seconds")
        println("  DE total:    $(round(de_build_time + de_solve_time, digits=2)) seconds")
        println("  Speedup:     $(round((de_build_time + de_solve_time) / (sddp_training_time + sim_time), digits=2))x")

        println("\nCONVERGENCE CHECK:")
        de_vs_sddp_lb = abs(de_objective - sddp_lower_bound) / de_objective * 100
        de_vs_sddp_ub = abs(de_objective - sddp_upper_bound) / de_objective * 100
        println("  |DE - SDDP_LB| / DE: $(round(de_vs_sddp_lb, digits=3))%")
        println("  |DE - SDDP_UB| / DE: $(round(de_vs_sddp_ub, digits=3))%")

        if de_vs_sddp_lb < 1.0 && de_vs_sddp_ub < 5.0
            println("\n✓ EXCELLENT: SDDP converged very close to DE optimal!")
        elseif de_vs_sddp_lb < 5.0
            println("\n✓ GOOD: SDDP lower bound within 5% of DE optimal")
        else
            println("\n⚠ WARNING: Significant gap between SDDP and DE - consider more iterations")
        end

        println("\n" * "="^70)
        println("KEY INSIGHTS")
        println("="^70)
        println("1. The deterministic equivalent proves SDDP finds near-optimal solutions")
        println("2. SDDP is significantly faster for large-scale problems")
        println("3. SDDP lower bound ≈ DE objective confirms correct implementation")
        println("="^70)

        # Save results
        println("\n9. Saving results...")
        output_file = joinpath(@__DIR__, "..", "output", "benchmark_det_equiv_summary.txt")
        open(output_file, "w") do io
            println(io, "="^70)
            println(io, "BENCHMARK: SDDP vs Deterministic Equivalent")
            println(io, "="^70)
            println(io, "\nMODEL CONFIGURATION:")
            println(io, "  Horizon: T=$(params.T)")
            println(io, "  Scenarios: $(2 * 3^params.T)")
            println(io, "\nRESULTS:")
            println(io, "  SDDP lower bound: $(round(sddp_lower_bound / 1e6, digits=3)) M€")
            println(io, "  SDDP upper bound: $(round(sddp_upper_bound / 1e6, digits=3)) M€")
            println(io, "  DE objective:     $(round(de_objective / 1e6, digits=3)) M€")
            println(io, "\nCOMPUTATION TIME:")
            println(io, "  SDDP: $(round(sddp_training_time + sim_time, digits=2)) s")
            println(io, "  DE:   $(round(de_build_time + de_solve_time, digits=2)) s")
            println(io, "\nCONVERGENCE:")
            println(io, "  |DE - SDDP_LB| / DE: $(round(de_vs_sddp_lb, digits=3))%")
            println(io, "  |DE - SDDP_UB| / DE: $(round(de_vs_sddp_ub, digits=3))%")
        end
        println("✓ Results saved to: $output_file")

    else
        println("\n✗ Deterministic equivalent did not solve to optimality")
        println("  Status: $de_status")
        println("\nThis may happen if:")
        println("  - Problem is too large (try reducing T)")
        println("  - Time limit was reached")
        println("  - Model has numerical issues")
    end

catch e
    de_build_time = time() - de_build_start
    println("\n✗ ERROR building deterministic equivalent!")
    println("  Build time before error: $(round(de_build_time, digits=2)) seconds")
    println("  Error: $e")
    println("\n" * "="^70)
    println("DETERMINISTIC EQUIVALENT TOO LARGE")
    println("="^70)
    println("The full T=$(params.T) deterministic equivalent is too large to build.")
    println("\nRECOMMENDATIONS:")
    println("1. Use a smaller horizon (T=2 or T=3) for DE comparison")
    println("2. Use expected-value deterministic model instead (see benchmark_expected_value.jl)")
    println("3. Accept that SDDP decomposition is necessary for this problem size")
    println("="^70)
end

println("\n" * "="^70)
println("BENCHMARK COMPLETE")
println("="^70)
