"""
    ev_policy_evaluation.jl

Functions for evaluating the Expected-Value (EV) deterministic policy under uncertainty
to calculate the proper Value of Stochastic Solution (VSS).

VSS = EEV - RP
where:
- RP = Recourse Problem cost (SDDP optimized under uncertainty)
- EEV = Expected value of EV solution (deterministic investments evaluated under uncertainty)
"""

"""
    extract_ev_investments(ev_model, ev_variables, params)

Extract investment decisions from the solved deterministic (EV) model.

# Arguments
- `ev_model`: Solved deterministic JuMP model
- `ev_variables`: Dictionary of variables from deterministic model
- `params`: ModelParameters structure

# Returns
Dictionary mapping stage → Dict(:tech => Dict(tech => MW), :storage => MW)
Only extracts from vintage_stages (excludes last investment stage).
"""
function extract_ev_investments(ev_model, ev_variables, params)
    # Calculate vintage stages (same logic as model_builder.jl)
    last_inv_stage = 2 * params.T - 1
    vintage_stages = filter(s -> s != last_inv_stage, params.investment_stages)

    ev_investments = Dict()

    for t in vintage_stages
        # Extract technology investments
        tech_investments = Dict{Symbol, Float64}()
        for tech in params.technologies
            tech_investments[tech] = SDDP.JuMP.value(ev_variables[:inv_vars][t][:u_expansion_tech][tech])
        end

        # Extract storage investment
        storage_investment = SDDP.JuMP.value(ev_variables[:inv_vars][t][:u_expansion_storage])

        ev_investments[t] = Dict(
            :tech => tech_investments,
            :storage => storage_investment
        )
    end

    return ev_investments
end


"""
    evaluate_ev_policy(sddp_model, ev_investments, params, n_scenarios)

Evaluate the EV (deterministic) investment policy under uncertainty by:
1. Fixing investment decisions to EV values in the SDDP model
2. Simulating with fixed investments under uncertainty
3. Calculating expected cost (EEV)

# Arguments
- `sddp_model`: Trained SDDP PolicyGraph model
- `ev_investments`: Investment decisions from deterministic model (from extract_ev_investments)
- `params`: ModelParameters structure
- `n_scenarios`: Number of Monte Carlo simulations to run
- `verbose`: Print detailed diagnostics (default: false)
- `random_seed`: Optional random seed for reproducibility (default: nothing)

# Returns
Tuple of (simulations, eev_mean, eev_std) where:
- `simulations`: Full simulation results for detailed analysis
- `eev_mean`: Expected cost of using EV decisions under uncertainty (EEV)
- `eev_std`: Standard deviation of costs across scenarios
"""
function evaluate_ev_policy(sddp_model, ev_investments, params, n_scenarios; verbose=false, random_seed=nothing)
    # Set random seed BEFORE any model modifications for reproducibility
    # This ensures SDDP and EEV simulations use identical random scenarios
    if random_seed !== nothing
        Random.seed!(random_seed)
    end

    println("  Fixing EV investment decisions in SDDP model...")

    # Sort stages for consistent output
    stages_to_fix = sort(collect(keys(ev_investments)))
    println("    Fixing investment stages: $stages_to_fix")

    # Track nodes fixed per stage for diagnostics
    nodes_fixed_per_stage = Dict{Int, Int}()

    # Fix investment variables to EV decisions across ALL nodes of each investment stage
    # This correctly handles stage 1 (2 system temp nodes) and stages 3/5/7 (6 temp×energy nodes)
    for stage in stages_to_fix
        invs = ev_investments[stage]

        if verbose
            println("\n    Stage $stage investments to fix:")
            for tech in params.technologies
                if invs[:tech][tech] > 0.1
                    println("      $tech: $(round(invs[:tech][tech], digits=2)) MW")
                end
            end
            if invs[:storage] > 0.1
                println("      Storage: $(round(invs[:storage], digits=2)) MWh")
            end
        end

        # Iterate over all nodes in the graph, fixing those that belong to this stage
        n_nodes_fixed = 0
        for (node_index, node) in sddp_model.nodes
            if node_index[1] == stage  # This node belongs to the current investment stage
                sp = node.subproblem

                # Fix technology investments
                for tech in params.technologies
                    SDDP.JuMP.fix(sp[:u_expansion_tech][tech], invs[:tech][tech]; force=true)
                end

                # Fix storage investment
                SDDP.JuMP.fix(sp[:u_expansion_storage], invs[:storage]; force=true)

                n_nodes_fixed += 1
            end
        end

        nodes_fixed_per_stage[stage] = n_nodes_fixed
        if verbose
            println("      Fixed $n_nodes_fixed nodes at stage $stage")
        end
    end

    # Print summary
    total_nodes = sum(values(nodes_fixed_per_stage))
    println("    Successfully fixed investments in $total_nodes nodes total:")
    for stage in sort(collect(keys(nodes_fixed_per_stage)))
        println("      Stage $stage: $(nodes_fixed_per_stage[stage]) nodes")
    end

    println("  Running $(n_scenarios) simulations with fixed EV investments...")

    # Note: Random seed was already set at the beginning of the function

    # Simulate with fixed investments
    # Operations are still optimized based on realized uncertainty
    # Record key variables for analysis (matches run_simulation pattern)
    simulation_symbols = [:u_production, :u_expansion_tech, :u_expansion_storage, :u_unmet]
    simulations = SDDP.simulate(
        sddp_model,
        n_scenarios,
        simulation_symbols;
        parallel_scheme=SDDP.Threaded()
    )

    # Unfix investment variables to restore original SDDP model
    println("  Restoring SDDP model (unfixing investment variables)...")
    for stage in stages_to_fix
        # Iterate over all nodes in the graph, unfixing those that belong to this stage
        for (node_index, node) in sddp_model.nodes
            if node_index[1] == stage  # This node belongs to the current investment stage
                sp = node.subproblem

                # Unfix technology investments
                for tech in params.technologies
                    SDDP.JuMP.unfix(sp[:u_expansion_tech][tech])
                end

                # Unfix storage investment
                SDDP.JuMP.unfix(sp[:u_expansion_storage])
            end
        end
    end

    # Calculate EEV: expected cost across all simulations
    scenario_costs = [sum(stage[:stage_objective] for stage in sim) for sim in simulations]
    eev_mean = Statistics.mean(scenario_costs)
    eev_std = Statistics.std(scenario_costs)

    return simulations, eev_mean, eev_std
end
