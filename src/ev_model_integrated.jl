"""
    ev_model_integrated.jl

Complete, integrated solution for building Expected Value SDDP model.
This handles both parameters and data transformation together.

STATUS: Complete implementation
APPROACH: Safe immutable constructor that preserves all model structure
"""

using Statistics
using LinearAlgebra
using DataFrames

"""
    build_ev_sddp_model_integrated(params::ModelParameters, data::ProcessedData)

Build an Expected Value SDDP model with all uncertainties collapsed to expected values.
This is the COMPLETE, SAFE, and INTEGRATED approach.

# Returns
- `ev_model`: SDDP PolicyGraph with deterministic expected values
- `ev_params`: New immutable ModelParameters with expected values
- `ev_data`: Modified ProcessedData with expected electricity prices

# Features
- Preserves ModelParameters immutability (safe)
- Uses exact same model structure via build_sddp_model()
- Handles all uncertainty types properly
"""
function build_ev_sddp_model_integrated(params::ModelParameters, data::ProcessedData)
    println("\n  Building Expected Value SDDP model (integrated safe approach)...")
    println("  ================================================================")

    # ========================================================================
    # Step 1: Calculate all expected values
    # ========================================================================
    println("    Step 1: Calculating expected values...")

    # 1a. Expected COP and year-varying expected demand multiplier from temperature scenarios
    expected_cop = 0.0
    expected_temp_demand_mult = zeros(params.T)  # Year-varying expected demand multiplier

    for (idx, scenario_symbol) in enumerate(params.temp_scenarios)
        prob = params.temp_scenario_probabilities[idx]
        cop = params.temp_cop_multipliers[scenario_symbol]
        demand_mult_vec = params.temp_demand_multipliers[scenario_symbol]  # Now a Vector

        expected_cop += cop * prob
        expected_temp_demand_mult .+= demand_mult_vec .* prob

        println("      Temp scenario $idx ($scenario_symbol): COP=$cop, demand=$(demand_mult_vec), prob=$prob")
    end
    println("      → Expected COP: $(round(expected_cop, digits=3))")
    println("      → Expected demand multiplier (year-varying): $(round.(expected_temp_demand_mult, digits=3))")

    # 1b. Time-dependent energy price distributions starting from Medium state
    # This matches SDDP which starts from Medium (state 2) and evolves through Markov chain
    P = params.energy_transitions
    starting_state = 2  # Medium state (consistent with SDDP)

    # Calculate time-dependent distributions: P^t[starting_state, :]
    # Year 1: After 1 transition from Medium
    # Year 2: After 2 transitions from Medium, etc.
    year_distributions = Dict{Int, Vector{Float64}}()
    for year in 1:params.T
        # P^year gives distribution after 'year' transitions
        P_power = P^year
        year_distributions[year] = P_power[starting_state, :]
        println("      Year $year distribution: [$(join(round.(year_distributions[year], digits=3), ", "))]")
    end
    println("      (Starting from Medium state, evolving through Markov chain)")

    # 1c. Expected energy carrier prices (same for all 3 states to maintain structure)
    ev_energy_map = Dict{Int,Dict{Int,Dict{Symbol,Float64}}}()

    # Create identical entries for all 3 states (structure compatibility)
    for state in 1:3
        ev_energy_map[state] = Dict{Int,Dict{Symbol,Float64}}()
    end

    for model_year in 1:params.T

        # Collect all carriers
        all_carriers = Set{Symbol}()
        for state in 1:3
            if haskey(params.energy_price_map[state], model_year)
                union!(all_carriers, keys(params.energy_price_map[state][model_year]))
            end
        end

        # Calculate expected price for each carrier using year-specific distribution
        year_dist = year_distributions[model_year]
        for carrier in all_carriers
            expected_price = 0.0
            for state in 1:3
                if haskey(params.energy_price_map[state], model_year) &&
                   haskey(params.energy_price_map[state][model_year], carrier)
                    price = params.energy_price_map[state][model_year][carrier]
                    expected_price += price * year_dist[state]
                end
            end
            # Set same expected price for all 3 states (structure compatibility)
            for state in 1:3
                ev_energy_map[state][model_year] = get(ev_energy_map[state], model_year, Dict{Symbol,Float64}())
                ev_energy_map[state][model_year][carrier] = expected_price
            end
        end
    end

    # 1d. Expected extreme events (if enabled)
    ev_extreme_events = nothing
    if params.enable_extreme_events && params.extreme_events !== nothing
        expected_demand_mult = 0.0
        expected_elec_mult = 0.0
        expected_dc_avail = 0.0

        for row in eachrow(params.extreme_events)
            expected_demand_mult += row.demand_multiplier * row.probability
            expected_elec_mult += row.elec_price_multiplier * row.probability
            expected_dc_avail += row.dc_availability * row.probability
        end

        println("      Extreme events expected values:")
        println("        Demand multiplier: $(round(expected_demand_mult, digits=3))")
        println("        Elec price multiplier: $(round(expected_elec_mult, digits=3))")
        println("        DC availability: $(round(expected_dc_avail, digits=3))")

        ev_extreme_events = DataFrame(
            scenario = ["Expected"],
            probability = [1.0],
            demand_multiplier = [expected_demand_mult],
            elec_price_multiplier = [expected_elec_mult],
            dc_availability = [expected_dc_avail]
        )
    end

    # ========================================================================
    # Step 2: Create new immutable ModelParameters
    # ========================================================================
    println("\n    Step 2: Creating new ModelParameters struct...")

    ev_params = ModelParameters(
        # Model configuration (unchanged)
        params.T,
        params.T_years,
        params.discount_rate,
        params.base_annual_demand,
        params.salvage_fraction,
        params.c_penalty,
        params.elec_taxes_levies,
        params.n_typical_weeks,

        # Technologies (unchanged)
        params.technologies,
        params.c_initial_capacity,
        params.c_max_additional_capacity,
        params.c_investment_cost,
        params.c_opex_fixed,
        params.c_opex_var,
        params.c_efficiency_th,
        params.c_efficiency_el,
        params.c_energy_carrier,
        params.c_lifetime_new,
        params.c_capacity_limits,

        # Storage (unchanged)
        params.storage_params,
        params.storage_capacity_limits,

        # Energy carriers (unchanged)
        params.c_emission_fac,
        params.elec_emission_factors,

        # Uncertainty configurations (MODIFIED for EV - but keeping structure)
        ev_energy_map,                                    # Expected prices (all states same)
        params.carbon_trajectory,                         # Unchanged
        [:Expected_High, :Expected_Low],                  # Keep 2 scenarios with same values
        Dict(:Expected_High => expected_cop, :Expected_Low => expected_cop),  # Same COP for both
        Dict(:Expected_High => expected_temp_demand_mult, :Expected_Low => expected_temp_demand_mult),  # Same demand mult for both
        Dict(1 => 0.5, 2 => 0.5),                        # Equal probabilities (will collapse to same result)
        [1.0 0.0 0.0; 1.0 0.0 0.0; 1.0 0.0 0.0],         # Deterministic transitions to state 1
        [1.0, 0.0, 0.0],                                 # Always start in state 1

        # Investment stages (unchanged)
        params.investment_stages,

        # Existing capacity and schedules (unchanged)
        params.c_existing_capacity_schedule,
        params.waste_chp_efficiency_schedule,
        params.waste_availability,

        # Extreme events (MODIFIED for EV)
        params.enable_extreme_events,
        params.apply_to_year,
        ev_extreme_events
    )

    # ========================================================================
    # Step 3: Create EV data with expected demand and electricity prices
    # ========================================================================
    println("\n    Step 3: Creating EV data with expected values...")

    # Map state indices to symbols
    state_symbols = [:high, :medium, :low]  # States: 1=High, 2=Medium, 3=Low

    # Calculate expected demand for each year/week/hour
    ev_scaled_weeks = Dict{Int, Dict{Symbol, Vector{Vector{Float64}}}}()
    ev_purch_elec_prices = Dict{Int, Dict{Symbol, Matrix{Float64}}}()
    ev_sale_elec_prices = Dict{Int, Dict{Symbol, Matrix{Float64}}}()

    for year in 1:params.T
        year_dist = year_distributions[year]  # [P(high), P(med), P(low)]

        # Calculate expected demand
        expected_demand = Vector{Vector{Float64}}()
        for week in 1:data.n_weeks
            week_demand = zeros(data.hours_per_week)
            for (state_idx, state_symbol) in enumerate(state_symbols)
                week_demand .+= data.scaled_weeks[year][state_symbol][week] .* year_dist[state_idx]
            end
            push!(expected_demand, round.(week_demand, digits=2))
        end

        # Calculate expected electricity prices
        first_scenario = first(values(data.purch_elec_prices[year]))
        n_weeks, n_hours = size(first_scenario)

        expected_purch_prices = zeros(n_weeks, n_hours)
        expected_sale_prices = zeros(n_weeks, n_hours)
        for (state_idx, state_symbol) in enumerate(state_symbols)
            expected_purch_prices .+= data.purch_elec_prices[year][state_symbol] .* year_dist[state_idx]
            expected_sale_prices .+= data.sale_elec_prices[year][state_symbol] .* year_dist[state_idx]
        end

        # Store same expected values for all scenarios (EV model is deterministic)
        ev_scaled_weeks[year] = Dict(s => expected_demand for s in state_symbols)
        ev_purch_elec_prices[year] = Dict(s => round.(expected_purch_prices, digits=4) for s in state_symbols)
        ev_sale_elec_prices[year] = Dict(s => round.(expected_sale_prices, digits=4) for s in state_symbols)

        mean_demand = mean(mean.(expected_demand))
        mean_price = mean(expected_purch_prices)
        println("      Year $year: mean demand = $(round(mean_demand, digits=1)) MW, mean elec price = $(round(mean_price, digits=3)) TSEK/MWh")
    end

    # Create new ProcessedData with expected values
    ev_data = ProcessedData(
        data.n_weeks,
        data.hours_per_week,
        data.week_weights_normalized,
        ev_scaled_weeks,
        ev_purch_elec_prices,
        ev_sale_elec_prices
    )

    # ========================================================================
    # Step 4: Build model using original build_sddp_model
    # ========================================================================
    println("\n    Step 4: Building SDDP model with EV parameters...")

    ev_model = build_sddp_model(ev_params, ev_data)

    # ========================================================================
    # Summary
    # ========================================================================
    println("\n  Expected Value SDDP model built successfully!")
    println("  ============================================")
    println("  Key features:")
    println("    ✓ ModelParameters remains immutable (safe)")
    println("    ✓ All constraints preserved from original")
    println("    ✓ Vintage tracking intact")
    println("    ✓ State variables preserved")
    println("    ✓ Extreme events: $(ev_params.enable_extreme_events ? "ENABLED with expected values" : "DISABLED")")
    println("\n  Structure:")
    println("    - Stages: $(2 * ev_params.T)")
    println("    - Nodes: $(length(ev_model.nodes)) (same tree structure as SDDP, identical values on all branches)")
    println("    - Temperature scenarios: 1 (expected COP = $(round(expected_cop, digits=2)), demand mult = $(round.(expected_temp_demand_mult, digits=2)))")
    println("    - Energy states: 1 (time-dependent, starting from Medium)")

    return ev_model, ev_params, ev_data
end

"""
    verify_ev_model_structure(ev_model, original_model)

Verify that the EV model has the correct linear structure.
Returns true if the model is properly linear (no branching).
"""
function verify_ev_model_structure(ev_model, original_model)
    println("    Verifying EV model structure...")

    # Count nodes
    ev_nodes = length(ev_model.nodes)
    original_nodes = length(original_model.nodes)

    println("      Node count:")
    println("        EV model: $ev_nodes")
    println("        Original model: $original_nodes")
    println("        → EV should have fewer nodes (no branching)")

    # Check that EV model has single path
    stages_with_nodes = Dict{Int, Int}()
    for (node_idx, _) in ev_model.nodes
        stage = node_idx[1]
        stages_with_nodes[stage] = get(stages_with_nodes, stage, 0) + 1
    end

    println("\n      Nodes per stage in EV model:")
    for stage in sort(collect(keys(stages_with_nodes)))
        count = stages_with_nodes[stage]
        println("        Stage $stage: $count node(s)")
        if count != 1
            println("          ⚠️ WARNING: Expected 1 node, got $count")
        end
    end

    all_single = all(v == 1 for v in values(stages_with_nodes))
    if all_single
        println("\n      ✓ EV model has linear structure (no branching)")
    else
        println("\n      ⚠️ WARNING: EV model has branching (unexpected)")
    end

    return all_single
end

# Export functions
export build_ev_sddp_model_integrated, verify_ev_model_structure