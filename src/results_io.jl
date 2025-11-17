"""
Results I/O module for saving and loading simulation results

This module provides functionality to save simulation results to disk and load them
back for analysis without rerunning the model. Uses JLD2 for efficient binary storage.
"""

using JLD2
using Dates

"""
    save_simulation_results(simulations, params::ModelParameters, data::ProcessedData, filepath::String)

Save simulation results to a JLD2 file for later analysis.

# Arguments
- `simulations`: Vector of simulation results from SDDP.simulate
- `params::ModelParameters`: Model parameters used in the simulation
- `data::ProcessedData`: Processed data used in the simulation
- `filepath::String`: Path where the results will be saved

# Example
```julia
simulations = run_simulation(model, params, data)
save_simulation_results(simulations, params, data, "output/results_2024.jld2")
```
"""
function save_simulation_results(simulations, params::ModelParameters, data::ProcessedData, filepath::String)
    # Ensure output directory exists
    dir = dirname(filepath)
    if !isdir(dir)
        mkpath(dir)
    end

    # Create metadata for the save file
    metadata = Dict(
        "timestamp" => now(),
        "julia_version" => VERSION,
        "n_simulations" => length(simulations),
        "n_stages" => params.T * 2,
        "planning_horizon_years" => params.T,
        "model_years_per_period" => params.T_years,
        "technologies" => params.technologies,
        "enable_extreme_events" => params.enable_extreme_events,
        "extreme_events_year" => params.enable_extreme_events ? params.apply_to_year : nothing,
        "risk_measure" => "Unknown",  # Would need to be passed from run_simulation
        "base_annual_demand" => params.base_annual_demand,
        "discount_rate" => params.discount_rate,
        "n_representative_weeks" => data.n_weeks
    )

    # Save everything to JLD2 file
    @save filepath simulations params data metadata

    println("✓ Simulation results saved to: $filepath")
    println("  - $(length(simulations)) simulations")
    println("  - $(params.T * 2) stages")
    println("  - File size: $(round(filesize(filepath) / 1024^2, digits=2)) MB")

    return filepath
end

"""
    save_simulation_results_auto(simulations, params::ModelParameters, data::ProcessedData;
                                 output_dir::String="output")

Save simulation results with an auto-generated timestamped filename.

# Arguments
- `simulations`: Vector of simulation results from SDDP.simulate
- `params::ModelParameters`: Model parameters used in the simulation
- `data::ProcessedData`: Processed data used in the simulation
- `output_dir::String`: Directory where results will be saved (default: "output")

# Returns
- `filepath::String`: The path where the file was saved

# Example
```julia
filepath = save_simulation_results_auto(simulations, params, data)
# Saves to: output/simulation_results_2024_11_17_1430.jld2
```
"""
function save_simulation_results_auto(simulations, params::ModelParameters, data::ProcessedData;
                                      output_dir::String="output")
    # Generate timestamped filename
    timestamp_str = Dates.format(now(), "yyyy_mm_dd_HHMM")

    # Add extreme events indicator to filename if enabled
    if params.enable_extreme_events
        filename = "simulation_results_$(timestamp_str)_with_extreme.jld2"
    else
        filename = "simulation_results_$(timestamp_str).jld2"
    end

    filepath = joinpath(output_dir, filename)
    return save_simulation_results(simulations, params, data, filepath)
end

"""
    load_simulation_results(filepath::String)

Load simulation results from a JLD2 file.

# Arguments
- `filepath::String`: Path to the saved results file

# Returns
- `simulations`: The simulation results
- `params`: Model parameters used in the simulation
- `data`: Processed data used in the simulation
- `metadata`: Metadata about the simulation run

# Example
```julia
simulations, params, data, metadata = load_simulation_results("output/results_2024.jld2")
println("Loaded \$(metadata["n_simulations"]) simulations from \$(metadata["timestamp"])")
```
"""
function load_simulation_results(filepath::String)
    if !isfile(filepath)
        error("File not found: $filepath")
    end

    # Load from JLD2 file
    @load filepath simulations params data metadata

    println("✓ Loaded simulation results from: $filepath")
    println("  - Timestamp: $(metadata["timestamp"])")
    println("  - $(metadata["n_simulations"]) simulations")
    println("  - $(metadata["n_stages"]) stages")
    if metadata["enable_extreme_events"]
        println("  - Extreme events enabled at year $(metadata["extreme_events_year"])")
    end

    return simulations, params, data, metadata
end

"""
    list_saved_results(directory::String="output")

List all saved simulation result files in a directory.

# Arguments
- `directory::String`: Directory to search (default: "output")

# Returns
- Vector of file information tuples: (filename, filepath, timestamp, file_size_MB)

# Example
```julia
results = list_saved_results("output")
for (name, path, time, size) in results
    println("\$name - \$time - \$size MB")
end
```
"""
function list_saved_results(directory::String="output")
    if !isdir(directory)
        println("Directory not found: $directory")
        return []
    end

    # Find all JLD2 files
    files = filter(f -> endswith(f, ".jld2"), readdir(directory))

    if isempty(files)
        println("No saved simulation results found in $directory")
        return []
    end

    results = []
    for file in files
        filepath = joinpath(directory, file)

        # Try to extract timestamp from filename
        timestamp = nothing
        if occursin(r"\d{4}_\d{2}_\d{2}_\d{4}", file)
            # Extract timestamp from filename pattern
            m = match(r"(\d{4})_(\d{2})_(\d{2})_(\d{2})(\d{2})", file)
            if m !== nothing
                year, month, day, hour, minute = parse.(Int, m.captures)
                timestamp = DateTime(year, month, day, hour, minute)
            end
        end

        # Get file size
        size_mb = round(filesize(filepath) / 1024^2, digits=2)

        push!(results, (file, filepath, timestamp, size_mb))
    end

    # Sort by timestamp (newest first)
    sort!(results, by=x -> x[3] !== nothing ? x[3] : DateTime(0), rev=true)

    println("Found $(length(results)) saved simulation results in $directory:")
    for (i, (name, _, time, size)) in enumerate(results)
        time_str = time !== nothing ? Dates.format(time, "yyyy-mm-dd HH:MM") : "Unknown"
        println("  $i. $name ($time_str, $size MB)")
    end

    return results
end

"""
    extract_extreme_events_info(simulations, params::ModelParameters)

Extract information about which extreme events occurred in each simulation.

# Arguments
- `simulations`: Simulation results
- `params::ModelParameters`: Model parameters

# Returns
- Dictionary mapping simulation index to extreme event occurrences

# Example
```julia
extreme_info = extract_extreme_events_info(simulations, params)
for (sim_idx, events) in extreme_info
    println("Simulation \$sim_idx: \$(events)")
end
```
"""
function extract_extreme_events_info(simulations, params::ModelParameters)
    if !params.enable_extreme_events
        println("Extreme events were not enabled in this simulation")
        return Dict()
    end

    extreme_stage = params.apply_to_year * 2  # Operational stage where extreme events occur
    extreme_info = Dict{Int, Any}()

    for (sim_idx, sim) in enumerate(simulations)
        # Check if noise_term exists at the extreme event stage
        if haskey(sim[extreme_stage], :noise_term)
            noise = sim[extreme_stage][:noise_term]
            extreme_info[sim_idx] = noise
        end
    end

    # Calculate statistics
    if !isempty(extreme_info)
        println("\nExtreme Event Statistics:")

        # Count occurrences of each scenario
        scenario_counts = Dict()
        for (_, noise) in extreme_info
            key = (noise.demand_mult, noise.elec_price_mult, noise.dc_avail)
            scenario_counts[key] = get(scenario_counts, key, 0) + 1
        end

        # Print statistics
        println("  Total simulations with extreme events: $(length(extreme_info))")
        println("\n  Scenario occurrences:")
        for (scenario, count) in sort(collect(scenario_counts), by=x->x[2], rev=true)
            freq = round(100 * count / length(simulations), digits=1)
            demand_mult, elec_mult, dc_avail = scenario
            println("    Demand×$(demand_mult), Elec×$(elec_mult), DC=$(dc_avail): $count times ($freq%)")
        end
    end

    return extreme_info
end

"""
    get_simulation_costs(simulations)

Extract total costs from all simulations.

# Arguments
- `simulations`: Simulation results

# Returns
- Vector of total costs for each simulation

# Example
```julia
costs = get_simulation_costs(simulations)
println("Mean cost: \$(mean(costs))")
println("Std dev: \$(std(costs))")
```
"""
function get_simulation_costs(simulations)
    n_stages = length(simulations[1])
    costs = Float64[]

    for sim in simulations
        total_cost = sum(sim[t][:stage_objective] for t in 1:n_stages)
        push!(costs, total_cost)
    end

    return costs
end

"""
    filter_simulations_by_extreme_event(simulations, params::ModelParameters,
                                       demand_mult::Float64, elec_price_mult::Float64,
                                       dc_avail::Float64)

Filter simulations that experienced a specific extreme event scenario.

# Arguments
- `simulations`: All simulation results
- `params::ModelParameters`: Model parameters
- `demand_mult`: Demand multiplier to filter for
- `elec_price_mult`: Electricity price multiplier to filter for
- `dc_avail`: Data center availability to filter for

# Returns
- Vector of simulation indices that match the criteria

# Example
```julia
# Get all simulations with cold snap (demand_mult=1.5)
cold_snap_sims = filter_simulations_by_extreme_event(simulations, params, 1.5, 1.2, 1.0)
println("Found \$(length(cold_snap_sims)) simulations with cold snap")
```
"""
function filter_simulations_by_extreme_event(simulations, params::ModelParameters,
                                            demand_mult::Float64, elec_price_mult::Float64,
                                            dc_avail::Float64)
    if !params.enable_extreme_events
        println("Extreme events were not enabled in this simulation")
        return Int[]
    end

    extreme_stage = params.apply_to_year * 2
    matching_sims = Int[]

    for (sim_idx, sim) in enumerate(simulations)
        if haskey(sim[extreme_stage], :noise_term)
            noise = sim[extreme_stage][:noise_term]
            if noise.demand_mult ≈ demand_mult &&
               noise.elec_price_mult ≈ elec_price_mult &&
               noise.dc_avail ≈ dc_avail
                push!(matching_sims, sim_idx)
            end
        end
    end

    return matching_sims
end