"""
Data loading and preprocessing for representative weeks and electricity prices.

Loads from typical_weeks_{n}.csv which contains scenario-specific demand and prices.
"""

using CSV, DataFrames, Statistics

"""
Structure to hold processed demand and price data.

Both demand and electricity prices are scenario-specific (low, medium, high).
"""
struct ProcessedData
    # Representative weeks configuration
    n_weeks::Int
    hours_per_week::Int
    week_weights_normalized::Vector{Float64}

    # Scenario-specific demand: [model_year][scenario][week_idx] -> Vector{Float64} (168 hours)
    # scenario ∈ {:low, :medium, :high}
    scaled_weeks::Dict{Int, Dict{Symbol, Vector{Vector{Float64}}}}

    # Scenario-specific electricity prices: [model_year][scenario] -> Matrix{Float64} (n_weeks × 168)
    # scenario ∈ {:low, :medium, :high}
    purch_elec_prices::Dict{Int, Dict{Symbol, Matrix{Float64}}}
    sale_elec_prices::Dict{Int, Dict{Symbol, Matrix{Float64}}}
end

"""
    load_all_data(params::ModelParameters, data_dir::String)

Load and process all required data from typical_weeks_{n}.csv.

The CSV contains scenario-specific demand and electricity prices for each year/week/hour.

# Arguments
- `params::ModelParameters`: Model parameters (T, T_years, n_typical_weeks, base_annual_demand, elec_taxes_levies)
- `data_dir::String`: Directory containing typical_weeks_{n}.csv

# Returns
- `ProcessedData`: Structure containing scenario-specific demand and prices
"""
function load_all_data(params::ModelParameters, data_dir::String)
    hours_per_week = 168

    # Load CSV file based on n_typical_weeks
    filename = joinpath(data_dir, "typical_weeks_$(params.n_typical_weeks).csv")
    if !isfile(filename)
        error("Data file not found: $filename. Please ensure typical_weeks_$(params.n_typical_weeks).csv exists.")
    end
    df = CSV.read(filename, DataFrame)

    println("  Loading data from: typical_weeks_$(params.n_typical_weeks).csv")

    # Build year mapping (consistent with parameters.jl)
    # Year 1 → 2023 (base year with data)
    # Year 2 → 2030
    # Year 3+ → 2030 + (m-2)*T_years
    # T_years=5:  2023, 2030, 2035, 2040
    # T_years=10: 2023, 2030, 2040, 2050
    model_to_csv_year = Dict{Int,Int}(1 => 2023, 2 => 2030)
    for m in 3:params.T
        model_to_csv_year[m] = 2030 + (m - 2) * params.T_years
    end
    csv_to_model_year = Dict(v => k for (k, v) in model_to_csv_year)

    println("  Year mapping: ", join(["$m → $(model_to_csv_year[m])" for m in 1:params.T], ", "))

    # Map CSV scenario names to model symbols
    scenario_map = Dict("low" => :low, "mid" => :medium, "high" => :high)

    # Get available periods (weeks) and validate
    available_periods = sort(unique(df.period))
    n_weeks = length(available_periods)
    println("  Typical weeks: $n_weeks (periods: $(available_periods))")

    # Extract week weights (should be consistent across scenarios)
    # Use first year and first scenario to get weights
    first_year = first(unique(df.year))
    first_scenario = first(unique(df.scenario_price))
    week_weights = Float64[]
    for period in available_periods
        subset = filter(row -> row.year == first_year && row.scenario_price == first_scenario && row.period == period, df)
        weight = first(subset.weight)
        push!(week_weights, weight)
    end

    # Normalize weights to sum to 52
    week_weights_normalized = week_weights .* (52 / sum(week_weights))
    week_weights_normalized = round.(week_weights_normalized, digits=4)
    println("  Week weights: $(week_weights_normalized) (sum = $(round(sum(week_weights_normalized), digits=1)))")

    # Initialize data structures
    scaled_weeks = Dict{Int, Dict{Symbol, Vector{Vector{Float64}}}}()
    purch_elec_prices = Dict{Int, Dict{Symbol, Matrix{Float64}}}()
    sale_elec_prices = Dict{Int, Dict{Symbol, Matrix{Float64}}}()

    # Process each model year
    for model_year in 1:params.T
        csv_year = model_to_csv_year[model_year]

        # Check if this year exists in the CSV
        if !(csv_year in unique(df.year))
            available_years = sort(unique(df.year))
            error("Year $csv_year (model year $model_year) not found in CSV. Available years: $available_years")
        end

        scaled_weeks[model_year] = Dict{Symbol, Vector{Vector{Float64}}}()
        purch_elec_prices[model_year] = Dict{Symbol, Matrix{Float64}}()
        sale_elec_prices[model_year] = Dict{Symbol, Matrix{Float64}}()

        # Process each scenario
        for (csv_scenario, model_scenario) in scenario_map
            # Filter data for this year/scenario
            year_scen_df = filter(row -> row.year == csv_year && row.scenario_price == csv_scenario, df)

            # Extract demand profiles for each week
            demand_profiles = Vector{Vector{Float64}}()
            price_matrix = zeros(n_weeks, hours_per_week)

            for (week_idx, period) in enumerate(available_periods)
                week_df = filter(row -> row.period == period, year_scen_df)
                # Sort by hour to ensure correct ordering
                sort!(week_df, :hour)

                @assert nrow(week_df) == hours_per_week "Expected $hours_per_week hours for period $period, got $(nrow(week_df))"

                # Extract demand (Load Profile column)
                demand = Float64.(week_df[:, "Load Profile"])
                push!(demand_profiles, demand)

                # Extract prices and convert SEK → TSEK
                prices = Float64.(week_df.price) ./ 1000.0
                price_matrix[week_idx, :] = prices
            end

            # Scale demand to match base_annual_demand
            # Annual demand = sum of (weekly demand × weight × hours_represented)
            # Since weights are normalized to 52 weeks: annual = sum(weekly_sum × weight)
            raw_annual_demand = sum(
                sum(demand_profiles[w]) * week_weights_normalized[w]
                for w in 1:n_weeks
            )
            scaling_factor = params.base_annual_demand / raw_annual_demand

            # Apply scaling to demand profiles
            scaled_demand = [round.(profile .* scaling_factor, digits=2) for profile in demand_profiles]
            scaled_weeks[model_year][model_scenario] = scaled_demand

            # Electricity prices: sale = spot price, purchase = spot + taxes
            sale_elec_prices[model_year][model_scenario] = round.(price_matrix, digits=4)
            purch_elec_prices[model_year][model_scenario] = round.(price_matrix .+ params.elec_taxes_levies, digits=4)
        end

        # Print summary for this year
        mean_demand_low = mean(mean.(scaled_weeks[model_year][:low]))
        mean_demand_high = mean(mean.(scaled_weeks[model_year][:high]))
        mean_price_low = mean(sale_elec_prices[model_year][:low])
        mean_price_high = mean(sale_elec_prices[model_year][:high])

        println("  Year $model_year ($(csv_year)): demand low=$(round(mean_demand_low, digits=1)) high=$(round(mean_demand_high, digits=1)) MW, " *
                "price low=$(round(mean_price_low, digits=3)) high=$(round(mean_price_high, digits=3)) TSEK/MWh")
    end

    return ProcessedData(
        n_weeks, hours_per_week, week_weights_normalized,
        scaled_weeks, purch_elec_prices, sale_elec_prices
    )
end
