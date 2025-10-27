"""
Data loading and preprocessing for representative weeks and electricity prices
"""

using CSV, DataFrames, Dates, Statistics, XLSX

"""
Structure to hold processed demand and price data
"""
struct ProcessedData
    # Representative weeks
    n_weeks::Int
    hours_per_week::Int
    representative_weeks::Vector{Vector{Float64}}
    week_weights_normalized::Vector{Float64}
    scaled_weeks::Vector{Vector{Float64}}

    # Electricity prices - 3 scenarios per model year (low, medium, high)
    # Stored as Dict[model_year][scenario] where scenario ∈ {:low, :medium, :high}
    purch_elec_prices::Dict{Int, Dict{Symbol, Matrix{Float64}}}
    sale_elec_prices::Dict{Int, Dict{Symbol, Matrix{Float64}}}
end

"""
    load_representative_weeks(data_dir::String, base_annual_demand::Float64)

Load and process representative weeks from TSAM output.

# Arguments
- `data_dir::String`: Directory containing the data files
- `base_annual_demand::Float64`: Base annual demand in MWh

# Returns
- Processed representative weeks data
"""
function load_representative_weeks(data_dir::String, base_annual_demand::Float64)
    println("Loading representative weeks data...")
    hours_per_week = 168  # 7 * 24 hours

    # Load TSAM data
    filename_tsam = joinpath(data_dir, "typical_weeks.csv")
    tsam_data = CSV.read(filename_tsam, DataFrame)

    # Group rows by typical-week id and extract 168-hour profiles
    week_ids = sort(unique(tsam_data[!, "Typical Week"]))
    @assert all(count(==(w), tsam_data[!, "Typical Week"]) == hours_per_week for w in week_ids) "Each week must have 168 rows."

    insertcols!(tsam_data, :hour_in_week => zeros(Int, nrow(tsam_data)))
    for w in week_ids
        idx = findall(tsam_data[!, "Typical Week"] .== w)
        @assert length(idx) == hours_per_week
        tsam_data[idx, :hour_in_week] .= 1:hours_per_week
    end

    representative_weeks = [collect(tsam_data[tsam_data[!, "Typical Week"] .== w, "load"]) for w in week_ids]
    n_weeks = length(representative_weeks)

    week_weights = [mean(skipmissing(tsam_data[tsam_data[!, "Typical Week"] .== w, "weight_abs"])) |> float for w in week_ids]
    # Normalize weights to 52 weeks
    week_weights_normalized = week_weights .* (52 / sum(week_weights))
    # Round weights to avoid false precision (4 decimal places is sufficient)
    week_weights_normalized = round.(week_weights_normalized, digits=4)

    # Scale demand to annual total
    scaling_factor = base_annual_demand /
        sum(sum(week) * w for (week, w) in zip(representative_weeks, week_weights_normalized))

    # Round demand profiles to 2 decimal places (precision of ~0.01 MW is sufficient)
    scaled_weeks = [round.(week .* scaling_factor, digits=2) for week in representative_weeks]

    println("Representative weeks loaded: $(n_weeks) * $(hours_per_week)h; weights sum = $(sum(week_weights_normalized))")
    println("Week weights: ", week_weights_normalized)

    return n_weeks, hours_per_week, representative_weeks, week_weights_normalized, scaled_weeks
end

"""
    hourly_profile_from_full_year(p_full::AbstractVector{<:Real}, start_dt::DateTime)

Convert full-year hourly prices to average hourly profile for a week.

# Arguments
- `p_full::AbstractVector{<:Real}`: Full year of hourly prices (8760 or 8784 hours)
- `start_dt::DateTime`: Start datetime for the price data

# Returns
- Vector of 168 average hourly prices for each hour of the week
"""
function hourly_profile_from_full_year(p_full::AbstractVector{<:Real}, start_dt::DateTime)
    H = 168
    sums = zeros(Float64, H)
    counts = zeros(Int, H)

    for (k, p) in pairs(p_full)
        t = start_dt + Hour(k - 1)
        # Julia: dayofweek(Mon=1,…,Sun=7), hour(t)=0…23  → index 1…168
        hW = (dayofweek(t) - 1) * 24 + hour(t) + 1
        sums[hW] += p
        counts[hW] += 1
    end
    @assert all(counts .> 0)
    # Convert SEK to TSEK and round to 4 decimal places
    return round.(sums ./ counts ./ 1000, digits=4)  # 168-length average price for each hour-of-week in TSEK/MWh
end

"""
    process_electricity_prices(data_dir::String, n_weeks::Int, model_year::Int, elec_taxes_levies::Float64)

Load and process electricity price data for a specific model year from Stockholm data.
Creates 3 price scenarios (low, medium, high) where:
- Low: Nuclear-heavy scenario (low, stable prices)
- High: Renewable-heavy scenario (high, volatile prices)
- Medium: Average of low and high

The Excel data represents spot market prices (used as sale prices).
Purchase prices are calculated by adding taxes and levies to the sale prices.

These are correlated with energy price states (low gas → low elec, etc.)

# Arguments
- `data_dir::String`: Directory containing the data files
- `n_weeks::Int`: Number of representative weeks
- `model_year::Int`: Model year (1, 2, 3, 4 corresponding to 2020, 2030, 2040, 2050)
- `elec_taxes_levies::Float64`: Taxes and levies to add to spot prices (EUR/MWh)

# Returns
- Tuple of 6 price matrices: (purch_low, sale_low, purch_med, sale_med, purch_high, sale_high)
"""
function process_electricity_prices(data_dir::String, n_weeks::Int, model_year::Int, elec_taxes_levies::Float64)
    println("Processing electricity prices for model year $model_year...")

    # Map model years to data years
    # Model year 1 (2020) → use 2023 data (earliest available)
    # Model year 2 (2030) → use 2030 data
    # Model year 3 (2040) → use 2040 data
    # Model year 4 (2050) → use 2050 data
    data_year_map = Dict(1 => "2023", 2 => "2030", 3 => "2040", 4 => "2050")
    data_year = data_year_map[model_year]

    # Load electricity prices from Excel
    filename_elec = joinpath(data_dir, "ElectricityPrices.xlsx")
    xf = XLSX.readxlsx(filename_elec)

    # Load low (nuclear) and high (renewable) price scenarios
    df_low = DataFrame(XLSX.gettable(xf["Electricity price - low"]))
    df_high = DataFrame(XLSX.gettable(xf["Electricity price - high"]))

    # Extract the specific year column (convert to Float64 vector)
    elec_price_low_full = Float64.(df_low[:, data_year])
    elec_price_high_full = Float64.(df_high[:, data_year])

    # Convert to hourly profiles (168 hours per week)
    # Use year from data for datetime (leap year handling)
    year_num = parse(Int, data_year)
    p_low_hw = hourly_profile_from_full_year(elec_price_low_full, DateTime(year_num, 1, 1))
    p_high_hw = hourly_profile_from_full_year(elec_price_high_full, DateTime(year_num, 1, 1))

    # Create MEDIUM scenario as average of low and high (rounded)
    p_medium_hw = round.((p_low_hw .+ p_high_hw) ./ 2, digits=4)

    # Build (n_weeks × 168) matrices for each scenario
    # Excel data represents spot market prices (sale prices)
    sale_elec_price_low = repeat(permutedims(p_low_hw), n_weeks, 1)
    sale_elec_price_medium = repeat(permutedims(p_medium_hw), n_weeks, 1)
    sale_elec_price_high = repeat(permutedims(p_high_hw), n_weeks, 1)

    # Purchase prices = sale prices + taxes and levies (rounded)
    # Note: elec_taxes_levies already converted to TSEK in parameters.jl
    purch_elec_price_low = round.(sale_elec_price_low .+ elec_taxes_levies, digits=4)
    purch_elec_price_medium = round.(sale_elec_price_medium .+ elec_taxes_levies, digits=4)
    purch_elec_price_high = round.(sale_elec_price_high .+ elec_taxes_levies, digits=4)

    println("  Using year $data_year from Stockholm data")
    println("  Created 3 electricity price scenarios:")
    println("    Low (nuclear):     mean = $(round(mean(p_low_hw), digits=4)) TSEK/MWh")
    println("    Medium (average):  mean = $(round(mean(p_medium_hw), digits=4)) TSEK/MWh")
    println("    High (renewable):  mean = $(round(mean(p_high_hw), digits=4)) TSEK/MWh")

    return purch_elec_price_low, sale_elec_price_low,
           purch_elec_price_medium, sale_elec_price_medium,
           purch_elec_price_high, sale_elec_price_high
end

"""
    load_all_data(params::ModelParameters, data_dir::String)

Load and process all required data for the model.

# Arguments
- `params::ModelParameters`: Model parameters structure
- `data_dir::String`: Directory containing data files

# Returns
- `ProcessedData`: Structure containing all processed data
"""
function load_all_data(params::ModelParameters, data_dir::String)
    # Load representative weeks
    n_weeks, hours_per_week, representative_weeks, week_weights_normalized, scaled_weeks =
        load_representative_weeks(data_dir, params.base_annual_demand)

    # Load electricity prices for all model years
    purch_elec_prices = Dict{Int, Dict{Symbol, Matrix{Float64}}}()
    sale_elec_prices = Dict{Int, Dict{Symbol, Matrix{Float64}}}()

    for model_year in 1:params.T
        purch_low, sale_low, purch_med, sale_med, purch_high, sale_high =
            process_electricity_prices(data_dir, n_weeks, model_year, params.elec_taxes_levies)

        purch_elec_prices[model_year] = Dict(:low => purch_low, :medium => purch_med, :high => purch_high)
        sale_elec_prices[model_year] = Dict(:low => sale_low, :medium => sale_med, :high => sale_high)
    end

    return ProcessedData(
        n_weeks, hours_per_week, representative_weeks, week_weights_normalized, scaled_weeks,
        purch_elec_prices, sale_elec_prices
    )
end
