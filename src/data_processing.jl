"""
Data loading and preprocessing for representative weeks and electricity prices
"""

using CSV, DataFrames, Dates, Statistics

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

    # Electricity prices
    purch_elec_price_2030_weeks::Matrix{Float64}
    purch_elec_price_2050_weeks::Matrix{Float64}
    sale_elec_price_2030_weeks::Matrix{Float64}
    sale_elec_price_2050_weeks::Matrix{Float64}
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

    # Scale demand to annual total
    scaling_factor = base_annual_demand /
        sum(sum(week) * w for (week, w) in zip(representative_weeks, week_weights_normalized))

    scaled_weeks = [week .* scaling_factor for week in representative_weeks]

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
    return sums ./ counts  # 168-length average price for each hour-of-week
end

"""
    process_electricity_prices(data_dir::String, n_weeks::Int)

Load and process electricity price data for representative weeks.

# Arguments
- `data_dir::String`: Directory containing the data files
- `n_weeks::Int`: Number of representative weeks

# Returns
- Processed electricity price matrices for purchase and sale
"""
function process_electricity_prices(data_dir::String, n_weeks::Int)
    println("Processing electricity prices...")

    # Load or create electricity price data
    filename_elec_2030 = joinpath(data_dir, "ElectricityPrice2030.csv")
    filename_elec_2050 = joinpath(data_dir, "ElectricityPrice2050.csv")

    # Load actual price data
    elec_price_2030_full = CSV.read(filename_elec_2030, DataFrame, delim=",", decimal='.')[:, "price"]
    elec_price_2050_full = CSV.read(filename_elec_2050, DataFrame, delim=",", decimal='.')[:, "price"]

    p2030_hw = hourly_profile_from_full_year(elec_price_2030_full, DateTime(2030, 1, 1))
    p2050_hw = hourly_profile_from_full_year(elec_price_2050_full, DateTime(2050, 1, 1))

    # Build (n_weeks × 168) matrices consistent with rep-week indexing
    sale_elec_price_2030_weeks = repeat(permutedims(p2030_hw), n_weeks, 1)
    sale_elec_price_2050_weeks = repeat(permutedims(p2050_hw), n_weeks, 1)

    # For now, assume purchase prices as a fraction:
    purch_elec_price_2030_weeks = sale_elec_price_2030_weeks .* 1.2
    purch_elec_price_2050_weeks = sale_elec_price_2050_weeks .* 1.2

    return purch_elec_price_2030_weeks, purch_elec_price_2050_weeks,
           sale_elec_price_2030_weeks, sale_elec_price_2050_weeks
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

    # Process electricity prices
    purch_elec_price_2030_weeks, purch_elec_price_2050_weeks,
    sale_elec_price_2030_weeks, sale_elec_price_2050_weeks =
        process_electricity_prices(data_dir, n_weeks)

    return ProcessedData(
        n_weeks, hours_per_week, representative_weeks, week_weights_normalized, scaled_weeks,
        purch_elec_price_2030_weeks, purch_elec_price_2050_weeks,
        sale_elec_price_2030_weeks, sale_elec_price_2050_weeks
    )
end
