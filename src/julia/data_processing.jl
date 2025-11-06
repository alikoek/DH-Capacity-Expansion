"""
Data loading and preprocessing for representative weeks and electricity prices
"""

using CSV, DataFrames, Dates, Statistics


"""
Structure to hold processed demand and price data
"""
struct ProcessedData
    # Representative weeks
    weeks::DataFrame
    week_weights::DataFrame
    errors::DataFrame
    weights::DataFrame

    rep_years::Vector{Int64}
    week_indexes::Vector{Int64}
    hour_indexes::Vector{Int64}
    price_scen::Vector{String}
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
function load_representative_weeks(data_dir::String)
    println("Loading representative weeks data...")
    hours_per_week = 168  # 7 * 24 hours
    project_dir = pwd()
    data_dir = joinpath(project_dir, "data","preprocessed")

    # Load TSAM data
    filename_tsam = joinpath(data_dir, "typical_weeks_all.csv")
    tsam_data1 = CSV.read(filename_tsam, DataFrame)


    filename_tsam = joinpath(data_dir, "typical_metrics_all.csv")
    tsam_data2 = CSV.read(filename_tsam, DataFrame)


    filename_tsam = joinpath(data_dir, "typical_weights_all.csv")
    tsam_data3 = CSV.read(filename_tsam, DataFrame)

    return select!(tsam_data1, Not("Column1")), select!(tsam_data2, Not("Column1")), select!(tsam_data3, Not("Column1"))
    
end
# weeks, errors, weights = load_representative_weeks(data_dir)


function load_all_data(data_dir::String)
    # Load representative weeks
    weeks, errors, weights = load_representative_weeks(data_dir)

    rep_years = unique(weeks[!,"year"])
    price_scen = unique(weeks[!,"scenario_price"])
    week_indexes = unique(weeks[!,"typical_week"])
    hour_indexes = unique(weeks[!,"hour"])

    return ProcessedData(
        weeks,
        weights,
        errors,
        weights,
        sort(rep_years),
        week_indexes,
        hour_indexes,
        price_scen
    )
end

# a = load_all_data(data_dir)
