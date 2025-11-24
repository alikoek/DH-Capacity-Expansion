"""
Data loading and preprocessing for representative periods and electricity prices
"""

using CSV, DataFrames, Dates, Statistics


"""
Structure to hold processed demand and price data
"""
struct ProcessedData
    # Representative periods
    periods::DataFrame
    period_weights::DataFrame
    errors::DataFrame
    weights::DataFrame

    rep_years::Vector{Int64}
    period_indexes::Vector{Int64}
    hour_indexes::Vector{Int64}
    price_scen::Vector{String}
end

"""
    load_representative_periods(data_dir::String, base_annual_demand::Float64)

Load and process representative periods from TSAM output.

# Arguments
- `data_dir::String`: Directory containing the data files
- `base_annual_demand::Float64`: Base annual demand in MWh

# Returns
- Processed representative periods data
"""
function load_representative_periods(data_dir::String)
    println("Loading representative periods data...")
    project_dir = pwd()
    data_dir = joinpath(project_dir, "data","preprocessed")

    # Load TSAM data
    filename_tsam = joinpath(data_dir, "typical_periods_all.csv")
    tsam_data1 = CSV.read(filename_tsam, DataFrame)


    filename_tsam = joinpath(data_dir, "typical_metrics_all.csv")
    tsam_data2 = CSV.read(filename_tsam, DataFrame)


    filename_tsam = joinpath(data_dir, "typical_weights_all.csv")
    tsam_data3 = CSV.read(filename_tsam, DataFrame)

    return select!(tsam_data1, Not("Column1")), select!(tsam_data2, Not("Column1")), select!(tsam_data3, Not("Column1"))
    
end
# periods, errors, weights = load_representative_periods(data_dir)


function load_all_data(data_dir::String)
    # Load representative periods
    periods, errors, weights = load_representative_periods(data_dir)

    rep_years = unique(periods[!,"year"])
    price_scen = unique(periods[!,"scenario_price"])
    period_indexes = unique(periods[!,"period"])
    hour_indexes = unique(periods[!,"hour"])

    return ProcessedData(
        periods,
        weights,
        errors,
        weights,
        sort(rep_years),
        period_indexes,
        hour_indexes,
        price_scen
    )
end

# a = load_all_data(data_dir)
