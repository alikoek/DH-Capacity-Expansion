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

    CO2_data::DataFrame
    policy_transitions::DataFrame
    price_transitions::DataFrame
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
    println(data_dir)

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

    data_dir = joinpath(project_dir, "data")
    filename_co2 = joinpath(data_dir, "policy_transitions.csv")
    policy_transitions = CSV.read(filename_co2, DataFrame)


    data_dir = joinpath(project_dir, "data")
    filename_co2 = joinpath(data_dir, "price_transitions.csv")
    price_transitions = CSV.read(filename_co2, DataFrame)

    data_dir = joinpath(project_dir, "data")
    filename_co2 = joinpath(data_dir, "co2_costs.csv")
    CO2_data = CSV.read(filename_co2, DataFrame)

    rep_years = unique(weeks[!,"year"])
    price_scen = unique(weeks[!,"scenario_price"])
    week_indexes = unique(weeks[!,"typical_week"])
    hour_indexes = unique(weeks[!,"hour"])
    show(sort(rep_years))
    show(hour_indexes)

    return ProcessedData(
        weeks,
        weights,
        errors,
        weights,
        sort(rep_years),
        week_indexes,
        hour_indexes,
        price_scen,
        CO2_data,
        policy_transitions,
        price_transitions
    )
end

# a = load_all_data(data_dir)ß