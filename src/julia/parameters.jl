"""
Parameter loading and management for the District Heating Capacity Expansion model
"""

using XLSX, DataFrames
using AxisKeys

struct ModelParameters
    technologies ::Vector{String}
    config_dict ::Dict
    tech_dict::Dict
    stor_dict::Dict    
    carrier_dict::Dict
    carbon_dict::Dict  #osef

    # Demand multipliers
    demand_multipliers::Vector{Float64}

    # Investment stages
    investment_stages::Vector{Int}
end

"""
    load_parameters(excel_path::String)

Load all model parameters from an Excel file.

# Arguments
- `excel_path::String`: Path to the Excel parameter file

# Returns
- `ModelParameters`: Structure containing all model parameters
"""
function load_parameters(excel_path::String)
    # Open Excel file
    xf = XLSX.readxlsx(excel_path)

    # # Load ModelConfig sheet

    # # Helper function to get parameter value
    # get_param(name) = config_df[config_df.parameter .== name, :value][1]
    config_sheet = xf["ModelConfig"]
    config_df = DataFrame(XLSX.gettable(config_sheet))
    config_dict = Dict(Symbol(row.parameter) => row.value for row in eachrow(config_df))
    println(config_dict)
    T = config_dict[:T]

    # Load Technologies sheet
    tech_sheet = xf["Technologies"]
    tech_df = DataFrame(XLSX.gettable(tech_sheet))
    tech_dict = Dict(
        Symbol(row.technology) => Dict(
            name => row[name] for name in names(tech_df) if name != :parameter
        )
        for row in eachrow(tech_df)
    )
    technologies = String.(tech_df.technology)

    # Load Storage sheet
    stor_sheet = xf["Storage"]
    stor_df = DataFrame(XLSX.gettable(stor_sheet))
    # setindex!(stor_df, stor_df.storage)
    stor_dict = Dict(
        Symbol(row.storage) => Dict(
            name => row[name] for name in names(stor_df) if name != :parameter
        )
        for row in eachrow(stor_df)
    )

    # Load EnergyCarriers sheet
    carrier_sheet = xf["EnergyCarriers"]
    carrier_df = DataFrame(XLSX.gettable(carrier_sheet))
    # setindex!(carrier_df, carrier_df.carrier)
    carrier_dict = Dict(
        Symbol(row.carrier) => Dict(
            name => row[name] for name in names(carrier_df) if name != :parameter
        )
        for row in eachrow(carrier_df)
    )

    # Load CarbonPrice sheet
    carbon_sheet = xf["CarbonPrice"]
    carbon_df = DataFrame(XLSX.gettable(carbon_sheet))
    # setindex!(carbon_df, carbon_df.year)
    carbon_dict = Dict(
        Symbol(row.year) => Dict(
            name => row[name] for name in names(carbon_df) if name != :parameter
        )
        for row in eachrow(carbon_df)
    )


    # Load DemandMultipliers sheet
    demand_sheet = xf["DemandMultipliers"]
    demand_df = DataFrame(XLSX.gettable(demand_sheet))
    demand_multipliers = Float64.(demand_df.multiplier)

    # Calculate investment stages
    investment_stages = [0; collect(1:2:(2*T-1))]

    return ModelParameters(
        technologies,
        config_dict,
        tech_dict,
        stor_dict,
        carrier_dict,
        carbon_dict,
        demand_multipliers,
        investment_stages
    )
end

# Define paths
project_dir = joinpath(dirname(@__DIR__), "..")
data_dir = joinpath(project_dir, "data")
print(data_dir)
output_dir = joinpath(project_dir, "output")
excel_file = joinpath(data_dir, "model_parameters.xlsx")
res = load_parameters(excel_file)
