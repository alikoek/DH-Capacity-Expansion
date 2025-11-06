"""
Parameter loading and management for the District Heating Capacity Expansion model
"""

using XLSX, DataFrames
using AxisKeys

struct ModelParameters
    technologies::Vector{String}
    config_dict::Dict
    tech_dict::Dict
    stor_dict::Dict
    carrier_dict::Dict
    carbon_df::DataFrame
    demand_uncertainty_df::DataFrame
    price_df::DataFrame
    elec_CO2_df::DataFrame

    dem_uncertainty_df::DataFrame
    policy_proba_df::DataFrame
    price_proba_df::DataFrame
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
    carrier_sheet = xf["EmissionFactors"]
    carrier_df = DataFrame(XLSX.gettable(carrier_sheet))
    # setindex!(carrier_df, carrier_df.carrier)
    carrier_dict = Dict(
        Symbol(row.carrier) => Dict(
            name => row[name] for name in names(carrier_df) if name != :parameter
        )
        for row in eachrow(carrier_df)
    )

    # Load CarbonPrice sheet
    carbon_sheet = xf["CarbonTrajectories"]
    carbon_df = DataFrame(XLSX.gettable(carbon_sheet))
    
    # Load DemandMultipliers sheet
    demand_sheet = xf["DemandUncertainty"]
    demand_uncertainty_df = DataFrame(XLSX.gettable(demand_sheet))


    # Load EnergyPrices
    sheet = xf["EnergyPriceMap"]
    price_df = DataFrame(XLSX.gettable(sheet))
    

    # Load ElectricityCO2
    sheet = xf["EmissionFactorElectricity"]
    elec_CO2_df = DataFrame(XLSX.gettable(sheet))
    

    sheet = xf["DemandUncertainty"]
    dem_uncertainty_df = DataFrame(XLSX.gettable(sheet))

    sheet = xf["CarbonProbabilities"]
    policy_proba_df = DataFrame(XLSX.gettable(sheet))

    sheet = xf["PriceTransitions"]
    price_proba_df = DataFrame(XLSX.gettable(sheet))

    return ModelParameters(
        technologies,
        config_dict,
        tech_dict,
        stor_dict,
        carrier_dict,
        carbon_df,
        demand_uncertainty_df,
        price_df,
        elec_CO2_df,
        dem_uncertainty_df,
        policy_proba_df,
        price_proba_df
    )
end

# Define paths
# project_dir = joinpath(dirname(@__DIR__), "..")
# data_dir = joinpath(project_dir, "data")
# output_dir = joinpath(project_dir, "output")
# excel_file = joinpath(data_dir, "model_parameters.xlsx")
# res = load_parameters(excel_file)

# show(res)
