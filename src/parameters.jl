"""
Parameter loading and management for the District Heating Capacity Expansion model
"""

using XLSX, DataFrames

"""
Structure to hold all model parameters
"""
struct ModelParameters
    # Model configuration
    T::Int
    T_years::Int
    discount_rate::Float64
    base_annual_demand::Float64
    salvage_fraction::Float64
    c_penalty::Float64
    num_price_scenarios::Int
    mean_price::Float64
    price_volatility::Float64

    # Technologies
    technologies::Vector{Symbol}
    c_initial_capacity::Dict{Symbol, Float64}
    c_max_additional_capacity::Dict{Symbol, Float64}
    c_investment_cost::Dict{Symbol, Float64}
    c_opex_fixed::Dict{Symbol, Float64}
    c_opex_var::Dict{Symbol, Float64}
    c_efficiency_th::Dict{Symbol, Float64}
    c_efficiency_el::Dict{Symbol, Float64}
    c_energy_carrier::Dict{Symbol, Symbol}
    c_lifetime_new::Dict{Symbol, Int}
    c_lifetime_initial::Dict{Symbol, Int}

    # Storage
    storage_params::Dict{Symbol, Float64}

    # Energy carriers and carbon
    c_emission_fac::Dict{Symbol, Float64}
    c_carbon_price::Dict{Int, Float64}

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

    # Load ModelConfig sheet
    config_sheet = xf["ModelConfig"]
    config_df = DataFrame(XLSX.gettable(config_sheet))

    # Helper function to get parameter value
    get_param(name) = config_df[config_df.parameter .== name, :value][1]

    T = Int(get_param("T"))
    T_years = Int(get_param("T_years"))
    discount_rate = Float64(get_param("discount_rate"))
    base_annual_demand = Float64(get_param("base_annual_demand"))
    salvage_fraction = Float64(get_param("salvage_fraction"))
    c_penalty = Float64(get_param("c_penalty"))
    num_price_scenarios = Int(get_param("num_price_scenarios"))
    mean_price = Float64(get_param("mean_price"))
    price_volatility = Float64(get_param("price_volatility"))

    # Load Technologies sheet
    tech_sheet = xf["Technologies"]
    tech_df = DataFrame(XLSX.gettable(tech_sheet))

    technologies = Symbol.(tech_df.technology)
    c_initial_capacity = Dict(zip(technologies, Float64.(tech_df.initial_capacity)))
    c_max_additional_capacity = Dict(zip(technologies, Float64.(tech_df.max_additional_capacity)))
    c_investment_cost = Dict(zip(technologies, Float64.(tech_df.investment_cost)))
    c_opex_fixed = Dict(zip(technologies, Float64.(tech_df.fixed_om)))
    c_opex_var = Dict(zip(technologies, Float64.(tech_df.variable_om)))
    c_efficiency_th = Dict(zip(technologies, Float64.(tech_df.efficiency_th)))
    c_efficiency_el = Dict(zip(technologies, Float64.(tech_df.efficiency_el)))
    c_energy_carrier = Dict(zip(technologies, Symbol.(tech_df.energy_carrier)))
    c_lifetime_new = Dict(zip(technologies, Int.(tech_df.lifetime_new)))
    c_lifetime_initial = Dict(zip(technologies, Int.(tech_df.lifetime_initial)))

    # Load Storage sheet
    stor_sheet = xf["Storage"]
    stor_df = DataFrame(XLSX.gettable(stor_sheet))
    storage_params = Dict{Symbol, Float64}()
    for row in eachrow(stor_df)
        storage_params[Symbol(row.parameter)] = Float64(row.value)
    end

    # Load EnergyCarriers sheet
    carrier_sheet = xf["EnergyCarriers"]
    carrier_df = DataFrame(XLSX.gettable(carrier_sheet))
    c_emission_fac = Dict(zip(Symbol.(carrier_df.carrier), Float64.(carrier_df.emission_factor)))

    # Load CarbonPrice sheet
    carbon_sheet = xf["CarbonPrice"]
    carbon_df = DataFrame(XLSX.gettable(carbon_sheet))
    c_carbon_price = Dict(zip(Int.(carbon_df.year), Float64.(carbon_df.carbon_price)))

    # Load DemandMultipliers sheet
    demand_sheet = xf["DemandMultipliers"]
    demand_df = DataFrame(XLSX.gettable(demand_sheet))
    demand_multipliers = Float64.(demand_df.multiplier)

    # Calculate investment stages
    investment_stages = [0; collect(1:2:(2*T-1))]

    return ModelParameters(
        T, T_years, discount_rate, base_annual_demand, salvage_fraction,
        c_penalty, num_price_scenarios, mean_price, price_volatility,
        technologies, c_initial_capacity, c_max_additional_capacity,
        c_investment_cost, c_opex_fixed, c_opex_var, c_efficiency_th,
        c_efficiency_el, c_energy_carrier, c_lifetime_new, c_lifetime_initial,
        storage_params, c_emission_fac, c_carbon_price, demand_multipliers,
        investment_stages
    )
end
