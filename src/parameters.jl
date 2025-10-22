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

    # Energy carriers
    c_emission_fac::Dict{Symbol, Float64}

    # Uncertainty configurations
    energy_price_map::Dict{Int, Float64}
    carbon_trajectories::Dict{Int, Vector{Float64}}
    carbon_probabilities::Dict{Int, Float64}
    demand_multipliers::Vector{Float64}
    demand_probabilities::Vector{Float64}
    energy_transitions::Matrix{Float64}
    initial_energy_dist::Vector{Float64}

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

    # Load EnergyPriceMap sheet
    energy_price_sheet = xf["EnergyPriceMap"]
    energy_price_df = DataFrame(XLSX.gettable(energy_price_sheet))
    energy_price_map = Dict(zip(Int.(energy_price_df.state), Float64.(energy_price_df.price_eur_per_mwh)))

    # Load CarbonTrajectories sheet
    carbon_traj_sheet = xf["CarbonTrajectories"]
    carbon_traj_df = DataFrame(XLSX.gettable(carbon_traj_sheet))
    carbon_trajectories = Dict{Int, Vector{Float64}}()
    for row in eachrow(carbon_traj_df)
        scenario = Int(row.scenario)
        trajectory = [Float64(row.year_1), Float64(row.year_2), Float64(row.year_3), Float64(row.year_4)]
        carbon_trajectories[scenario] = trajectory
    end

    # Load CarbonProbabilities sheet
    carbon_prob_sheet = xf["CarbonProbabilities"]
    carbon_prob_df = DataFrame(XLSX.gettable(carbon_prob_sheet))
    carbon_probabilities = Dict(zip(Int.(carbon_prob_df.scenario), Float64.(carbon_prob_df.probability)))

    # Load DemandUncertainty sheet
    demand_unc_sheet = xf["DemandUncertainty"]
    demand_unc_df = DataFrame(XLSX.gettable(demand_unc_sheet))
    demand_multipliers = Float64.(demand_unc_df.multiplier)
    demand_probabilities = Float64.(demand_unc_df.probability)

    # Load EnergyTransitions sheet
    energy_trans_sheet = xf["EnergyTransitions"]
    energy_trans_df = DataFrame(XLSX.gettable(energy_trans_sheet))
    energy_transitions = Matrix{Float64}(undef, 3, 3)
    for (i, row) in enumerate(eachrow(energy_trans_df))
        energy_transitions[i, :] = [Float64(row.to_high), Float64(row.to_medium), Float64(row.to_low)]
    end

    # Set initial energy distribution (could also be added to Excel if needed)
    initial_energy_dist = [0.3, 0.4, 0.3]

    # Calculate investment stages
    investment_stages = [0; collect(1:2:(2*T-1))]

    return ModelParameters(
        T, T_years, discount_rate, base_annual_demand, salvage_fraction,
        c_penalty,
        technologies, c_initial_capacity, c_max_additional_capacity,
        c_investment_cost, c_opex_fixed, c_opex_var, c_efficiency_th,
        c_efficiency_el, c_energy_carrier, c_lifetime_new, c_lifetime_initial,
        storage_params, c_emission_fac,
        energy_price_map, carbon_trajectories, carbon_probabilities,
        demand_multipliers, demand_probabilities, energy_transitions, initial_energy_dist,
        investment_stages
    )
end
