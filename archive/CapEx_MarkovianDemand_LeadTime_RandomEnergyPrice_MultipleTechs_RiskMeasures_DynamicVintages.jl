using SDDP, Gurobi, LinearAlgebra, Distributions, CSV, DataFrames, Plots, Statistics, StatsPlots, Random

##############################################################################
# Basic Setup
##############################################################################

# Problem Data
T = 4  # Total number of model years --> 2020, 2030, 2040, 2050

# Investment stages (odd-numbered stages plus stage 0 for initial capacities)
investment_stages = [0; collect(1:2:(2*T-1))]  # [0, 1, 3, 5, 7] for T=4
T_years = 10 # number of years represented by the mmodel years

# Technological parameters
# Technology Names
technologies = [:CHP, :Boiler, :HeatPump, :Geothermal]

##############################################################################
# Heat Storage Parameters
##############################################################################

# Storage technology parameters
# Maximum storage capacity (MWh_th)
c_max_storage_capacity = 5000  # Large thermal storage tank

# Storage efficiency (charging/discharging)
c_storage_efficiency_charge = 0.95    # 95% efficiency when charging
c_storage_efficiency_discharge = 0.95  # 95% efficiency when discharging

# Storage heat loss rate per hour (fraction)
c_storage_heat_loss_rate = 0.001  # 0.1% heat loss per hour

# Storage investment cost (€/MWh)
c_storage_investment_cost = 50000  # €50k per MWh of storage capacity

# Storage fixed O&M cost (€/MWh per year)
c_storage_opex_fixed = 1000  # €1k per MWh per year

# Storage lifetime (years)
c_storage_lifetime = 3  # Same as other technologies for consistency

# Initial storage capacity (MWh_th)
c_initial_storage_capacity = 0  # No initial storage

# Lifetime in "years" (or pairs of stages)
# e.g., lifetime[:CHP] = 2 means "2 years" of operational usage
c_lifetime_new = Dict(
    :CHP => 3,
    :Boiler => 3,     # e.g. 3-year lifetime => no retirement in 3-year horizon
    :HeatPump => 2,   # only 1-year lifetime => retires quickly
    :Geothermal => 3
)
# Remaining lifetime of the existing capacities
c_lifetime_initial = Dict(
    :CHP => 2,
    :Boiler => 1,
    :HeatPump => 2,
    :Geothermal => 0
)

# Operational Costs (€/MWh)
c_opex_var = Dict(
    :Geothermal => 0.5,
    :HeatPump => 1.1,
    :CHP => 3.3,
    :Boiler => 0.75,
)


# Investment Costs (€/MW_th)
c_investment_cost = Dict(
    :CHP => 1100000,
    :Boiler => 100000,
    :HeatPump => 591052,
    :Geothermal => 1000000
)

# Fixed O&M Costs (€/MW_th per year)
c_opex_fixed = Dict(
    :CHP => 15000,
    :Boiler => 2000,
    :HeatPump => 3000,
    :Geothermal => 2000
)

c_energy_carrier = Dict(
    :CHP => :nat_gas,
    :Boiler => :nat_gas,
    :HeatPump => :elec,
    :Geothermal => :geothermal
)

# Maximum Additional Capacity (MW_th)
c_max_additional_capacity = Dict(
    :CHP => 500,
    :Boiler => 500,
    :HeatPump => 250,
    :Geothermal => 100
)

# Initial Capacities (MW_th)
c_initial_capacity = Dict(
    :CHP => 500,
    :Boiler => 350,
    :HeatPump => 0,
    :Geothermal => 0
)

# Efficiencies (as a fraction)
# Thermal efficiency
c_efficiency_th = Dict(
    :CHP => 0.44,
    :Boiler => 0.9,
    :HeatPump => 3.0,
    :Geothermal => 1.0
)

# Electrical efficiency
c_efficiency_el = Dict(
    :CHP => 0.43,
    :Boiler => 0.0,
    :HeatPump => 0.0,
    :Geothermal => 0.0
)

salvage_fraction = 1

# Emission factors (tCO2/MWh_th)
c_emission_fac = Dict(
    :nat_gas => 0.2,
    :elec => 0,
    :geothermal => 0
)

# Carbon Price (€/t) for years
c_carbon_price = Dict(
    1 => 50,
    2 => 150,
    3 => 250,
    4 => 350
)

# Function to get discount factor for stage t.
#   Each stage is 5 years. If t=1 => 2025, t=2 => 2030, etc.
#   We'll treat "year index" = t (1..T). 
discount_rate = 0.05
function discount_factor(t::Int, T_years::Int)
    # Each stage = 5 years
    exponent = T_years * (ceil(t / 2) - 1)
    return 1.0 / (1.0 + discount_rate)^exponent
end
##############################################################################
# Typical hours
##############################################################################
# read excel file for load profile
filename = joinpath(@__DIR__, "LoadProfile.csv")
load_profile = CSV.read(filename, DataFrames.DataFrame, delim=",", decimal='.')
load_profile = collect(load_profile[:, "load"])
load_profile_normalized = load_profile ./ sum(load_profile)

profile = load_profile_normalized
n_typical_hours = 100
quants = quantile(profile, LinRange(0, 1, n_typical_hours))
diff = (reshape(profile, (size(load_profile)..., 1)) .- quants') .^ 2
mindist = argmin(diff, dims=2)
second_component = getindex.(mindist, 2)

typical_hours = []
for (index, quant) in enumerate(quants)
    typical_hour = Dict(
        :value => quant,
        :qty => count(x -> x == index, second_component)
    )

    push!(typical_hours, typical_hour)
end
# print(typical_hours)

plot(load_profile_normalized, label="Original", title="Approximation versus Original")
plot!(quants[second_component[:]], label="approximation")

plot(sort(load_profile_normalized), label="Original", title="Approximation versus Original - Sorted")
plot!(sort(quants[second_component[:]]), label="approximation")

# Base annual demand
base_annual_demand = 2e6 # MWh

# Compute the absolute demand profile for the first year.
# For each typical hour, the demand is the base annual demand scaled by the normalized value.
c_base_demand_profile = [base_annual_demand * typical_hours[i][:value] for i in 1:n_typical_hours]
c_base_demand_profile = round.(c_base_demand_profile)
maximum(c_base_demand_profile)
#sum(c_base_demand_profile[i] * typical_hours[i][:qty] for i in 1:n_typical_hours)

maximum(c_base_demand_profile)
# Penalty cost for unmet demand (€/MWh)
c_penalty = 100000

# Define the stochastic demand multipliers: +10%, no change, -10%
demand_multipliers = [1.1, 1.0, 0.9]

############## Process electricity prices into typical hours ##################
filename_elec_2030 = joinpath(@__DIR__, "ElectricityPrice2030.csv")
elec_price_2030 = CSV.read(filename_elec_2030, DataFrames.DataFrame, delim=",", decimal='.')
elec_price_2030 = collect(elec_price_2030[:, "price"])
sale_elec_price_2030 = round.(elec_price_2030, digits=2) * 1.5
purch_elec_price_2030 = elec_price_2030 .+ 50

filename_elec_2050 = joinpath(@__DIR__, "ElectricityPrice2050.csv")
elec_price_2050 = CSV.read(filename_elec_2050, DataFrames.DataFrame, delim=",", decimal='.')
elec_price_2050 = collect(elec_price_2050[:, "price"])
sale_elec_price_2050 = round.(elec_price_2050, digits=2) * 1.5
purch_elec_price_2050 = elec_price_2050 .+ 50

typical_hour_indices = second_component
n_original_hours = length(load_profile)

# Initialize typical electricity prices for 2030 and 2050
c_purch_elec_price_2030_typical = zeros(n_typical_hours)
c_purch_elec_price_2050_typical = zeros(n_typical_hours)
c_sale_elec_price_2030_typical = zeros(n_typical_hours)
c_sale_elec_price_2050_typical = zeros(n_typical_hours)

unique(second_component)
sort(unique(second_component))

for i in 1:n_typical_hours
    hours_in_cluster = findall(x -> x == i, typical_hour_indices)
    c_purch_elec_price_2030_typical[i] = mean(purch_elec_price_2030[hours_in_cluster])
    c_purch_elec_price_2050_typical[i] = mean(purch_elec_price_2050[hours_in_cluster])
    c_sale_elec_price_2030_typical[i] = mean(sale_elec_price_2030[hours_in_cluster])
    c_sale_elec_price_2050_typical[i] = mean(sale_elec_price_2050[hours_in_cluster])
end

# plot(purch_elec_price_2030[hours_in_cluster])
# mean(purch_elec_price_2030[hours_in_cluster])
# median(purch_elec_price_2030[hours_in_cluster])

##############################################################################
# MarkovianPolicyGraph: Demand transitions + Root distribution
##############################################################################

# Create a 3x3 identity matrix
I = Diagonal(ones(3))

# Transition Matrices (explicitly converted to Matrix{Float64})
transition_matrices = Array{Float64,2}[
    [1.0]',  # Initial state: deterministic (start in the single initial state)
    [1.0]',
    [0.3 0.5 0.2],  # Transition probabilities between the 3 demand states
    I,  # put here an identity matrix
    [0.3 0.5 0.2
        0.3 0.5 0.2
        0.3 0.5 0.2],
    I,
    [0.3 0.5 0.2
        0.3 0.5 0.2
        0.3 0.5 0.2],
    I,
]

# the code below prints the markovian graph
SDDP.MarkovianGraph(transition_matrices)

##############################################################################
# Stagewise-Independent Energy Price
##############################################################################

# Log Normal distribution for the energy price
mean_price = 37.0  # Average energy price in currency units per unit of energy
price_volatility = 10.0  # Standard deviation
μ_normal = log(mean_price^2 / sqrt(price_volatility^2 + mean_price^2))
σ_normal = sqrt(log(1 + (price_volatility / mean_price)^2))
price_distribution = LogNormal(μ_normal, σ_normal)
num_price_scenarios = 4
price_quantiles = range(0.05, 0.95; length=num_price_scenarios)
price_values = quantile.(price_distribution, price_quantiles)
price_probabilities = pdf(price_distribution, price_values)
price_probabilities_normalized = price_probabilities / sum(price_probabilities)

##############################################################################
# Function to check if capacity built at investment stage s_invest 
# is still alive at current stage s_current.
##############################################################################
function is_alive(s_invest::Int, s_current::Int, lifetime::Dict, tech::Symbol)
    # The "year" index for s_invest is ceil(s_invest/2).
    # The "year" index for s_current is  ceil(s_current/2).
    year_invest = ceil(s_invest / 2)
    year_current = ceil(s_current / 2)
    # If lifetime[tech] = L, that means capacity is alive for L "years" 
    # after it's built (starting from the year built + 1 --> ensures lead time of 1 model year (5 year representation)).
    return (1 <= (year_current - year_invest)) && ((year_current - year_invest) <= lifetime[tech])
end

##############################################################################
# Create SDDP model
##############################################################################
# Create the Markovian policy graph
model = SDDP.MarkovianPolicyGraph(
    transition_matrices=transition_matrices,
    sense=:Min,
    lower_bound=0.0,  # Set the lower bound
    optimizer=Gurobi.Optimizer
) do sp, node
    # Unpack stage and demand state (t is stage, demand_state is 1, 2, or 3)
    t, demand_state = node

    # Variables
    @variables(sp, begin
        # Investment Decision Variables for each technology
        0 <= u_expansion_tech[tech in technologies] <= c_max_additional_capacity[tech]
        # Storage investment decision variable
        0 <= u_expansion_storage <= c_max_storage_capacity
        # Production Variables for each technology
        u_production_tech[tech in technologies, 1:n_typical_hours] >= 0
        # Storage charging and discharging variables
        u_storage_charge[1:n_typical_hours] >= 0
        u_storage_discharge[1:n_typical_hours] >= 0
        # Storage energy level state variable (between typical hours)
        0 <= x_storage_level[1:n_typical_hours] <= c_max_storage_capacity, SDDP.State, (initial_value = 0.0)
        # Unmet demand per hour
        u_unmet[1:n_typical_hours] >= 0

        # State variable: cumulative demand multiplier.
        # It is 1.0 in the first year, so the first-year demand profile is simply c_base_demand_profile.
        # In subsequent years, it is multiplied by the demand multiplier of the respective Markovian state.
        0 <= x_demand_mult, SDDP.State, (initial_value = 1.0)
    end)

    # Create vintage capacity variables dynamically
    for stage in investment_stages
        if stage == 0
            # Initial capacities
            @variable(sp, 0 <= cap_vintage[stage, tech in technologies] <= 3000, SDDP.State, (initial_value = c_initial_capacity[tech]))
            # Initial storage capacity
            @variable(sp, 0 <= storage_cap_vintage[stage] <= c_max_storage_capacity, SDDP.State, (initial_value = c_initial_storage_capacity))
        else
            # New investment capacities  
            @variable(sp, 0 <= cap_vintage[stage, tech in technologies] <= c_max_additional_capacity[tech], SDDP.State, (initial_value = 0))
            # New storage investment capacity
            @variable(sp, 0 <= storage_cap_vintage[stage] <= c_max_storage_capacity, SDDP.State, (initial_value = 0))
        end
    end

    # Initial capacities stay the same regardless of the stage
    @constraint(sp, [tech in technologies],
        cap_vintage[0, tech].out == cap_vintage[0, tech].in
    )
    # Initial storage capacity constraint
    @constraint(sp, storage_cap_vintage[0].out == storage_cap_vintage[0].in)

    ################### Investment stage (odd-numbered stages) ###################
    if t % 2 == 1
        # No change in the demand multiplier in investment stages.
        @constraint(sp, x_demand_mult.out == x_demand_mult.in)
        
        # Dynamic investment logic: update vintages based on current stage
        for stage in investment_stages[2:end]  # Skip stage 0 (initial capacities)
            if stage == t
                # This is the current investment stage - add new expansions
                @constraint(sp, [tech in technologies],
                    cap_vintage[stage, tech].out == cap_vintage[stage, tech].in + u_expansion_tech[tech]
                )
                # Storage expansion for current investment stage
                @constraint(sp, 
                    storage_cap_vintage[stage].out == storage_cap_vintage[stage].in + u_expansion_storage
                )
            else
                # Other investment stages - no change
                @constraint(sp, [tech in technologies],
                    cap_vintage[stage, tech].out == cap_vintage[stage, tech].in
                )
                # Other storage investment stages - no change
                @constraint(sp, 
                    storage_cap_vintage[stage].out == storage_cap_vintage[stage].in
                )
            end
        end

        # Stage objective: expansion cost + fixed O&M costs
        # discount_factor
        local df = discount_factor(t, T_years)

        # Expansion cost
        expr_invest_cost = sum(
            c_investment_cost[tech] * u_expansion_tech[tech]
            for tech in technologies
        ) + c_storage_investment_cost * u_expansion_storage
        
        # Fixed O&M costs for the capacities that are still alive
        expr_opex_fix = 0.0
        for tech in technologies
            for stage in investment_stages
                lifetime_dict = (stage == 0) ? c_lifetime_initial : c_lifetime_new
                if is_alive(stage, t, lifetime_dict, tech)
                    expr_opex_fix += c_opex_fixed[tech] * cap_vintage[stage, tech].in
                end
            end
        end
        
        # Fixed O&M costs for storage
        for stage in investment_stages
            if is_alive(stage, t, Dict(:Storage => c_storage_lifetime), :Storage)
                expr_opex_fix += c_storage_opex_fixed * storage_cap_vintage[stage].in
            end
        end
        # Multiply the O&M by 5 because each stage is 5 years
        expr_opex_fix *= T_years
        @stageobjective(sp, df * (expr_invest_cost + expr_opex_fix))

        ################### Operational stage (even-numbered stages) ###################
    else
        model_year = Int(ceil(t / 2))
        # ensure that annual demand is not changed in the first year
        if t == 2
            new_demand_mult = 1.0
        else
            # Annual demand adjustment based on the stochastic demand state
            new_demand_mult = demand_multipliers[demand_state] * x_demand_mult.in
        end

        ### Apply constraints for production and unmet demand based on the stochastic annual demand
        # Total production from all technologies plus storage discharge minus storage charge plus unmet demand meets the actual demand
        # The actual hourly demand is computed by scaling the first year's profile
        @constraint(sp, [hour in 1:n_typical_hours],
            sum(u_production_tech[tech, hour] for tech in technologies) + u_storage_discharge[hour] - u_storage_charge[hour] + u_unmet[hour] ==
            c_base_demand_profile[hour] * new_demand_mult
        )

        # Pre-compute the capacity alive for each technology at this stage
        # = sum of expansions built in each prior investment stage that is still alive
        capacity_alive = Dict{Symbol,JuMP.GenericAffExpr{Float64,VariableRef}}()
        for tech in technologies
            capacity_expr = 0.0
            for stage in investment_stages
                lifetime_dict = (stage == 0) ? c_lifetime_initial : c_lifetime_new
                if is_alive(stage, t, lifetime_dict, tech)
                    capacity_expr += cap_vintage[stage, tech].in
                end
            end
            capacity_alive[tech] = capacity_expr
        end

        # Capacity constraints for each technology and hour:
        # For each technology, 
        #   u_production_tech[tech,hour] <= (sum of alive expansions)
        @constraint(sp, [tech in technologies, hour in 1:n_typical_hours],
            u_production_tech[tech, hour] <= capacity_alive[tech]
        )

        # Storage capacity and energy balance constraints
        # Compute total alive storage capacity
        storage_capacity_alive = 0.0
        for stage in investment_stages
            if is_alive(stage, t, Dict(:Storage => c_storage_lifetime), :Storage)
                storage_capacity_alive += storage_cap_vintage[stage].in
            end
        end

        # Storage charging and discharging power constraints
        # Assume charging/discharging rate is limited to 20% of storage capacity per hour
        max_charge_discharge_rate = 0.2
        @constraint(sp, [hour in 1:n_typical_hours],
            u_storage_charge[hour] <= max_charge_discharge_rate * storage_capacity_alive
        )
        @constraint(sp, [hour in 1:n_typical_hours],
            u_storage_discharge[hour] <= max_charge_discharge_rate * storage_capacity_alive
        )

        # Storage energy level constraints
        @constraint(sp, [hour in 1:n_typical_hours],
            x_storage_level[hour].out <= storage_capacity_alive
        )

        # Storage energy balance constraints (simplified for typical hours)
        # For the first typical hour
        @constraint(sp,
            x_storage_level[1].out == x_storage_level[1].in * (1 - c_storage_heat_loss_rate * typical_hours[1][:qty]) +
            c_storage_efficiency_charge * u_storage_charge[1] * typical_hours[1][:qty] -
            u_storage_discharge[1] * typical_hours[1][:qty] / c_storage_efficiency_discharge
        )

        # For subsequent typical hours
        @constraint(sp, [hour in 2:n_typical_hours],
            x_storage_level[hour].out == x_storage_level[hour-1].out * (1 - c_storage_heat_loss_rate * typical_hours[hour][:qty]) +
            c_storage_efficiency_charge * u_storage_charge[hour] * typical_hours[hour][:qty] -
            u_storage_discharge[hour] * typical_hours[hour][:qty] / c_storage_efficiency_discharge
        )

        # State updates for investment states: no new expansions at operation stages
        for stage in investment_stages[2:end]  # Skip stage 0 (handled separately)
            @constraint(sp, [tech in technologies],
                cap_vintage[stage, tech].out == cap_vintage[stage, tech].in
            )
            # Storage vintage states - no new expansions at operation stages
            @constraint(sp, 
                storage_cap_vintage[stage].out == storage_cap_vintage[stage].in
            )
        end

        # Update the state variable for the cumulative demand multiplier.
        @constraint(sp, x_demand_mult.out == new_demand_mult)

        # If this is the *final* stage, include salvage for any capacity that extends beyond planning horizon
        local salvage = 0.0
        if t == T * 2
            for tech in technologies
                for stage in investment_stages
                    stage_year = (stage == 0) ? 0 : Int(ceil(stage / 2))
                    lifetime_dict = (stage == 0) ? c_lifetime_initial : c_lifetime_new
                    
                    if (model_year - stage_year) < lifetime_dict[tech]
                        remaning_life = lifetime_dict[tech] - (model_year - stage_year)
                        salvage += c_investment_cost[tech] * cap_vintage[stage, tech].in * (remaning_life / lifetime_dict[tech])
                    end
                end
            end
            
            # Storage salvage value calculation
            for stage in investment_stages
                stage_year = (stage == 0) ? 0 : Int(ceil(stage / 2))
                if (model_year - stage_year) < c_storage_lifetime
                    remaning_life = c_storage_lifetime - (model_year - stage_year)
                    salvage += c_storage_investment_cost * storage_cap_vintage[stage].in * (remaning_life / c_storage_lifetime)
                end
            end
        end

        ### Stage objective (operational costs) ###
        SDDP.parameterize(sp, price_values, price_probabilities_normalized) do ω
            # discount factor for this stage
            local df = discount_factor(t, T_years)
            # Determine the relevant electricity purchase and sale price for the current model year
            local purch_elec_price = model_year <= 2 ? c_purch_elec_price_2030_typical : c_purch_elec_price_2050_typical
            local sale_elec_price = model_year <= 2 ? c_sale_elec_price_2030_typical : c_sale_elec_price_2050_typical

            local expr_annual_cost = sum(
                (
                    sum(
                        c_opex_var[tech] * u_production_tech[tech, hour] #O&M
                        + (
                            (c_energy_carrier[tech] == :nat_gas ? ω :
                             (c_energy_carrier[tech] == :geothermal ? 0.0 : purch_elec_price[hour]))
                            + (c_carbon_price[model_year] * c_emission_fac[c_energy_carrier[tech]]) # carbon price
                            - (c_efficiency_el[tech] * sale_elec_price[hour]) # electricity revenue
                        )
                        * (u_production_tech[tech, hour] / c_efficiency_th[tech] # primary energy conversion
                        )
                        for tech in technologies
                    )
                    + u_unmet[hour] * c_penalty
                ) * typical_hours[hour][:qty]
                for hour in 1:n_typical_hours)
            # multiply by the 5-year block and discount factor
            @stageobjective(sp, df * (T_years * expr_annual_cost - salvage * salvage_fraction))
        end
    end
end

# Train the model
SDDP.train(model; risk_measure=SDDP.CVaR(0.95), iteration_limit=1000)

# Retrieve results
println("Optimal Cost: ", SDDP.calculate_bound(model))

# set seed for reproducibility
Random.seed!(1234)

# Simulation - dynamically create vintage symbols for simulation
vintage_symbols = [Symbol("cap_vintage"), Symbol("storage_cap_vintage")]
simulation_symbols = vcat(vintage_symbols, [:x_demand_mult, :u_production_tech, :u_expansion_tech, :u_expansion_storage, :u_storage_charge, :u_storage_discharge, :x_storage_level, :u_unmet])
simulations = SDDP.simulate(model, 400, simulation_symbols)

# Print simulation results
for t in 1:(T*2)
    sp = simulations[3][t]
    if t % 2 == 1  # Investment stages (odd stages)
        println("Year $(div(t + 1, 2)) - Investment Stage")
        for tech in technologies
            println("  Technology: $tech")
            println("    Expansion = ", value(sp[:u_expansion_tech][tech]))
            for stage in investment_stages
                println("    Capacity_s$(stage)_in = ", value(sp[:cap_vintage][stage, tech].in))
                println("    Capacity_s$(stage)_out = ", value(sp[:cap_vintage][stage, tech].out))
            end
        end
        # Print storage expansion information
        println("  Storage:")
        println("    Expansion = ", value(sp[:u_expansion_storage]))
        for stage in investment_stages
            println("    Storage_Capacity_s$(stage)_in = ", value(sp[:storage_cap_vintage][stage].in))
            println("    Storage_Capacity_s$(stage)_out = ", value(sp[:storage_cap_vintage][stage].out))
        end
    else  # Operational stages
        println("Year $(div(t, 2)) - Operational Stage")
        println("Noise term: ", value(sp[:noise_term]))
        println("   Demand Multiplier (in/out) = ", value(sp[:x_demand_mult].in), " / ", value(sp[:x_demand_mult].out))
        println("   Annual Demand = ", sum(c_base_demand_profile[i] * value(sp[:x_demand_mult].out) * typical_hours[i][:qty] for i in 1:n_typical_hours))
        for tech in technologies
            # sum expansions from all investment stages that are alive
            local capacity_alive = 0.0
            for stage in investment_stages
                lifetime_dict = (stage == 0) ? c_lifetime_initial : c_lifetime_new
                if is_alive(stage, t, lifetime_dict, tech)
                    capacity_alive += value(sp[:cap_vintage][stage, tech].in)
                end
            end
            println("    $tech alive Capacity = ", capacity_alive)
        end
        
        # Print storage information
        local storage_capacity_alive = 0.0
        for stage in investment_stages
            if is_alive(stage, t, Dict(:Storage => c_storage_lifetime), :Storage)
                storage_capacity_alive += value(sp[:storage_cap_vintage][stage].in)
            end
        end
        println("    Storage alive Capacity = ", storage_capacity_alive)
        println("    Storage charge total = ", sum(value(sp[:u_storage_charge][hour] * typical_hours[hour][:qty]) for hour in 1:n_typical_hours))
        println("    Storage discharge total = ", sum(value(sp[:u_storage_discharge][hour] * typical_hours[hour][:qty]) for hour in 1:n_typical_hours))
        println("    Storage level at end = ", value(sp[:x_storage_level][n_typical_hours].out))
        total_production = sum(sum(value(sp[:u_production_tech][tech, hour] * typical_hours[hour][:qty]) for hour in 1:n_typical_hours) for tech in technologies)
        total_unmet = sum(value(sp[:u_unmet][hour] * typical_hours[hour][:qty]) for hour in 1:n_typical_hours)
        println("  Total Production = ", total_production)
        println("  Total Unmet Demand = ", total_unmet)
        local salvage = 0.0
        if t == T * 2
            for tech in technologies
                for stage in investment_stages
                    stage_year = (stage == 0) ? 0 : Int(ceil(stage / 2))
                    lifetime_dict = (stage == 0) ? c_lifetime_initial : c_lifetime_new
                    
                    if (ceil(t / 2) - stage_year) < lifetime_dict[tech]
                        println(" The investment year is $stage_year")
                        remaning_life = lifetime_dict[tech] - (ceil(t / 2) - stage_year)
                        println("   Remaining Life of $tech = ", remaning_life)
                        println("   Remaining capacity of $tech = ", value(sp[:cap_vintage][stage, tech].in))
                        stage_salvage = c_investment_cost[tech] * value(sp[:cap_vintage][stage, tech].in) * (remaning_life / lifetime_dict[tech])
                        salvage += stage_salvage
                        println("   Salvage Value of $tech = ", stage_salvage)
                        println("********************************")
                    end
                end
            end
            
            # Storage salvage value calculation
            for stage in investment_stages
                stage_year = (stage == 0) ? 0 : Int(ceil(stage / 2))
                if (ceil(t / 2) - stage_year) < c_storage_lifetime
                    println(" Storage investment year is $stage_year")
                    remaning_life = c_storage_lifetime - (ceil(t / 2) - stage_year)
                    println("   Remaining Life of Storage = ", remaning_life)
                    println("   Remaining capacity of Storage = ", value(sp[:storage_cap_vintage][stage].in))
                    stage_salvage = c_storage_investment_cost * value(sp[:storage_cap_vintage][stage].in) * (remaning_life / c_storage_lifetime)
                    salvage += stage_salvage
                    println("   Salvage Value of Storage = ", stage_salvage)
                    println("********************************")
                end
            end
        end
        println("  Total Salvage Value = ", salvage * salvage_fraction)
    end
end

##############################################################################
# Visulaizations
##############################################################################
################################# Spaghetti Plot #############################
##############################################################################
plt = SDDP.SpaghettiPlot(simulations)

for tech in technologies
    # SDDP.add_spaghetti(plt; title="Capacity_in_$tech") do data
    #     return data[:x_capacity_tech][tech].in
    # end
    # SDDP.add_spaghetti(plt; title="Capacity_out_$tech") do data
    #     return data[:x_capacity_tech][tech].out
    # end
    SDDP.add_spaghetti(plt; title="Expansion_$tech") do data
        return data[:u_expansion_tech][tech]
    end
    SDDP.add_spaghetti(plt; title="Production_$tech") do data
        return sum(data[:u_production_tech][tech, :])
    end
end

# Add storage spaghetti plots
SDDP.add_spaghetti(plt; title="Storage_Expansion") do data
    return data[:u_expansion_storage]
end
SDDP.add_spaghetti(plt; title="Storage_Charge") do data
    return sum(data[:u_storage_charge])
end
SDDP.add_spaghetti(plt; title="Storage_Discharge") do data
    return sum(data[:u_storage_discharge])
end
SDDP.add_spaghetti(plt; title="Storage_Level_End") do data
    return data[:x_storage_level][n_typical_hours].out
end


SDDP.add_spaghetti(plt; title="Demand Multiplier_in") do data
    return data[:x_demand_mult].in
end
SDDP.add_spaghetti(plt; title="Demand Multiplier_out") do data
    return data[:x_demand_mult].out
end
SDDP.add_spaghetti(plt; title="Total Production") do data
    return sum(sum((data[:u_production_tech][tech, hour]) for hour in 1:n_typical_hours) for tech in technologies)
end
SDDP.add_spaghetti(plt; title="Unmet Demand") do data
    return sum(data[:u_unmet])
end

SDDP.plot(plt, "spaghetti_plot.html")

############################## Publication Plot ################################
Plots.plot(
    SDDP.publication_plot(simulations; title="Expansion") do data
        return data[:u_expansion_tech][:HeatPump]
    end,
    SDDP.publication_plot(simulations; title="Thermal generation") do data
        return sum((data[:u_production_tech][:HeatPump, hour]) for hour in 1:n_typical_hours)
    end;
    xlabel="Stage",
    # ylims = (0, 200),
    layout=(1, 2),
)

Plots.plot(
    SDDP.publication_plot(simulations; title="Objective") do data
        return data[:stage_objective]
    end,
    xlabel="Stage",
    # ylims = (0, 200),
    # layout = (1, 2),
)
####################################################################################
##############################  Manual band plot ###################################
####################################################################################
function plot_bands(data, title, x_values, xticks, xlabel, ylabel, legend_pos)
    # Compute statistics for each column (apply quantile to each column)
    q5 = [quantile(data[:, i], 0.05) for i in 1:size(data, 2)]
    q25 = [quantile(data[:, i], 0.25) for i in 1:size(data, 2)]
    q50 = [quantile(data[:, i], 0.50) for i in 1:size(data, 2)]  # Median (50th percentile)
    q75 = [quantile(data[:, i], 0.75) for i in 1:size(data, 2)]
    q95 = [quantile(data[:, i], 0.95) for i in 1:size(data, 2)]
    # Create plot
    p = plot(x_values, q5, fillrange=q95, fillalpha=0.15, c=2, label="5-95 percentile", legend=:topleft, lw=0, xticks=xticks)
    plot!(x_values, q25, fillrange=q75, fillalpha=0.35, c=2, label="25-75 percentile", legend=:topleft, lw=0, xticks=xticks)
    plot!(x_values, q95, fillalpha=0.15, c=2, label="", legend=legend_pos, lw=0, xticks=xticks)
    plot!(x_values, q75, fillalpha=0.35, c=2, label="", legend=legend_pos, lw=0, xticks=xticks)
    plot!(x_values, q50, label="Median", lw=2, color=2)  # Median line
    # Labels and title
    xlabel!(xlabel)
    ylabel!(ylabel)
    # title!(title)
    return (p)
    # display()
end

for tech in technologies
    total_cap_alive = zeros(length(simulations), T * 2)
    for sim in (1:length(simulations))
        for t in (1:T*2)
            cap_alive = 0
            for stage in investment_stages
                lifetime_dict = (stage == 0) ? c_lifetime_initial : c_lifetime_new
                if is_alive(stage, t, lifetime_dict, tech)
                    cap_alive += value(simulations[sim][t][:cap_vintage][stage, tech].in)
                end
            end
            total_cap_alive[sim, t] = cap_alive
        end
    end
    if tech == :boiler
        legend_pos = :topright
    else
        legend_pos = :topleft
    end

    fig = plot_bands(total_cap_alive, "$tech Alive Capacities", 1:4, "Stages", "Capacity [MW_th]", legend_pos)
    display(fig)
    savefig("AliveCapacities_$tech.png")
end


for tech in technologies
    inv_stages = collect(1:2:T*2)
    u_invs = zeros(length(simulations), length(inv_stages))
    for sim in (1:length(simulations))
        counter = 1
        for t in inv_stages
            u_inv = value(simulations[sim][t][:u_expansion_tech][tech])
            u_invs[sim, counter] = u_inv
            counter += 1
        end
    end
    if tech == :geothermal
        legend_pos = :topleft
    else
        legend_pos = :topright
    end
    fig = plot_bands(u_invs, "$tech Investment", [1, 3, 5, 7], [1, 3, 5, 7], "Investment Stages", "Investment [MW_th]", legend_pos)
    display(fig)
    savefig("Investments_$tech.png")
end
####################################################################################
##############################  Load Duration Curve ################################
####################################################################################
n_sim = 1

plt = plot(xlabel="Hours", ylabel="Heat Generation [MWh_th]", legend=:topleft)
y0 = zeros(8760 * T, length(technologies))
labels = Vector{String}()

# sorted_keys = sort(collect(c_opex_var), by=x -> x[2], rev=true)  # Sort by value descending
enumerated_keys = [pair[1] for pair in collect(c_opex_var)]  # Extract sorted keys
new_vect = [:Geothermal, :HeatPump, :CHP, :Boiler]
println(enumerated_keys)
for (index, tech) in enumerate(new_vect)
    for t_prim in 1:(T)
        t = 2 * t_prim
        sp = simulations[n_sim][t]
        total_unmet = sum(value(sp[:u_unmet][hour] * typical_hours[hour][:qty]) for hour in 1:n_typical_hours)
        y0[1+8760*(t_prim-1):8760*t_prim, index] = sort(Array(values(sp[:u_production_tech][tech, :]))[second_component[:]])
        prod_tot = (sum(Array(values(sp[:u_production_tech][tech, :]))[second_component[:]]))
        # vline(1+8760*(t_prim-1))
        print(" $(tech) $(prod_tot)")
        println()
    end

    push!(labels, String(tech))
end
# print(labels)
# labels = string.(Any["CHP", "HeatPump", "Boiler"])
# println(hcat(labels))
areaplot!(y0, label=Matrix(reshape(labels, 1, length(labels))), fillalpha=[0.2 0.3 0.4], lw=0)
display(plt)
savefig("LoadDurationCurve_Simulation$n_sim.png")

subp = simulations[n_sim][2]
dict_keys = collect(keys(subp))
[println(dict_keys[i]) for i in 1:length(dict_keys)]
subp[:u_unmet]
c_base_demand_profile
minimum(c_base_demand_profile)


tech = :HeatPump
sort(Array(values(subp[:u_production_tech][tech, :]))[second_component[:]])
dispatch = values(subp[:u_production_tech][tech, :])
sum(dispatch)
##############################################################################
##############################  Violin Plots #################################
##############################################################################
# Create a violin plot for the total production of each technology at the operational stages
function plot_violins(data, title, xlabels, y_legend)

    # Prepare data for violin plot
    group_labels = repeat(xlabels, inner=size(data, 1))  # Repeat each x-axis label for each data point
    flattened_values = vec(data)  # Flatten the matrix into a single vector

    # Overlay means points for each group
    group_means = [mean(data[:, i]) for i in 1:size(data, 2)]
    scatter(group_labels, flattened_values, color=:black, markershape=:o, label="samples", markersize=2)
    # Create violin plot
    violin!(group_labels, flattened_values,
        xlabel="Periods", title=title, label="pdf",
        legend=true, alpha=0.5, c=:green)
    plot!(gridlinewidth=2)
    scatter!(xlabels, group_means, color=:red, markershape=:o, label="Mean")

    ylabel!(y_legend)

end

function plot_combined_violins(data1, data2, title, xlabels, y_legend, leg_pos)
    # Prepare data for violin plot
    group_labels = repeat(xlabels, inner=size(data1, 1))  # Repeat each x-axis label for each data point
    flattened_values_1 = vec(data1)  # Flatten the matrix into a single vector
    flattened_values_2 = vec(data2)  # Flatten the matrix into a single vector

    # Overlay means points for each group
    group_means = [mean(data2[:, i]) for i in 1:size(data1, 2)]
    scatter(group_labels, flattened_values_2, color=:red, markershape=:o, label="Mean annual demand", markersize=2, lw=0)
    # Create violin plot
    violin!(group_labels, flattened_values_1,
        xlabel="Periods", title=title, label="PDF of annual production",
        legend=leg_pos, alpha=0.5, c=:green, side=:right)
    # dotplot!(group_labels, flattened_values_1, side=:right, color=:green, markershape=:o, label="samples", markersize = 2)
    # Create violin plot
    violin!(group_labels, flattened_values_2,
        xlabel="Periods", title=title, label="PDF of annnual demand",
        legend=leg_pos, alpha=0.5, c=:red, side=:left)
    plot!(gridlinewidth=2)
    # scatter!(xlabels, group_means, color=:red, markershape=:o, label="Mean")

    ylabel!(y_legend)
end

### Demand ###
xlabels = []
ope_var_cons = zeros(length(simulations), T)
for (ope_stage, stage) in enumerate(2:2:2*T)
    push!(xlabels, "$(2010 + ope_stage*T_years)")
    for sim in 1:length(simulations)
        ope_var_cons[sim, ope_stage] = values(simulations[sim][stage][:x_demand_mult].out) * base_annual_demand
    end
end


ope_var = zeros(length(simulations), T)
# Boiler
for (ope_stage, stage) in enumerate(2:2:2*T)
    push!(xlabels, "$(2010 + ope_stage*T_years)")
    for sim in 1:length(simulations)
        ope_var[sim, ope_stage] = (sum(Array(values(simulations[sim][stage][:u_production_tech][:Boiler, :]))[second_component[:]]))
    end
end
q1 = plot_combined_violins(ope_var, ope_var_cons, "Boiler", xlabels, "Production in MWh", :inside)
plot!(ylim=(0, upper_bound))

# CHP
for (ope_stage, stage) in enumerate(2:2:2*T)
    for sim in 1:length(simulations)
        ope_var[sim, ope_stage] = (sum(Array(values(simulations[sim][stage][:u_production_tech][:CHP, :]))[second_component[:]]))
    end
end
q2 = plot_combined_violins(ope_var, ope_var_cons, "CHP", xlabels, "Production in MWh", :inside)
plot!(ylim=(0, upper_bound))

# HeatPump
for (ope_stage, stage) in enumerate(2:2:2*T)
    for sim in 1:length(simulations)
        ope_var[sim, ope_stage] = (sum(Array(values(simulations[sim][stage][:u_production_tech][:HeatPump, :]))[second_component[:]]))
    end
end
q3 = plot_combined_violins(ope_var, ope_var_cons, "Heat Pump", xlabels, "Production in MWh", :inside)

# Geothermal
for (ope_stage, stage) in enumerate(2:2:2*T)
    for sim in 1:length(simulations)
        ope_var[sim, ope_stage] = (sum(Array(values(simulations[sim][stage][:u_production_tech][:Geothermal, :]))[second_component[:]]))
    end
end
q4 = plot_combined_violins(ope_var, ope_var_cons, "Geothermal", xlabels, "Production in MWh", :bottom)


plot!(ylim=(0, upper_bound))
plot!(size=(1500, 500))
l = @layout [a b c d]
q = plot(q1, q2, q3, q4, link=:y, layout=l)
savefig("ViolinPlots_ProductionVsDemand.png")