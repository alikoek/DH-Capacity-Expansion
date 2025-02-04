using SDDP, Gurobi, LinearAlgebra, Distributions, CSV, DataFrames, Plots, Statistics

##############################################################################
# Basic Setup
##############################################################################

# Problem Data
T = 3  # Total number of model years --> 2025, 2030, 2035
T_years = 5 # number of years represented by the mmodel years

# Technological parameters
# Technology Names
technologies = [:CHP, :Boiler, :HeatPump]

# Lifetime in "years" (or pairs of stages)
# e.g., lifetime[:CHP] = 2 means "2 years" of operational usage
c_lifetime = Dict(
    :CHP => 2,
    :Boiler => 3,     # e.g. 3-year lifetime => no retirement in 3-year horizon
    :HeatPump => 2,   # only 1-year lifetime => retires quickly
)

# Operational Costs (€/MWh)
c_opex_var = Dict(
    :CHP => 30.0,
    :Boiler => 10.0,
    :HeatPump => 20.0
)

# Investment Costs (€/MW_th)
c_investment_cost = Dict(
    :CHP => 1000.0,
    :Boiler => 500.0,
    :HeatPump => 800.0
)

# Fixed O&M Costs (€/MW_th per year)
c_opex_fixed = Dict(
    :CHP => 50.0,
    :Boiler => 30.0,
    :HeatPump => 40.0
)

c_energy_carrier = Dict(
    :CHP => :nat_gas,
    :Boiler => :nat_gas,
    :HeatPump => :elec
)

# Energy carrier prices (€/MWh)
c_energy_carrier_price = Dict(
    :elec => 100
)

# Maximum Additional Capacity (MW_th)
c_max_additional_capacity = Dict(
    :CHP => 1000,
    :Boiler => 1000,
    :HeatPump => 1000
)

# Initial Capacities (MW_th)
c_initial_capacity = Dict(
    :CHP => 1800,
    :Boiler => 750,
    :HeatPump => 100,
)

# Efficiency (as a fraction)
c_efficiency = Dict(
    :CHP => 0.9,
    :Boiler => 0.85,
    :HeatPump => 3.0  # Coefficient of Performance (COP)
)

# Emission factors (tCO2/MWh_th)
c_emission_fac = Dict(
    :nat_gas => 0.2,
    :elec => 0,
)

# Carbon Price (€/t) for years
c_carbon_price = Dict(
    1 => 50,
    2 => 150,
    3 => 250
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
base_annual_demand = 6.5e6 # MWh

# Compute the absolute demand profile for the first year.
# For each typical hour, the demand is the base annual demand scaled by the normalized value.
c_base_demand_profile = [base_annual_demand * typical_hours[i][:value] for i in 1:n_typical_hours]
c_base_demand_profile = round.(c_base_demand_profile)
#sum(c_base_demand_profile[i] * typical_hours[i][:qty] for i in 1:n_typical_hours)

maximum(c_base_demand_profile)
# Penalty cost for unmet demand (€/MWh)
c_penalty = 1000

# Define the stochastic demand multipliers: +10%, no change, -10%
demand_multipliers = [1.1, 1.0, 0.9]

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
num_price_scenarios = 10
price_quantiles = range(0.05, 0.95; length=num_price_scenarios)
price_values = quantile.(price_distribution, price_quantiles)
price_probabilities = pdf(price_distribution, price_values)
price_probabilities_normalized = price_probabilities / sum(price_probabilities)

##############################################################################
# Function to check if capacity built at investment stage s_invest 
# is still alive at current stage s_current.
##############################################################################
function is_alive(s_invest::Int, s_current::Int, tech::Symbol)
    # The "year" index for s_invest is ceil(s_invest/2).
    # The "year" index for s_current is  ceil(s_current/2).
    year_invest = ceil(s_invest / 2)
    year_current = ceil(s_current / 2)
    # If lifetime[tech] = L, that means capacity is alive for L "years" 
    # after it's built (starting from the year built + 1 --> ensures lead time of 1 model year (5 year representation)).
    return (1 <= (year_current - year_invest)) && ((year_current - year_invest) <= c_lifetime[tech])
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
        # Production Variables for each technology
        u_production_tech[tech in technologies, 1:n_typical_hours] >= 0
        # Unmet demand per hour
        u_unmet[1:n_typical_hours] >= 0

        # State variable: cumulative demand multiplier.
        # It is 1.0 in the first year, so the first-year demand profile is simply c_base_demand_profile.
        # In subsequent years, it is multiplied by the demand multiplier of the respective Markovian state.
        0 <= x_demand_mult, SDDP.State, (initial_value = 1.0)

        # Define a dictionary of states that track expansions built at each 
        # investment stage for t=1,3,5 (since T=3 => 3 investment stages).
        # cap_invest_s0 represents the initial capacities
        0 <= cap_invest_s0[tech in technologies] <= 3000, SDDP.State, (initial_value = c_initial_capacity[tech])
        # 0 <= x_capacity_tech[tech in technologies] <= 1000, SDDP.State, (initial_value = c_initial_capacity[tech])
        0 <= cap_invest_s1[tech in technologies] <= c_max_additional_capacity[tech], SDDP.State, (initial_value = 0)
        0 <= cap_invest_s3[tech in technologies] <= c_max_additional_capacity[tech], SDDP.State, (initial_value = 0)
        0 <= cap_invest_s5[tech in technologies] <= c_max_additional_capacity[tech], SDDP.State, (initial_value = 0)
    end)

    ### inital capacities stays the same regardless of the stage
    @constraint(sp, [tech in technologies],
        cap_invest_s0[tech].out == cap_invest_s0[tech].in
    )

    ################### Investment stage (odd-numbered stages) ###################
    if t % 2 == 1
        # No change in the demand multiplier in investment stages.
        @constraint(sp, x_demand_mult.out == x_demand_mult.in)
        # Decide expansions in this stage => add to the relevant "cap_invest_sX"
        if t == 1
            @constraint(sp, [tech in technologies],
                cap_invest_s1[tech].out == cap_invest_s1[tech].in + u_expansion_tech[tech]
            )
            @constraint(sp, [tech in technologies],
                cap_invest_s3[tech].out == cap_invest_s3[tech].in
            )
            @constraint(sp, [tech in technologies],
                cap_invest_s5[tech].out == cap_invest_s5[tech].in
            )
        elseif t == 3
            @constraint(sp, [tech in technologies],
                cap_invest_s1[tech].out == cap_invest_s1[tech].in
            )
            @constraint(sp, [tech in technologies],
                cap_invest_s3[tech].out == cap_invest_s3[tech].in + u_expansion_tech[tech]
            )
            @constraint(sp, [tech in technologies],
                cap_invest_s5[tech].out == cap_invest_s5[tech].in
            )
        elseif t == 5
            @constraint(sp, [tech in technologies],
                cap_invest_s1[tech].out == cap_invest_s1[tech].in
            )
            @constraint(sp, [tech in technologies],
                cap_invest_s3[tech].out == cap_invest_s3[tech].in
            )
            @constraint(sp, [tech in technologies],
                cap_invest_s5[tech].out == cap_invest_s5[tech].in + u_expansion_tech[tech]
            )
        end

        # Stage objective: expansion cost + fixed O&M costs
        # discount_factor
        local df = discount_factor(t, T_years)

        # Expansion cost
        expr_invest_cost = sum(
            c_investment_cost[tech] * u_expansion_tech[tech]
            for tech in technologies
        )
        # Fixed O&M costs for the capacities that are still alive
        expr_opex_fix = 0.0
        for tech in technologies
            # sum capacity from stages 1,3,5 that is alive
            # in the "year" = ceil(t/2).
            if is_alive(0, t, tech)
                expr_opex_fix += c_opex_fixed[tech] * cap_invest_s0[tech].in
            end
            if is_alive(1, t, tech)
                expr_opex_fix += c_opex_fixed[tech] * cap_invest_s1[tech].in
            end
            if is_alive(3, t, tech)
                expr_opex_fix += c_opex_fixed[tech] * cap_invest_s3[tech].in
            end
            if is_alive(5, t, tech)
                expr_opex_fix += c_opex_fixed[tech] * cap_invest_s5[tech].in
            end
        end
        # Multiply the O&M by 5 because each stage is 5 years
        expr_opex_fix *= T_years
        @stageobjective(sp, df * (expr_invest_cost + expr_opex_fix))

        ################### Operational stage (even-numbered stages) ###################
    else
        # ensure that annual demand is not changed in the first year
        if t == 2
            new_demand_mult = 1.0
        else
            # Annual demand adjustment based on the stochastic demand state
            new_demand_mult = demand_multipliers[demand_state] * x_demand_mult.in
        end

        ### Apply constraints for production and unmet demand based on the stochastic annual demand
        # Total production from all technologies plus unmet demand meets the actual demand
        # The actual hourly demand is computed by scaling the first year's profile
        @constraint(sp, [hour in 1:n_typical_hours],
            sum(u_production_tech[tech, hour] for tech in technologies) + u_unmet[hour] ==
            c_base_demand_profile[hour] * new_demand_mult
        )

        # Pre-compute the capacity alive for each technology at this stage
        # = sum of expansions built in each prior investment stage that is still alive
        capacity_alive = Dict{Symbol,JuMP.GenericAffExpr{Float64,VariableRef}}()
        for tech in technologies
            capacity_expr = 0.0
            if is_alive(0, t, tech)
                capacity_expr += cap_invest_s0[tech].in
            end
            if is_alive(1, t, tech)
                capacity_expr += cap_invest_s1[tech].in
            end
            if is_alive(3, t, tech)
                capacity_expr += cap_invest_s3[tech].in
            end
            if is_alive(5, t, tech)
                capacity_expr += cap_invest_s5[tech].in
            end
            capacity_alive[tech] = capacity_expr
        end

        # Capacity constraints for each technology and hour:
        # For each technology, 
        #   u_production_tech[tech,hour] <= (sum of alive expansions)
        @constraint(sp, [tech in technologies, hour in 1:n_typical_hours],
            u_production_tech[tech, hour] <= capacity_alive[tech]
        )

        # State updates for investment states: no new expansions at operation stages
        # Update capacity with the investments in the pipeline  for each technology
        @constraint(sp, [tech in technologies],
            cap_invest_s1[tech].out == cap_invest_s1[tech].in
        )
        @constraint(sp, [tech in technologies],
            cap_invest_s3[tech].out == cap_invest_s3[tech].in
        )
        @constraint(sp, [tech in technologies],
            cap_invest_s5[tech].out == cap_invest_s5[tech].in
        )

        # Update the state variable for the cumulative demand multiplier.
        @constraint(sp, x_demand_mult.out == new_demand_mult)

        # If this is the *final* stage (t=6 for T=3), include salvage for 
        #  any capacity that extends beyond planning horizon.
            local salvage = 0.0
            if t == T * 2
                for tech in technologies
                    if (ceil(t/2) - 0) < c_lifetime[tech]
                        remaning_life = c_lifetime[tech] - (ceil(t/2) - 0)
                        salvage += c_investment_cost[tech] * cap_invest_s0[tech].in * (remaning_life / c_lifetime[tech]) 
                    end
                    if (ceil(t/2) - 1) < c_lifetime[tech]
                        remaning_life = c_lifetime[tech] - (ceil(t/2) - 1)
                        salvage += c_investment_cost[tech] * cap_invest_s1[tech].in * (remaning_life / c_lifetime[tech]) 
                    end
                    if (ceil(t/2) - 2) < c_lifetime[tech]
                        remaning_life = c_lifetime[tech] - (ceil(t/2) - 3)
                        salvage += c_investment_cost[tech] * cap_invest_s3[tech].in * (remaning_life / c_lifetime[tech]) 
                    end
                    if (ceil(t/2) - 3) < c_lifetime[tech]
                        remaning_life = c_lifetime[tech] - (ceil(t/2) - 5)
                        salvage += c_investment_cost[tech] * cap_invest_s5[tech].in * (remaning_life / c_lifetime[tech]) 
                    end
                end
            end 

        ### Stage objective (operational costs) ###
        SDDP.parameterize(sp, price_values, price_probabilities_normalized) do ω
            # discount factor for this stage
            local df = discount_factor(t, T_years)

            local expr_annual_cost = sum(
                (
                    sum(
                        c_opex_var[tech] * u_production_tech[tech, hour] #O&M
                        + (
                            (c_energy_carrier[tech] == :nat_gas ? ω : c_energy_carrier_price[c_energy_carrier[tech]])
                            + c_carbon_price[ceil(t / 2)] * c_emission_fac[c_energy_carrier[tech]]
                        )
                        * (u_production_tech[tech, hour] / c_efficiency[tech]
                        )
                        for tech in technologies
                    )
                    + u_unmet[hour] * c_penalty
                ) * typical_hours[hour][:qty]
                for hour in 1:n_typical_hours)
            # multiply by the 5-year block and discount factor
            @stageobjective(sp, df * (T_years * expr_annual_cost - salvage))
        end
    end
end

# Train the model
SDDP.train(model; iteration_limit=100)

# Retrieve results
println("Optimal Cost: ", SDDP.calculate_bound(model))

# Simulation
simulations = SDDP.simulate(model, 10, [:cap_invest_s0, :cap_invest_s1, :cap_invest_s3, :cap_invest_s5, :x_demand_mult, :u_production_tech, :u_expansion_tech, :u_unmet])

# Print simulation results
for t in 1:(T*2)
    sp = simulations[1][t]
    if t % 2 == 1  # Investment stages (odd stages)
        println("Year $(div(t + 1, 2)) - Investment Stage")
        for tech in technologies
            println("  Technology: $tech")
            println("    Expansion = ", value(sp[:u_expansion_tech][tech]))
            println("    Capacity_s0_in = ", value(sp[:cap_invest_s0][tech].in))
            println("    Capacity_s0_out = ", value(sp[:cap_invest_s0][tech].out))
            println("    Capacity_s1_in = ", value(sp[:cap_invest_s1][tech].in))
            println("    Capacity_s1_out = ", value(sp[:cap_invest_s1][tech].out))
            println("    Capacity_s3_in = ", value(sp[:cap_invest_s3][tech].in))
            println("    Capacity_s3_out = ", value(sp[:cap_invest_s3][tech].out))
            println("    Capacity_s5_in = ", value(sp[:cap_invest_s5][tech].in))
            println("    Capacity_s5_out = ", value(sp[:cap_invest_s5][tech].out))
        end
    else  # Operational stages
        println("Year $(div(t, 2)) - Operational Stage")
        println("   Demand Multiplier (in/out) = ", value(sp[:x_demand_mult].in), " / ", value(sp[:x_demand_mult].out))
        println("   Annual Demand = ", sum(c_base_demand_profile[i] * value(sp[:x_demand_mult].out) * typical_hours[i][:qty] for i in 1:n_typical_hours))
        for tech in technologies
            # sum expansions from s=1,3,5 that are alive
            local capacity_alive = 0.0
            if is_alive(0, t, tech)
                capacity_alive += value(sp[:cap_invest_s0][tech].in)
            end
            if is_alive(1, t, tech)
                capacity_alive += value(sp[:cap_invest_s1][tech].in)
            end
            if is_alive(3, t, tech)
                capacity_alive += value(sp[:cap_invest_s3][tech].in)
            end
            if is_alive(5, t, tech)
                capacity_alive += value(sp[:cap_invest_s5][tech].in)
            end
            println("    $tech alive Capacity = ", capacity_alive)
        end
        total_production = sum(sum(value(sp[:u_production_tech][tech, hour] * typical_hours[hour][:qty]) for hour in 1:n_typical_hours) for tech in technologies)
        total_unmet = sum(value(sp[:u_unmet][hour] * typical_hours[hour][:qty]) for hour in 1:n_typical_hours)
        println("  Total Production = ", total_production)
        println("  Total Unmet Demand = ", total_unmet)
    end
end


plt = SDDP.SpaghettiPlot(simulations)

for tech in technologies
    SDDP.add_spaghetti(plt; title="Capacity_in_$tech") do data
        return data[:x_capacity_tech][tech].in
    end
    SDDP.add_spaghetti(plt; title="Capacity_out_$tech") do data
        return data[:x_capacity_tech][tech].out
    end
    SDDP.add_spaghetti(plt; title="Expansion_$tech") do data
        return data[:u_expansion_tech][tech]
    end
    SDDP.add_spaghetti(plt; title="Production_$tech") do data
        return sum(data[:u_production_tech][tech, :])
    end
end

SDDP.add_spaghetti(plt; title="Annual Demand_in") do data
    return data[:x_annual_demand].in
end
SDDP.add_spaghetti(plt; title="Annual Demand_out") do data
    return data[:x_annual_demand].out
end
SDDP.add_spaghetti(plt; title="Total Production") do data
    return sum(sum((data[:u_production_tech][tech, hour]) for hour in 1:c_hours) for tech in technologies)
end
SDDP.add_spaghetti(plt; title="Unmet Demand") do data
    return sum(data[:u_unmet])
end

SDDP.plot(plt, "spaghetti_plot.html")

last_dispatch = [sum(value(simulations[i][6][:u_production][hour]) for hour in 1:c_hours) for i in 1:100]
# get the unique values of the last dispatch
unique_dispatch = unique(last_dispatch)