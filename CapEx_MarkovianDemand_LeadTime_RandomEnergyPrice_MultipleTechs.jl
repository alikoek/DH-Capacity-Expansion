using SDDP, Gurobi, LinearAlgebra, Distributions, CSV, DataFrames, Plots, Statistics

##############################################################################
# Basic Setup
##############################################################################

# Problem Data
T = 3  # Total number of years --> 2025, 2030, 2035
c_hours = 8760

# Technological parameters
# Technology Names
technologies = [:CHP, :Boiler, :HeatPump]

# Lifetime in "years" (or pairs of stages)
# e.g., lifetime[:CHP] = 2 means "2 years" of operational usage
lifetime = Dict(
    :CHP => 2,
    :Boiler => 3,     # e.g. 3-year lifetime => no retirement in 3-year horizon
    :HeatPump => 1,   # only 1-year lifetime => retires quickly
)

# Operational Costs (€/MWh)
c_operational_cost = Dict(
    :CHP => 30.0,
    :Boiler => 10.0,
    :HeatPump => 20.0
)

# Investment Costs (€/MW)
c_investment_cost = Dict(
    :CHP => 1000.0,
    :Boiler => 500.0,
    :HeatPump => 800.0
)

# Fixed O&M Costs (€/MW per year)
c_fixed_cost = Dict(
    :CHP => 50.0,
    :Boiler => 30.0,
    :HeatPump => 40.0
)

# Maximum Additional Capacity (MW)
c_max_additional_capacity = Dict(
    :CHP => 100.0,
    :Boiler => 100.0,
    :HeatPump => 100.0
)

# Initial Capacities (MW)
c_initial_capacity = Dict(
    :CHP => 25,
    :Boiler => 20,
    :HeatPump => 5,
)

# Efficiency (as a fraction)
c_efficiency = Dict(
    :CHP => 0.9,
    :Boiler => 0.85,
    :HeatPump => 3.0  # Coefficient of Performance (COP)
)

##############################################################################
# Typical hours
##############################################################################
# read excel file for load profile
filename = joinpath(@__DIR__, "LoadProfile.csv")
load_profile = CSV.read(filename, DataFrames.DataFrame, delim=";", decimal=',')
load_profile = collect(load_profile[:, "load"])
load_profile_normalized = load_profile ./ sum(load_profile)

profile = load_profile_normalized
n_typical_hours = 1000
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
print(typical_hours)

plot(load_profile_normalized, label="Original", title="Approximation versus Original")
plot!(quants[second_component[:]], label="approximation")

plot(sort(load_profile_normalized), label="Original", title="Approximation versus Original - Sorted")
plot!(sort(quants[second_component[:]]), label="approximation")

# Base annual demand
base_annual_demand = 100000.0
# Penalty cost for unmet demand (€/MWh)
c_penalty = 1000.0

load = load_profile_normalized * base_annual_demand
maximum(load)

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
    # after it's built (including the year it's built).
    return (year_current - year_invest) < lifetime[tech]
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
        # Demand state
        0 <= x_annual_demand, SDDP.State, (initial_value = base_annual_demand)
        # Unmet demand per hour
        u_unmet[1:n_typical_hours] >= 0

        # Define a dictionary of states that track expansions built at each 
        # investment stage for t=1,3,5 (since T=3 => 3 investment stages).
        0 <= cap_invest_s1[tech in technologies] <= c_max_additional_capacity[tech], SDDP.State, (initial_value = 0)
        0 <= cap_invest_s3[tech in technologies] <= c_max_additional_capacity[tech], SDDP.State, (initial_value = 0)
        0 <= cap_invest_s5[tech in technologies] <= c_max_additional_capacity[tech], SDDP.State, (initial_value = 0)
    end)

    ################### Investment stage (odd-numbered stages) ###################
    if t % 2 == 1
        # Maintain annual demand state
        @constraint(sp, x_annual_demand.out == x_annual_demand.in)
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
            if is_alive(1, t, tech)
                expr_opex_fix += c_fixed_cost[tech] * cap_invest_s1[tech].in
            end
            if is_alive(3, t, tech)
                expr_opex_fix += c_fixed_cost[tech] * cap_invest_s3[tech].in
            end
            if is_alive(5, t, tech)
                expr_opex_fix += c_fixed_cost[tech] * cap_invest_s5[tech].in
            end
        end
        @stageobjective(sp, expr_invest_cost + expr_opex_fix)

        ################### Operational stage (even-numbered stages) ###################
    else
        # ensure that annual demand is not changed in the first year
        if t == 2
            d_annual = base_annual_demand
        else
            # Annual demand adjustment based on the stochastic demand state
            d_annual = demand_multipliers[demand_state] * x_annual_demand.in
        end

        # Apply constraints for production and unmet demand based on the stochastic annual demand
        for hour in 1:n_typical_hours
            actual_demand = d_annual * typical_hours[hour][:value]

            # Total production from all technologies plus unmet demand meets the actual demand
            @constraint(sp, sum(u_production_tech[tech, hour] for tech in technologies) + u_unmet[hour] == actual_demand)

            ### Capacity constraints for each technology ###
            # Identify total capacity available
            # = sum of expansions built in each prior investment stage that is still alive
            # For each technology, 
            #   u_production_tech[tech,hour] <= (sum of alive expansions)
            for tech in technologies
                # sum expansions from s=1,3,5 that are alive
                local capacity_alive = 0.0
                if is_alive(1, t, tech)
                    capacity_alive += cap_invest_s1[tech].in
                end
                if is_alive(3, t, tech)
                    capacity_alive += cap_invest_s3[tech].in
                end
                if is_alive(5, t, tech)
                    capacity_alive += cap_invest_s5[tech].in
                end

                @constraint(sp,
                    u_production_tech[tech, hour] <= capacity_alive
                )
            end
        end

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

        # Update annual demand state
        @constraint(sp, x_annual_demand.out == d_annual)

        ### Stage objective (operational costs) ###
        SDDP.parameterize(sp, price_values, price_probabilities_normalized) do ω
            # We treat (c_operational_cost[tech] + ω) as the total variable cost
            @stageobjective(sp,
                sum(
                    (sum(
                        (c_operational_cost[tech] + ω) * u_production_tech[tech, hour] for tech in technologies
                    ) +
                     u_unmet[hour] * c_penalty
                    ) * typical_hours[hour][:qty] for hour in 1:n_typical_hours))
        end
    end
end

# Train the model
SDDP.train(model; iteration_limit=100)

# Retrieve results
println("Optimal Cost: ", SDDP.calculate_bound(model))

# Simulation
simulations = SDDP.simulate(model, 2, [:cap_invest_s1, :cap_invest_s3, :cap_invest_s5, :x_annual_demand, :u_production_tech, :u_expansion_tech, :u_unmet])

# Print simulation results
for t in 1:(T*2)
    sp = simulations[1][t]
    if t % 2 == 1  # Investment stages (odd stages)
        println("Year $(div(t + 1, 2)) - Investment Stage")
        for tech in technologies
            println("  Technology: $tech")
            println("    Expansion = ", value(sp[:u_expansion_tech][tech]))
            println("    Capacity_s1_in = ", value(sp[:cap_invest_s1][tech].in))
            println("    Capacity_s1_out = ", value(sp[:cap_invest_s1][tech].out))
            println("    Capacity_s3_in = ", value(sp[:cap_invest_s3][tech].in))
            println("    Capacity_s3_out = ", value(sp[:cap_invest_s3][tech].out))
            println("    Capacity_s5_in = ", value(sp[:cap_invest_s5][tech].in))
            println("    Capacity_s5_out = ", value(sp[:cap_invest_s5][tech].out))
        end
    else  # Operational stages
        println("Year $(div(t, 2)) - Operational Stage")
        println("   Demand_in = ", value(sp[:x_annual_demand].in))
        println("   Demand_out = ", value(sp[:x_annual_demand].out))
        println("  Total Production = ", total_production)
        total_production = sum(sum(value(sp[:u_production_tech][tech, hour]) for hour in 1:c_hours) for tech in technologies)
        total_unmet = sum(value(sp[:u_unmet][hour]) for hour in 1:c_hours)
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