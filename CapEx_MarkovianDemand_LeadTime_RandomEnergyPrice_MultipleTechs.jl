using SDDP, Gurobi, LinearAlgebra, Distributions, CSV, DataFrames

# Problem Data
T = 3  # Total number of years
c_hours = 8760

# read excel file for load profile
filename = joinpath(@__DIR__, "LoadProfile.csv")
load_profile = CSV.read(filename, DataFrames.DataFrame, delim=";", decimal = ',')
load_profile = collect(load_profile[:, "load"])
load_profile_normalized = load_profile ./ sum(load_profile)

# Base annual demand
base_annual_demand = 100000.0
# Penalty cost for unmet demand (€/MWh)
c_penalty = 1000.0

load = load_profile_normalized * base_annual_demand
maximum(load)

# Technological parameters
# Technology Names
technologies = [:CHP, :Boiler, :HeatPump]

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

# Define the stochastic demand multipliers: +10%, no change, -10%
demand_multipliers = [1.1, 1.0, 0.9]

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
        # Capacity State Variables for each technology
        0 <= x_capacity_tech[tech in technologies] <= 1000, SDDP.State, (initial_value = c_initial_capacity[tech])
        # Investment pipeline.
        #   pipeline.out are the investments made in the current stage
        #   pipeline.in are the investments that becomes available at this stage
        #     and can be added to the capacity.
        #   Investments move up one slot in the pipeline each stage.
        # Pipeline State Variables for each technology
        0 <= pipeline_tech[tech in technologies] <= c_max_additional_capacity[tech], SDDP.State, (initial_value = 0)
        # Investment Decision Variables for each technology
        0 <= u_expansion_tech[tech in technologies] <= c_max_additional_capacity[tech]
        # Production Variables for each technology
        u_production_tech[tech in technologies, 1:c_hours] >= 0
        # Demand state
        0 <= x_annual_demand, SDDP.State, (initial_value = base_annual_demand)
        # Unmet demand per hour
        u_unmet[1:c_hours] >= 0
    end)

    ################### Investment stage (odd-numbered stages) ###################
    if t % 2 == 1
        # Update capacity without adding pipeline contributions
        @constraint(sp, [tech in technologies], x_capacity_tech[tech].out == x_capacity_tech[tech].in)
        # Maintain annual demand state
        @constraint(sp, x_annual_demand.out == x_annual_demand.in)
        # Add the expansion decision to the pipeline for each technology
        @constraint(sp, [tech in technologies], pipeline_tech[tech].out == u_expansion_tech[tech])
        # Stage objective: expansion cost
        @stageobjective(sp, 
        sum(c_investment_cost[tech] * u_expansion_tech[tech] for tech in technologies) +
        sum(c_fixed_cost[tech] * x_capacity_tech[tech].out for tech in technologies)
        )

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
        for hour in 1:c_hours
            actual_demand = d_annual * load_profile_normalized[hour]
        
            # Total production from all technologies plus unmet demand meets the actual demand
            @constraint(sp, sum(u_production_tech[tech, hour] for tech in technologies) + u_unmet[hour] == actual_demand)
        
            # Capacity constraints for each technology
            @constraint(sp, [tech in technologies], u_production_tech[tech, hour] <= x_capacity_tech[tech].in)
        end

        # Update capacity with the investments in the pipeline  for each technology
        @constraint(sp, [tech in technologies], x_capacity_tech[tech].out == x_capacity_tech[tech].in + pipeline_tech[tech].in) 

        # Update annual demand state
        @constraint(sp, x_annual_demand.out == d_annual)

        # Reset the pipeline for each technology
        @constraint(sp, [tech in technologies], pipeline_tech[tech].out == 0)

        # Stage objective (operational costs)
        @stageobjective(sp,
            sum(c_operational_cost[tech] * u_production_tech[tech, hour] for tech in technologies, hour in 1:c_hours) +
            sum(u_unmet[hour] * c_penalty for hour in 1:c_hours)
            )
        
        SDDP.parameterize(sp, price_values, price_probabilities_normalized) do ω
            # We treat (c_operational_cost[tech] + ω) as the total variable cost
            @stageobjective(sp,
                sum( (c_operational_cost[tech] + ω) * u_production_tech[tech, hour]
                    for tech in technologies, hour in 1:c_hours ) +
                sum(u_unmet[hour] * c_penalty for hour in 1:c_hours)
            )
        end
        # Stage objective (operational costs)
        #SDDP.parameterize(sp, price_values, price_probabilities_normalized) do ω
        #    @stageobjective(sp,
        #        sum(c_operational_cost * u_production[hour] for hour in 1:c_hours) +
        #        sum(u_unmet[hour] * c_penalty for hour in 1:c_hours) +
        #        sum(ω * u_production[hour] for hour in 1:c_hours)
        #    )
        #end
    end
end

# Train the model
SDDP.train(model; iteration_limit=50)

# Retrieve results
println("Optimal Cost: ", SDDP.calculate_bound(model))

# Simulation
simulations = SDDP.simulate(model, 100, [:x_capacity_tech, :x_annual_demand, :u_production_tech, :u_expansion_tech, :u_unmet])

# Print simulation results

for t in 1:(T*2)
    sp = simulations[1][t]
    if t % 2 == 1  # Investment stages (odd stages)
        println("Year $(div(t + 1, 2)) - Investment Stage")
        for tech in technologies
            println("  Technology: $tech")
            println("    Capacity_in = ", value(sp[:x_capacity_tech][tech].in))
            println("    Expansion = ", value(sp[:u_expansion_tech][tech]))
            println("    Capacity_out = ", value(sp[:x_capacity_tech][tech].out))
        end
    else  # Operational stages
        println("Year $(div(t, 2)) - Operational Stage")
        println("   Demand_in = ", value(sp[:x_annual_demand].in))
        println("   Demand_out = ", value(sp[:x_annual_demand].out))
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

    SDDP.add_spaghetti(plt; title="Annual Demand_in") do data
        return data[:x_annual_demand].in
    end

    SDDP.add_spaghetti(plt; title="Annual Demand_out") do data
        return data[:x_annual_demand].out
    end

    SDDP.add_spaghetti(plt; title="Unmet Demand") do data
        return sum(data[:u_unmet])
    end
end

SDDP.plot(plt, "spaghetti_plot.html")


plt = SDDP.SpaghettiPlot(simulations)

SDDP.add_spaghetti(plt; title="Capacity_out") do data
    return data[:x_capacity].out
end

SDDP.add_spaghetti(plt; title="Capacity_in") do data
    return data[:x_capacity].in
end

SDDP.add_spaghetti(plt; title="Expansion") do data
    return data[:u_expansion]
end

SDDP.add_spaghetti(plt; title="Dispatch") do data
    return sum(data[:u_production])
end

SDDP.plot(plt, "spaghetti_plot.html")


last_dispatch = [sum(value(simulations[i][6][:u_production][hour]) for hour in 1:c_hours) for i in 1:100]
# get the unique values of the last dispatch
unique_dispatch = unique(last_dispatch)