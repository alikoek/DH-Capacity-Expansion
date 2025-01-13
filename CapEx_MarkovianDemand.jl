using SDDP, Gurobi, LinearAlgebra

# Problem Data
T = 3  # Total number of years
c_hours = 8760
c_initial_capacity = 8

# Normalized hourly demand profile (for one year)
c_demand1 = [8, 7, 7, 7, 7]
c_repeated_demand1 = repeat(c_demand1, 1752) # creates normalized demand vector of length 8760
c_demand2 = [16, 17, 15, 16.5, 17]
c_repeated_demand2 = repeat(c_demand2, 1752)
c_demand3 = [32, 30, 35, 31, 32]
c_repeated_demand3 = repeat(c_demand3, 1752)

# Store base normalized profiles
normalized_profile = [
    c_repeated_demand1 ./ sum(c_repeated_demand1), # Normalize demand
    c_repeated_demand2 ./ sum(c_repeated_demand2),
    c_repeated_demand3 ./ sum(c_repeated_demand3),
]

c_expansion_cost = 50
c_operational_cost = 3
c_fixed_cost = 100
c_max_additional_capacity = 100
c_penalty = 100  # Penalty cost for unmet demand

# Define the stochastic demand multipliers: +10%, no change, -10%
demand_multipliers = [1.1, 1.0, 0.9]

# Base annual demand (total for a typical year)
base_annual_demand = sum(c_repeated_demand1)

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
        0 <= x_capacity <= 500, SDDP.State, (initial_value = c_initial_capacity)
        0 <= x_annual_demand <= 100000, SDDP.State, (initial_value = base_annual_demand)
        0 <= u_expansion <= c_max_additional_capacity
        u_production[1:c_hours] >= 0  # Hourly production decision
        u_unmet[1:c_hours] >= 0       # Unmet demand per hour
    end)

    # Investment stage (odd-numbered stages)
    if t % 2 == 1
        @constraint(sp, x_capacity.out == x_capacity.in + u_expansion)
        @constraint(sp, x_annual_demand.out == x_annual_demand.in)
        @stageobjective(sp, c_expansion_cost * u_expansion + c_fixed_cost * x_capacity.out)

        # Operational stage (even-numbered stages)
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
            normalized_demand = normalized_profile[Int(t / 2)][hour]
            actual_demand = d_annual * normalized_demand
            @constraint(sp, u_production[hour] + u_unmet[hour] == actual_demand)
            @constraint(sp, u_production[hour] <= x_capacity.in)  # Capacity constraint
        end

        @constraint(sp, x_capacity.out == x_capacity.in)
        @constraint(sp, x_annual_demand.out == d_annual)

        # Stage objective (operational costs)
        @stageobjective(sp,
            sum(c_operational_cost * u_production[hour] for hour in 1:c_hours) +
            sum(u_unmet[hour] * c_penalty for hour in 1:c_hours)
        )
    end
end

# Train the model
SDDP.train(model; iteration_limit=100)

# Retrieve results
println("Optimal Cost: ", SDDP.calculate_bound(model))

# Simulation
simulations = SDDP.simulate(model, 100, [:x_capacity, :x_annual_demand, :u_production, :u_expansion, :u_unmet])

# Print simulation results for each stage
for t in 1:(T*2)
    sp = simulations[2][t]
    if t % 2 == 1  # Expansion stage (odd stages)
        println("Year $(div(t + 1, 2)) - Expansion Stage")
        println("  Capacity_in = ", value(sp[:x_capacity].in),
            ", Expansion = ", value(sp[:u_expansion]),
            ", Capacity_out = ", value(sp[:x_capacity].out))
    else  # Operation stage (even stages)
        println("Year $(div(t, 2)) - Operation Stage")

        # Sum the production and unmet demand over all hours
        total_production = sum(value(sp[:u_production][hour]) for hour in 1:c_hours)
        total_unmet = sum(value(sp[:u_unmet][hour]) for hour in 1:c_hours)

        println("  Demand_in = ", value(sp[:x_annual_demand].in),
            ", Demand_out = ", value(sp[:x_annual_demand].out),
            " Total Production = ", total_production,
            " Total Unmet Demand = ", total_unmet)
    end
end


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
    return data[:u_production][5]
end

SDDP.plot(plt, "spaghetti_plot.html")