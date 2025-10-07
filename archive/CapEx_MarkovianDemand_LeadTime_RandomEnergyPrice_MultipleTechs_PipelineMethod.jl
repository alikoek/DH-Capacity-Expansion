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

# Lifetime in "years" (or pairs of stages)
c_lifetime = Dict(
    :CHP => 3,
    :Boiler => 3,
    :HeatPump => 2,
    :Geothermal => 3
)

# Maximum age buckets (0 = retired, 1 to max_lifetime = operational ages)
max_lifetime = maximum(values(c_lifetime))

# Initial Capacities by age (MW_th)
# Format: initial_cap_dict[(tech, age)] where age is remaining operational years
function create_initial_cap_dict(technologies::Vector{Symbol}, max_lifetime::Int)
    initial_cap_dict = Dict{Tuple{Symbol, Int}, Float64}()
    
    for tech in technologies
        for age in 0:max_lifetime
            initial_cap_dict[(tech, age)] = 0.0
        end
    end
    
    return initial_cap_dict
end

initial_cap_dict = create_initial_cap_dict(technologies, max_lifetime)
# Set initial capacities with their remaining lifetimes
initial_cap_dict[(:CHP, 2)] = 500.0      # CHP with 2 years remaining
initial_cap_dict[(:Boiler, 1)] = 350.0   # Boiler with 1 year remaining
initial_cap_dict[(:HeatPump, 2)] = 0.0   # No initial heat pumps
initial_cap_dict[(:Geothermal, 0)] = 0.0 # No initial geothermal (retired)

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
discount_rate = 0.05
function discount_factor(t::Int, T_years::Int)
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

plot(load_profile_normalized, label="Original", title="Approximation versus Original")
plot!(quants[second_component[:]], label="approximation")

plot(sort(load_profile_normalized), label="Original", title="Approximation versus Original - Sorted")
plot!(sort(quants[second_component[:]]), label="approximation")

# Base annual demand
base_annual_demand = 2e6 # MWh

# Compute the absolute demand profile for the first year.
c_base_demand_profile = [base_annual_demand * typical_hours[i][:value] for i in 1:n_typical_hours]
c_base_demand_profile = round.(c_base_demand_profile)
maximum(c_base_demand_profile)

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
# Create SDDP model with Pipeline Method
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
        0 <= x_demand_mult, SDDP.State, (initial_value = 1.0)
        
        # Pipeline capacity variables: cap[tech, age] where age is remaining operational years
        # age = 0: retired capacity (not used)
        # age = 1 to c_lifetime[tech]: operational capacity with remaining life
        0 <= cap[tech in technologies, age in 0:max_lifetime] <= 3000, SDDP.State, (initial_value = initial_cap_dict[tech, age])
    end)

    ################### Investment stage (odd-numbered stages) ###################
    if t % 2 == 1
        # No change in the demand multiplier in investment stages.
        @constraint(sp, x_demand_mult.out == x_demand_mult.in)
        
        # Pipeline capacity transitions for investment stages
        for tech in technologies
            for age in 0:max_lifetime
                if age == c_lifetime[tech]
                    # New investments enter at maximum operational age
                    @constraint(sp, cap[tech, age].out == cap[tech, age].in + u_expansion_tech[tech])
                else
                    # All other ages remain unchanged during investment
                    @constraint(sp, cap[tech, age].out == cap[tech, age].in)
                end
            end
        end

        # Stage objective: expansion cost + fixed O&M costs
        local df = discount_factor(t, T_years)

        # Expansion cost
        expr_invest_cost = sum(
            c_investment_cost[tech] * u_expansion_tech[tech]
            for tech in technologies
        )
        
        # Fixed O&M costs for operational capacity (age >= 1)
        expr_opex_fix = sum(
            c_opex_fixed[tech] * cap[tech, age].in
            for tech in technologies, age in 1:c_lifetime[tech]
        )
        
        # Multiply the O&M by T_years because each stage represents T_years
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
        # Total production from all technologies plus unmet demand meets the actual demand
        @constraint(sp, [hour in 1:n_typical_hours],
            sum(u_production_tech[tech, hour] for tech in technologies) + u_unmet[hour] ==
            c_base_demand_profile[hour] * new_demand_mult
        )

        # Capacity constraints for each technology and hour:
        # Total operational capacity is sum of all age buckets >= 1
        @constraint(sp, [tech in technologies, hour in 1:n_typical_hours],
            u_production_tech[tech, hour] <= sum(cap[tech, age].in for age in 1:c_lifetime[tech])
        )

        # Pipeline aging: capacity moves down one age bucket each operational period
        for tech in technologies
            # Capacity ages by one year (moves down one age bucket)
            for age in 1:max_lifetime
                if age == 1
                    # Age 1 capacity retires (goes to age 0 = retired)
                    @constraint(sp, cap[tech, 0].out == cap[tech, 1].in)
                else
                    # Higher age capacity moves down one age level
                    @constraint(sp, cap[tech, age-1].out == cap[tech, age].in)
                end
            end
            # No capacity at maximum age after aging (it either came from higher ages or was retired)
            @constraint(sp, cap[tech, max_lifetime].out == 0.0)
        end

        # Update the state variable for the cumulative demand multiplier.
        @constraint(sp, x_demand_mult.out == new_demand_mult)

        # If this is the *final* stage, include salvage for any capacity that extends beyond planning horizon
        local salvage = 0.0
        if t == T * 2
            for tech in technologies
                for age in 1:c_lifetime[tech]
                    # Salvage value based on remaining operational life
                    salvage += c_investment_cost[tech] * cap[tech, age].in * (age / c_lifetime[tech])
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
            # multiply by the T_years-year block and discount factor
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

# Simulation
simulation_symbols = [:cap, :x_demand_mult, :u_production_tech, :u_expansion_tech, :u_unmet]
simulations = SDDP.simulate(model, 400, simulation_symbols)

# Print simulation results
for t in 1:(T*2)
    sp = simulations[3][t]
    if t % 2 == 1  # Investment stages (odd stages)
        println("Year $(div(t + 1, 2)) - Investment Stage")
        for tech in technologies
            println("  Technology: $tech")
            println("    Expansion = ", value(sp[:u_expansion_tech][tech]))
            for age in 0:max_lifetime
                println("    Capacity_age$(age)_in = ", value(sp[:cap][tech, age].in))
                println("    Capacity_age$(age)_out = ", value(sp[:cap][tech, age].out))
            end
        end
    else  # Operational stages
        println("Year $(div(t, 2)) - Operational Stage")
        println("Noise term: ", value(sp[:noise_term]))
        println("   Demand Multiplier (in/out) = ", value(sp[:x_demand_mult].in), " / ", value(sp[:x_demand_mult].out))
        println("   Annual Demand = ", sum(c_base_demand_profile[i] * value(sp[:x_demand_mult].out) * typical_hours[i][:qty] for i in 1:n_typical_hours))
        for tech in technologies
            # Sum operational capacity (age >= 1)
            local capacity_alive = sum(value(sp[:cap][tech, age].in) for age in 1:c_lifetime[tech])
            println("    $tech alive Capacity = ", capacity_alive)
        end
        total_production = sum(sum(value(sp[:u_production_tech][tech, hour] * typical_hours[hour][:qty]) for hour in 1:n_typical_hours) for tech in technologies)
        total_unmet = sum(value(sp[:u_unmet][hour] * typical_hours[hour][:qty]) for hour in 1:n_typical_hours)
        println("  Total Production = ", total_production)
        println("  Total Unmet Demand = ", total_unmet)
        
        # Calculate salvage value at final stage
        local salvage = 0.0
        if t == T * 2
            for tech in technologies
                for age in 1:c_lifetime[tech]
                    salvage_value = c_investment_cost[tech] * value(sp[:cap][tech, age].in) * (age / c_lifetime[tech])
                    salvage += salvage_value
                    if salvage_value > 0
                        println("   Salvage Value of $tech (age $age) = ", salvage_value)
                    end
                end
            end
        end
        if salvage > 0
            println("  Total Salvage Value = ", salvage * salvage_fraction)
        end
    end
end

##############################################################################
# Visualizations
##############################################################################
################################# Spaghetti Plot #############################
##############################################################################
plt = SDDP.SpaghettiPlot(simulations)

for tech in technologies
    SDDP.add_spaghetti(plt; title="Total_Capacity_$tech") do data
        return sum(data[:cap][tech, age].in for age in 1:c_lifetime[tech])
    end
    SDDP.add_spaghetti(plt; title="Expansion_$tech") do data
        return data[:u_expansion_tech][tech]
    end
    SDDP.add_spaghetti(plt; title="Production_$tech") do data
        return sum(data[:u_production_tech][tech, :])
    end
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

SDDP.plot(plt, "spaghetti_plot_pipeline.html")

############################## Publication Plot ################################
Plots.plot(
    SDDP.publication_plot(simulations; title="Expansion") do data
        return data[:u_expansion_tech][:HeatPump]
    end,
    SDDP.publication_plot(simulations; title="Thermal generation") do data
        return sum((data[:u_production_tech][:HeatPump, hour]) for hour in 1:n_typical_hours)
    end;
    xlabel="Stage",
    layout=(1, 2),
)

Plots.plot(
    SDDP.publication_plot(simulations; title="Objective") do data
        return data[:stage_objective]
    end,
    xlabel="Stage",
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
    return (p)
end

for tech in technologies
    total_cap_alive = zeros(length(simulations), T * 2)
    for sim in (1:length(simulations))
        for t in (1:T*2)
            cap_alive = sum(value(simulations[sim][t][:cap][tech, age].in) for age in 1:c_lifetime[tech])
            total_cap_alive[sim, t] = cap_alive
        end
    end
    if tech == :Boiler
        legend_pos = :topright
    else
        legend_pos = :topleft
    end

    fig = plot_bands(total_cap_alive, "$tech Alive Capacities", 1:8, 1:8, "Stages", "Capacity [MW_th]", legend_pos)
    display(fig)
    savefig("AliveCapacities_Pipeline_$tech.png")
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
    if tech == :Geothermal
        legend_pos = :topleft
    else
        legend_pos = :topright
    end
    fig = plot_bands(u_invs, "$tech Investment", [1, 3, 5, 7], [1, 3, 5, 7], "Investment Stages", "Investment [MW_th]", legend_pos)
    display(fig)
    savefig("Investments_Pipeline_$tech.png")
end