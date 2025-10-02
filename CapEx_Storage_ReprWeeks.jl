using SDDP, Gurobi, LinearAlgebra, Distributions, CSV, DataFrames, Plots, Statistics, StatsPlots, Random, Dates

##############################################################################
# Basic Setup
##############################################################################

# Problem Data
T = 4  # Total number of model years --> 2020, 2030, 2040, 2050
T_years = 10 # number of years represented by the model years

# Investment stages (odd-numbered stages plus stage 0 for initial capacities)
investment_stages = [0; collect(1:2:(2*T-1))]  # [0, 1, 3, 5, 7] for T=4

# Technological parameters
technologies = [:CHP, :Boiler, :HeatPump, :Geothermal]

# Storage parameters
storage_params = Dict(
    :capacity_cost => 25000,        # €/MWh storage capacity
    :fixed_om => 500,               # €/MWh/year
    :variable_om => 0.1,            # €/MWh discharged
    :efficiency => 0.95,            # Round-trip efficiency
    :loss_rate => 0.02,             # Daily heat loss rate (% per day)
    :max_charge_rate => 0.25,       # Can charge 25% of capacity per hour
    :max_discharge_rate => 0.25,    # Can discharge 25% of capacity per hour
    :lifetime => 4,                 # Model years
    :max_capacity => 1000,          # Maximum MWh storage capacity
    :initial_capacity => 0          # No initial storage capacity
)

# Lifetime in "model years"
c_lifetime_new = Dict(
    :CHP => 3,
    :Boiler => 3,
    :HeatPump => 2,
    :Geothermal => 3
)

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
    :Boiler => 0.75
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

# Efficiencies
c_efficiency_th = Dict(
    :CHP => 0.44,
    :Boiler => 0.9,
    :HeatPump => 3.0,
    :Geothermal => 1.0
)

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

# Penalty cost for unmet demand (€/MWh)
c_penalty = 100000

# Discount factor function
discount_rate = 0.05
function discount_factor(t::Int, T_years::Int)
    exponent = T_years * (ceil(t / 2) - 1)
    return 1.0 / (1.0 + discount_rate)^exponent
end

##############################################################################
# Load Representative Weeks from TSAM output
##############################################################################

println("Loading representative weeks data...")
hours_per_week = 168  # 7 * 24 hours

# Load TSAM data
filename_tsam = joinpath(@__DIR__, "typical_weeks.csv")
tsam_data = CSV.read(filename_tsam, DataFrames.DataFrame)

# Group rows by typical-week id and extract 168-hour profiles
week_ids = sort(unique(tsam_data[!, "Typical Week"]))
@assert all(count(==(w), tsam_data[!, "Typical Week"]) == hours_per_week for w in week_ids) "Each week must have 168 rows."

insertcols!(tsam_data, :hour_in_week => zeros(Int, nrow(tsam_data)))
for w in week_ids
    idx = findall(tsam_data[!, "Typical Week"] .== w)
    @assert length(idx) == hours_per_week
    tsam_data[idx, :hour_in_week] .= 1:hours_per_week            # note the `.=` broadcast
end

representative_weeks = [collect(tsam_data[tsam_data[!, "Typical Week"] .== w, "load"]) for w in week_ids]
n_weeks = length(representative_weeks)

week_weights = [mean(skipmissing(tsam_data[tsam_data[!, "Typical Week"] .== w, "weight_abs"])) |> float for w in week_ids]
# Normalize weights to 52 weeks
week_weights_normalized = week_weights .* (52 / sum(week_weights))

# Scale demand to annual total
base_annual_demand = 2e6  # MWh
scaling_factor = base_annual_demand /
    sum(sum(week) * w for (week, w) in zip(representative_weeks, week_weights_normalized))

scaled_weeks = [week .* scaling_factor for week in representative_weeks]

println("Representative weeks loaded: $(n_weeks) * $(hours_per_week)h; weights sum = $(sum(week_weights_normalized))")
println("Week weights: ", week_weights_normalized)

# Demand multipliers for Markovian states
demand_multipliers = [1.1, 1.0, 0.9]

##############################################################################
# Electricity Prices Processing for Representative Weeks
##############################################################################

# NOTE: This is a temporary workaround
# TODO: Integrate electricity price scenarios into TSAM and extract representative weeks directly

println("Processing electricity prices...")

# Load or create electricity price data
filename_elec_2030 = joinpath(@__DIR__, "ElectricityPrice2030.csv")
filename_elec_2050 = joinpath(@__DIR__, "ElectricityPrice2050.csv")

# Load actual price data
elec_price_2030_full = CSV.read(filename_elec_2030, DataFrames.DataFrame, delim=",", decimal='.')[:, "price"]
elec_price_2050_full = CSV.read(filename_elec_2050, DataFrames.DataFrame, delim=",", decimal='.')[:, "price"]

# p_full: Vector{Float64} length 8760 (or 8784 in leap years)
# start_dt: DateTime for the first price sample (e.g., DateTime(2030,1,1,0,0,0))
function hourly_profile_from_full_year(p_full::AbstractVector{<:Real}, start_dt::DateTime)
    H = 168
    sums  = zeros(Float64, H)
    counts = zeros(Int, H)

    for (k, p) in pairs(p_full)
        t  = start_dt + Hour(k-1)
        # Julia: dayofweek(Mon=1,…,Sun=7), hour(t)=0…23  → index 1…168
        hW = (dayofweek(t)-1) * 24 + hour(t) + 1
        sums[hW]   += p
        counts[hW] += 1
    end
    @assert all(counts .> 0)
    return sums ./ counts  # 168-length average price for each hour-of-week
end

p2030_hw = hourly_profile_from_full_year(elec_price_2030_full, DateTime(2030,1,1))
p2050_hw = hourly_profile_from_full_year(elec_price_2050_full, DateTime(2050,1,1))

# Build (n_weeks × 168) matrices consistent with rep-week indexing
sale_elec_price_2030_weeks = repeat(permutedims(p2030_hw), n_weeks, 1)
sale_elec_price_2050_weeks = repeat(permutedims(p2050_hw), n_weeks, 1)

# For now, assume purchase prices as a fraction:
purch_elec_price_2030_weeks  = sale_elec_price_2030_weeks .* 1.2
purch_elec_price_2050_weeks  = sale_elec_price_2050_weeks .* 1.2

##############################################################################
# MarkovianPolicyGraph Setup
##############################################################################

I = Diagonal(ones(3))

transition_matrices = Array{Float64,2}[
    [1.0]',
    [1.0]',
    [0.3 0.5 0.2],
    I,
    [0.3 0.5 0.2; 0.3 0.5 0.2; 0.3 0.5 0.2],
    I,
    [0.3 0.5 0.2; 0.3 0.5 0.2; 0.3 0.5 0.2],
    I,
]

##############################################################################
# Energy Price Distribution
##############################################################################

mean_price = 37.0
price_volatility = 10.0
μ_normal = log(mean_price^2 / sqrt(price_volatility^2 + mean_price^2))
σ_normal = sqrt(log(1 + (price_volatility / mean_price)^2))
price_distribution = LogNormal(μ_normal, σ_normal)
num_price_scenarios = 4
price_quantiles = range(0.05, 0.95; length=num_price_scenarios)
price_values = quantile.(price_distribution, price_quantiles)
price_probabilities = pdf(price_distribution, price_values)
price_probabilities_normalized = price_probabilities / sum(price_probabilities)

##############################################################################
# Helper Functions
##############################################################################

function is_alive(s_invest::Int, s_current::Int, lifetime::Dict, tech::Symbol)
    year_invest = ceil(s_invest / 2)
    year_current = ceil(s_current / 2)
    return (1 <= (year_current - year_invest)) && ((year_current - year_invest) <= lifetime[tech])
end

function is_storage_alive(s_invest::Int, s_current::Int)
    year_invest = ceil(s_invest / 2)
    year_current = ceil(s_current / 2)
    return (1 <= (year_current - year_invest)) && ((year_current - year_invest) <= storage_params[:lifetime])
end

##############################################################################
# Create SDDP Model
##############################################################################

println("Building SDDP model...")

model = SDDP.MarkovianPolicyGraph(
    transition_matrices=transition_matrices,
    sense=:Min,
    lower_bound=0.0,
    optimizer=Gurobi.Optimizer
) do sp, node
    t, demand_state = node

    # Variables
    @variables(sp, begin
        # Investment decisions for technologies
        0 <= u_expansion_tech[tech in technologies] <= c_max_additional_capacity[tech]
        # Investment decision for storage
        0 <= u_expansion_storage <= storage_params[:max_capacity]
        
        # Production variables
        u_production[tech in technologies, week=1:n_weeks, hour=1:hours_per_week] >= 0
        
        # Storage variables
        0 <= u_charge[week=1:n_weeks, hour=1:hours_per_week]
        0 <= u_discharge[week=1:n_weeks, hour=1:hours_per_week]
        0 <= u_level[week=1:n_weeks, hour=1:hours_per_week]
        
        # Unmet demand
        0 <= u_unmet[week=1:n_weeks, hour=1:hours_per_week]
        
        # State variable for demand multiplier
        0 <= x_demand_mult, SDDP.State, (initial_value = 1.0)
    end)

    # Create vintage capacity variables for technologies
    cap_vintage_tech = Dict()
    cap_vintage_stor = Dict()
    
    for stage in investment_stages
        if stage == 0
            # Initial capacities
            cap_vintage_tech[stage] = @variable(sp, [tech in technologies], 
                SDDP.State, (initial_value = c_initial_capacity[tech]), 
                lower_bound = 0, upper_bound = 3000, 
                base_name = "cap_vintage_tech_$(stage)")
        else
            # New investment capacities
            cap_vintage_tech[stage] = @variable(sp, [tech in technologies], 
                SDDP.State, (initial_value = 0), 
                lower_bound = 0, upper_bound = c_max_additional_capacity[tech], 
                base_name = "cap_vintage_tech_$(stage)")
        end
    end

    @variable(
    sp,
        cap_vintage_stor_state[s in investment_stages], SDDP.State,
        (initial_value = s == 0 ? storage_params[:initial_capacity] : 0.0),
        lower_bound = 0,
        upper_bound = storage_params[:max_capacity],
        base_name = "cap_vintage_stor_$(s)"
    )
    
    for stage in investment_stages
        cap_vintage_stor[stage] = cap_vintage_stor_state[stage]
    end

    # Initial capacities stay constant
    @constraint(sp, [tech in technologies],
        cap_vintage_tech[0][tech].out == cap_vintage_tech[0][tech].in)
    @constraint(sp, cap_vintage_stor[0].out == cap_vintage_stor[0].in)

    ################### Investment Stage ###################
    if t % 2 == 1
        @constraint(sp, x_demand_mult.out == x_demand_mult.in)
        
        # Update vintage capacities for technologies and storage
        for stage in investment_stages[2:end]
            if stage == t
                # Current investment stage - add new expansions
                @constraint(sp, [tech in technologies],
                    cap_vintage_tech[stage][tech].out == cap_vintage_tech[stage][tech].in + u_expansion_tech[tech])
                @constraint(sp, 
                    cap_vintage_stor[stage].out == cap_vintage_stor[stage].in + u_expansion_storage)
            else
                # Other stages - no change
                @constraint(sp, [tech in technologies],
                    cap_vintage_tech[stage][tech].out == cap_vintage_tech[stage][tech].in)
                @constraint(sp, 
                    cap_vintage_stor[stage].out == cap_vintage_stor[stage].in)
            end
        end

        # Investment objective
        local df = discount_factor(t, T_years)
        
        # Investment costs
        expr_invest = sum(c_investment_cost[tech] * u_expansion_tech[tech] for tech in technologies)
        expr_invest += storage_params[:capacity_cost] * u_expansion_storage
        
        # Fixed O&M costs for technologies
        expr_fix_om = 0.0
        for tech in technologies
            for stage in investment_stages
                lifetime_dict = (stage == 0) ? c_lifetime_initial : c_lifetime_new
                if is_alive(stage, t, lifetime_dict, tech)
                    expr_fix_om += c_opex_fixed[tech] * cap_vintage_tech[stage][tech].in
                end
            end
        end
        
        # Fixed O&M for storage
        for stage in investment_stages
            if is_storage_alive(stage, t)
                expr_fix_om += storage_params[:fixed_om] * cap_vintage_stor[stage].in
            end
        end
        
        expr_fix_om *= T_years
        @stageobjective(sp, df * (expr_invest + expr_fix_om))

    ################### Operational Stage ###################
    else
        model_year = Int(ceil(t / 2))
        
        # Update demand multiplier
        if t == 2
            new_demand_mult = 1.0
        else
            new_demand_mult = demand_multipliers[demand_state] * x_demand_mult.in
        end

        # Compute alive capacities for technologies
        capacity_alive = Dict{Symbol,JuMP.GenericAffExpr{Float64,VariableRef}}()
        for tech in technologies
            capacity_expr = 0.0
            for stage in investment_stages
                lifetime_dict = (stage == 0) ? c_lifetime_initial : c_lifetime_new
                if is_alive(stage, t, lifetime_dict, tech)
                    capacity_expr += cap_vintage_tech[stage][tech].in
                end
            end
            capacity_alive[tech] = capacity_expr
        end
        
        # Compute alive storage capacity
        storage_cap = 0.0
        for stage in investment_stages
            if is_storage_alive(stage, t)
                storage_cap += cap_vintage_stor[stage].in
            end
        end

        # Operational constraints for each representative week
        for week in 1:n_weeks
            # Demand balance
            @constraint(sp, [hour in 1:hours_per_week],
                sum(u_production[tech, week, hour] for tech in technologies) + 
                u_discharge[week, hour] + u_unmet[week, hour] ==
                scaled_weeks[week][hour] * new_demand_mult
            )

            # Technology capacity constraints
            @constraint(sp, [tech in technologies, hour in 1:hours_per_week],
                u_production[tech, week, hour] <= capacity_alive[tech]
            )
            
            # Storage rate constraints
            @constraint(sp, [hour in 1:hours_per_week],
                u_charge[week, hour] <= storage_params[:max_charge_rate] * storage_cap
            )
            
            @constraint(sp, [hour in 1:hours_per_week],
                u_discharge[week, hour] <= storage_params[:max_discharge_rate] * storage_cap
            )
            
            # Storage capacity constraint
            @constraint(sp, [hour in 1:hours_per_week],
                u_level[week, hour] <= storage_cap
            )
            
            # Storage dynamics
            for hour in 1:hours_per_week
                if hour == 1
                    # First hour: start empty
                    @constraint(sp, 
                        u_level[week, hour] == 
                        storage_params[:efficiency] * u_charge[week, hour] - u_discharge[week, hour]
                    )
                else
                    # Subsequent hours: include previous level and losses
                    @constraint(sp, 
                        u_level[week, hour] == 
                        u_level[week, hour-1] * (1 - storage_params[:loss_rate]/24) +
                        storage_params[:efficiency] * u_charge[week, hour] - u_discharge[week, hour]
                    )
                end
            end
            
            # End-of-week constraint: storage must be nearly empty
            @constraint(sp, u_level[week, hours_per_week] <= 0.01 * storage_cap)
        end

        # State updates
        @constraint(sp, x_demand_mult.out == new_demand_mult)
        
        for stage in investment_stages[2:end]
            @constraint(sp, [tech in technologies],
                cap_vintage_tech[stage][tech].out == cap_vintage_tech[stage][tech].in)
            @constraint(sp, 
                cap_vintage_stor[stage].out == cap_vintage_stor[stage].in)
        end

        # Salvage value calculation
        local salvage = 0.0
        if t == T * 2
            # Technology salvage
            for tech in technologies
                for stage in investment_stages
                    stage_year = (stage == 0) ? 0 : Int(ceil(stage / 2))
                    lifetime_dict = (stage == 0) ? c_lifetime_initial : c_lifetime_new
                    
                    if (model_year - stage_year) < lifetime_dict[tech]
                        remaining_life = lifetime_dict[tech] - (model_year - stage_year)
                        salvage += c_investment_cost[tech] * cap_vintage_tech[stage][tech].in * 
                                  (remaining_life / lifetime_dict[tech])
                    end
                end
            end
            
            # Storage salvage
            for stage in investment_stages
                stage_year = (stage == 0) ? 0 : Int(ceil(stage / 2))
                if (model_year - stage_year) < storage_params[:lifetime]
                    remaining_life = storage_params[:lifetime] - (model_year - stage_year)
                    salvage += storage_params[:capacity_cost] * cap_vintage_stor[stage].in * 
                              (remaining_life / storage_params[:lifetime])
                end
            end
        end

        # Operational objective with random energy prices
        SDDP.parameterize(sp, price_values, price_probabilities_normalized) do ω
            local df = discount_factor(t, T_years)
            local purch_elec_price = model_year <= 2 ? purch_elec_price_2030_weeks : purch_elec_price_2050_weeks
            local sale_elec_price = model_year <= 2 ? sale_elec_price_2030_weeks : sale_elec_price_2050_weeks

            local expr_annual_cost = 0.0
            
            for week in 1:n_weeks
                week_cost = 0.0
                for hour in 1:hours_per_week
                    # Technology production costs
                    for tech in technologies
                        fuel_cost = 0.0
                        if c_energy_carrier[tech] == :nat_gas
                            fuel_cost = ω
                        elseif c_energy_carrier[tech] == :elec
                            fuel_cost = purch_elec_price[week, hour]
                        end
                        
                        tech_cost = c_opex_var[tech] * u_production[tech, week, hour] +
                                   (fuel_cost + c_carbon_price[model_year] * c_emission_fac[c_energy_carrier[tech]]) * 
                                   (u_production[tech, week, hour] / c_efficiency_th[tech]) -
                                   c_efficiency_el[tech] * sale_elec_price[week, hour] * 
                                   (u_production[tech, week, hour] / c_efficiency_th[tech])
                        week_cost += tech_cost
                    end
                    
                    # Storage operational cost (only on discharge)
                    week_cost += storage_params[:variable_om] * u_discharge[week, hour]
                    
                    # Unmet demand penalty
                    week_cost += c_penalty * u_unmet[week, hour]
                end
                
                # Apply week weight
                expr_annual_cost += week_weights_normalized[week] * week_cost
            end
            
            @stageobjective(sp, df * (T_years * expr_annual_cost - salvage * salvage_fraction))
        end
    end
end

##############################################################################
# Train and Simulate
##############################################################################

println("Training SDDP model...")
SDDP.train(model; risk_measure=SDDP.CVaR(0.95), iteration_limit=400, log_frequency=100)

println("Optimal Cost: ", SDDP.calculate_bound(model))

# Set seed for reproducibility
Random.seed!(1234)

println("Running simulations...")
simulation_symbols = [:x_demand_mult, :u_production, :u_expansion_tech, :u_expansion_storage, 
                     :u_charge, :u_discharge, :u_level, :u_unmet]
simulations = SDDP.simulate(model, 400, simulation_symbols)

println("Simulations complete. Generating visualizations...")

##############################################################################
# Visualization Functions
##############################################################################

function plot_bands(data, title, x_values, xticks_vals, xlabel_str, ylabel_str, legend_pos)
    # Compute statistics for each column
    q5 = [quantile(data[:, i], 0.05) for i in 1:size(data, 2)]
    q25 = [quantile(data[:, i], 0.25) for i in 1:size(data, 2)]
    q50 = [quantile(data[:, i], 0.50) for i in 1:size(data, 2)]
    q75 = [quantile(data[:, i], 0.75) for i in 1:size(data, 2)]
    q95 = [quantile(data[:, i], 0.95) for i in 1:size(data, 2)]
    
    # Create plot
    p = plot(x_values, q5, fillrange=q95, fillalpha=0.15, c=2, label="5-95 percentile", 
             legend=legend_pos, lw=0, xticks=xticks_vals)
    plot!(x_values, q25, fillrange=q75, fillalpha=0.35, c=2, label="25-75 percentile", lw=0)
    plot!(x_values, q50, label="Median", lw=2, color=2)
    
    xlabel!(xlabel_str)
    ylabel!(ylabel_str)
    title!(title)
    
    return p
end

##############################################################################
# Generate Visualizations
##############################################################################

# 1. Alive Capacities Plot
println("Generating capacity plots...")
for tech in technologies
    total_cap_alive = zeros(length(simulations), T * 2)
    for sim in 1:length(simulations)
        for t in 1:(T*2)
            cap_alive = 0
            # Extract vintage capacities from simulation
            # Note: This requires access to state variables which may need adjustment
            # For now, calculate based on expansion decisions
            total_cap_alive[sim, t] = c_initial_capacity[tech]  # Placeholder
        end
    end
    
    legend_pos = tech == :Boiler ? :topright : :topleft
    stages_display = [2020, 2030, 2040, 2050]
    
    fig = plot_bands(total_cap_alive[:, 2:2:end], "$tech Alive Capacities", 
                    1:4, (1:4, stages_display), "Year", "Capacity [MW_th]", legend_pos)
    savefig("AliveCapacities_$tech.png")
end

# 2. Investment Decisions Plot
println("Generating investment plots...")
for tech in technologies
    inv_stages = collect(1:2:(T*2))
    u_invs = zeros(length(simulations), length(inv_stages))
    
    for sim in 1:length(simulations)
        counter = 1
        for t in inv_stages
            u_inv = value(simulations[sim][t][:u_expansion_tech][tech])
            u_invs[sim, counter] = u_inv
            counter += 1
        end
    end
    
    legend_pos = tech == :Geothermal ? :topleft : :topright
    stages_display = [2020, 2030, 2040, 2050]
    
    fig = plot_bands(u_invs, "$tech Investment", 1:4, (1:4, stages_display), 
                    "Investment Year", "Investment [MW_th]", legend_pos)
    savefig("Investments_$tech.png")
end

# 3. Storage Investment Plot
println("Generating storage investment plot...")
stor_invs = zeros(length(simulations), 4)
for sim in 1:length(simulations)
    counter = 1
    for t in collect(1:2:(T*2))
        stor_invs[sim, counter] = value(simulations[sim][t][:u_expansion_storage])
        counter += 1
    end
end

stages_display = [2020, 2030, 2040, 2050]
fig = plot_bands(stor_invs, "Storage Investment", 1:4, (1:4, stages_display), 
                "Investment Year", "Investment [MWh]", :topleft)
savefig("Investments_Storage.png")

# 4. Load Duration Curve for Sample Simulation
println("Generating load duration curve...")
n_sim = 1  # Use first simulation

plt = plot(xlabel="Hours", ylabel="Heat Generation [MWh_th]", legend=:topright)

# Calculate total hours across all weeks for visualization
total_viz_hours = n_weeks * hours_per_week

# For each operational stage
for t_year in 1:T
    t = 2 * t_year  # Operational stages are even
    sp = simulations[n_sim][t]
    
    # Prepare data for stacked area plot
    y_data = zeros(total_viz_hours, length(technologies) + 1)  # +1 for storage
    
    for (tech_idx, tech) in enumerate(technologies)
        hour_idx = 1
        for week in 1:n_weeks
            for hour in 1:hours_per_week
                # Weight the production by week weight for annual representation
                y_data[hour_idx, tech_idx] = value(sp[:u_production][tech, week, hour])
                hour_idx += 1
            end
        end
    end
    
    # Add storage discharge
    hour_idx = 1
    for week in 1:n_weeks
        for hour in 1:hours_per_week
            y_data[hour_idx, length(technologies) + 1] = value(sp[:u_discharge][week, hour])
            hour_idx += 1
        end
    end
    
    # Sort for load duration curve
    for col in 1:size(y_data, 2)
        y_data[:, col] = sort(y_data[:, col], rev=true)
    end
    
    # Create subplot for this year
    year_label = 2010 + t_year * 10
    p = plot(title="Load Duration Curve - Year $year_label")
    labels = [String(tech) for tech in technologies]
    push!(labels, "Storage")
    
    # Stacked area plot
    areaplot!(1:total_viz_hours, y_data, label=reshape(labels, 1, :), 
             fillalpha=0.7, legend=:topright)
    xlabel!("Hours (sorted)")
    ylabel!("Heat Generation [MWh_th]")
    
    savefig("LoadDurationCurve_Year$(year_label).png")
end

# 5. Violin Plots - Production vs Demand
println("Generating violin plots...")

function plot_combined_violins(data1, data2, title_str, xlabels, y_legend, leg_pos)
    # Prepare data for violin plot
    group_labels = repeat(xlabels, inner=size(data1, 1))
    flattened_values_1 = vec(data1)
    flattened_values_2 = vec(data2)
    
    # Create violin plot
    p = violin(group_labels, flattened_values_1,
        xlabel="Years", title=title_str, label="Annual production",
        legend=leg_pos, alpha=0.5, c=:green, side=:right)
    
    violin!(group_labels, flattened_values_2,
        label="Annual demand",
        alpha=0.5, c=:red, side=:left)
    
    ylabel!(y_legend)
    return p
end

# Calculate annual production and demand for violin plots
xlabels_years = ["2020", "2030", "2040", "2050"]
ope_var_demand = zeros(length(simulations), T)

for (ope_stage, stage) in enumerate(2:2:2*T)
    for sim in 1:length(simulations)
        ope_var_demand[sim, ope_stage] = value(simulations[sim][stage][:x_demand_mult].out) * base_annual_demand
    end
end

# Create combined violin plot for each technology
for tech in technologies
    ope_var_prod = zeros(length(simulations), T)
    
    for (ope_stage, stage) in enumerate(2:2:2*T)
        for sim in 1:length(simulations)
            # Sum production across all weeks and hours, weighted by week occurrence
            total_prod = 0.0
            for week in 1:n_weeks
                week_prod = sum(value(simulations[sim][stage][:u_production][tech, week, hour]) 
                               for hour in 1:hours_per_week)
                total_prod += week_prod * week_weights_normalized[week]
            end
            ope_var_prod[sim, ope_stage] = total_prod
        end
    end
    
    p = plot_combined_violins(ope_var_prod, ope_var_demand, String(tech), 
                              xlabels_years, "Annual Energy [MWh]", :best)
    savefig("ViolinPlot_$(tech).png")
end

# 6. Storage Operation Violin Plot
println("Generating storage operation plot...")
ope_var_storage = zeros(length(simulations), T)

for (ope_stage, stage) in enumerate(2:2:2*T)
    for sim in 1:length(simulations)
        # Sum storage discharge across all weeks and hours, weighted
        total_discharge = 0.0
        for week in 1:n_weeks
            week_discharge = sum(value(simulations[sim][stage][:u_discharge][week, hour]) 
                               for hour in 1:hours_per_week)
            total_discharge += week_discharge * week_weights_normalized[week]
        end
        ope_var_storage[sim, ope_stage] = total_discharge
    end
end

p = violin(repeat(xlabels_years, inner=length(simulations)), vec(ope_var_storage),
          xlabel="Years", title="Storage Discharge", 
          ylabel="Annual Discharge [MWh]",
          legend=false, alpha=0.5, c=:blue)
savefig("ViolinPlot_Storage.png")

# 7. Spaghetti Plot using SDDP's built-in functionality
println("Generating spaghetti plots...")
plt = SDDP.SpaghettiPlot(simulations)

for tech in technologies
    SDDP.add_spaghetti(plt; title="Expansion_$tech") do data
        return data[:u_expansion_tech][tech]
    end
    
    SDDP.add_spaghetti(plt; title="Production_$tech") do data
        # Sum across all weeks and hours
        total = 0.0
        for week in 1:n_weeks
            total += sum(data[:u_production][tech, week, :]) * week_weights_normalized[week]
        end
        return total
    end
end

SDDP.add_spaghetti(plt; title="Storage_Expansion") do data
    return data[:u_expansion_storage]
end

SDDP.add_spaghetti(plt; title="Storage_Discharge") do data
    total = 0.0
    for week in 1:n_weeks
        total += sum(data[:u_discharge][week, :]) * week_weights_normalized[week]
    end
    return total
end

SDDP.add_spaghetti(plt; title="Demand_Multiplier") do data
    return data[:x_demand_mult].out
end

SDDP.add_spaghetti(plt; title="Unmet_Demand") do data
    total = 0.0
    for week in 1:n_weeks
        total += sum(data[:u_unmet][week, :]) * week_weights_normalized[week]
    end
    return total
end

SDDP.plot(plt, "spaghetti_plot.html")

# 8. Summary Statistics Table
println("\n" * "="^80)
println("SUMMARY STATISTICS")
println("="^80)

# Calculate key metrics
total_costs = [sum(simulations[s][t][:stage_objective] for t in 1:(T*2)) for s in 1:length(simulations)]
mean_cost = mean(total_costs)
std_cost = std(total_costs)
cvar_95_cost = quantile(total_costs, 0.95)

println("\nTotal System Cost (€ million):")
println("  Mean: $(round(mean_cost/1e6, digits=2))")
println("  Std Dev: $(round(std_cost/1e6, digits=2))")
println("  CVaR 95%: $(round(cvar_95_cost/1e6, digits=2))")

# Technology investments summary
println("\nAverage Technology Investments (MW):")
for tech in technologies
    avg_investment = 0.0
    for sim in 1:length(simulations)
        for t in collect(1:2:(T*2))
            avg_investment += value(simulations[sim][t][:u_expansion_tech][tech])
        end
    end
    avg_investment /= length(simulations)
    println("  $tech: $(round(avg_investment, digits=1))")
end

# Storage investment summary
avg_storage = 0.0
for sim in 1:length(simulations)
    for t in collect(1:2:(T*2))
        avg_storage += value(simulations[sim][t][:u_expansion_storage])
    end
end
avg_storage /= length(simulations)
println("  Storage: $(round(avg_storage, digits=1)) MWh")

# Unmet demand analysis
println("\nUnmet Demand Analysis:")
for year_idx in 1:T
    t = 2 * year_idx
    year = 2010 + year_idx * 10
    
    unmet_values = []
    for sim in 1:length(simulations)
        total_unmet = 0.0
        for week in 1:n_weeks
            week_unmet = sum(value(simulations[sim][t][:u_unmet][week, hour]) 
                           for hour in 1:hours_per_week)
            total_unmet += week_unmet * week_weights_normalized[week]
        end
        push!(unmet_values, total_unmet)
    end
    
    mean_unmet = mean(unmet_values)
    max_unmet = maximum(unmet_values)
    
    println("  Year $year:")
    println("    Mean: $(round(mean_unmet, digits=2)) MWh")
    println("    Max: $(round(max_unmet, digits=2)) MWh")
end

# Storage utilization
println("\nStorage Utilization (Year 2050):")
t = T * 2
storage_utilization = []
for sim in 1:length(simulations)
    total_discharge = 0.0
    for week in 1:n_weeks
        week_discharge = sum(value(simulations[sim][t][:u_discharge][week, hour]) 
                           for hour in 1:hours_per_week)
        total_discharge += week_discharge * week_weights_normalized[week]
    end
    
    # Get storage capacity (would need to track this properly)
    # For now, use a placeholder
    push!(storage_utilization, total_discharge)
end

if length(storage_utilization) > 0 && maximum(storage_utilization) > 0
    println("  Mean Annual Discharge: $(round(mean(storage_utilization), digits=1)) MWh")
    println("  Max Annual Discharge: $(round(maximum(storage_utilization), digits=1)) MWh")
end

println("\n" * "="^80)
println("Visualization complete! Check generated PNG and HTML files.")
println("="^80)