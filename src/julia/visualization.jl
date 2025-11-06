"""
Visualization and plotting functions for simulation results
"""

using Plots, StatsPlots, Statistics, SDDP

"""
    plot_bands(data, title, x_values, xticks_vals, xlabel_str, ylabel_str, legend_pos)

Create a band plot showing percentile ranges.

# Arguments
- `data`: Matrix where each column represents data for one x-value
- `title`: Plot title
- `x_values`: X-axis values
- `xticks_vals`: X-axis tick positions and labels
- `xlabel_str`: X-axis label
- `ylabel_str`: Y-axis label
- `legend_pos`: Legend position
"""
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

"""
    plot_combined_violins(data1, data2, title_str, xlabels, y_legend, leg_pos)

Create a combined violin plot comparing two datasets.

# Arguments
- `data1`: First dataset matrix
- `data2`: Second dataset matrix
- `title_str`: Plot title
- `xlabels`: X-axis labels
- `y_legend`: Y-axis label
- `leg_pos`: Legend position
"""
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


"""
    generate_visualizations(simulations, params::ModelParameters, data::ProcessedData; output_dir="output/")

Generate all visualizations from simulation results.

# Arguments
- `simulations`: Simulation results from SDDP.simulate
- `params::ModelParameters`: Model parameters structure
- `data::ProcessedData`: Processed data structure
- `output_dir::String`: Directory to save plots (default: "output/")
"""
function generate_visualizations(simulations, params::ModelParameters, data::ProcessedData; output_dir="output/")
    println("Generating visualizations...")

    state2keys, keys2state, stage2year_phase, year_phase2stage = build_dictionnaries(params.policy_proba_df, params.price_proba_df, data.rep_years)
    
    all_stages = keys(stage2year_phase)
    inv_stages = [state for (state, dept) in stage2year_phase if dept[2] == "investment"]
    inv_years = [dept[1] for (state, dept) in stage2year_phase if dept[2] == "investment"]
    n_inv = length(inv_stages)

    ope_stages = [state for (state, dept) in stage2year_phase if dept[2] == "operations"]

    # 1. Investment Decisions Plot
    println("Generating investment plots...")
    for tech in keys(params.tech_dict)
        u_invs = zeros(length(simulations), length(inv_stages))
        for sim in 1:length(simulations)
            counter = 1
            for t in inv_stages
                u_inv = value(sum(simulations[sim][t][:X_tech][tech,live].in for live in 1:params.tech_dict[tech]["lifetime_new"]))
                u_invs[sim, counter] = u_inv
                counter += 1
            end
        end

        legend_pos = tech == :Geothermal ? :topleft : :topright
        stages_display = inv_years

        fig = plot_bands(u_invs, "$tech Capacity", 1:n_inv, (1:n_inv, stages_display),
                        "Investment Year", "Investment [MW_th]", legend_pos)
        savefig(joinpath(output_dir, "Capacity_$tech.png"))
    end

    for tech in keys(params.stor_dict)
        u_invs = zeros(length(simulations), length(inv_stages))
        for sim in 1:length(simulations)
            counter = 1
            for t in inv_stages
                u_inv = value(sum(simulations[sim][t][:X_stor][tech,live].in for live in 1:params.stor_dict[tech]["lifetime_new"]))
                u_invs[sim, counter] = u_inv
                counter += 1
            end
        end
        legend_pos = :topright
        stages_display = inv_years

        fig = plot_bands(u_invs, "$tech Capacity", 1:n_inv, (1:n_inv, stages_display),
                        "Investment Year", "Investment [MWh_th]", legend_pos)
        savefig(joinpath(output_dir, "Capacity_$tech.png"))
    end

    sim = 1  # Use first simulation
    df_prod = DataFrame(
        year = Int[],
        type = String[],
        technology = String[],
        week = Int[],
        hour = Int[],
        value = Float64[],
    )
    for t in ope_stages
        state= simulations[sim][t][:node_index][2]
        policy, price = state2keys[state]
        year, phase = stage2year_phase[t]
        u_prod = simulations[sim][t][:u_production]
        x_demand_mult = simulations[sim][t][:x_demand_mult].in
        techs, weeks, hours = axes(u_prod)
        for tech in techs, week in weeks, hour in hours
            push!(df_prod, (year, "Production",String(tech), week, hour, u_prod[tech, week, hour]))
        end

        u_prod = simulations[sim][t][:u_charge]
        techs, _, _ = axes(u_prod)
        for tech in techs, week in weeks, hour in hours
            push!(df_prod, (year, "Charge", String(tech), week, hour, u_prod[tech, week, hour]))
        end

        u_prod = simulations[sim][t][:u_discharge]
        techs, _, _ = axes(u_prod)
        for tech in techs, week in weeks, hour in hours
            push!(df_prod, (year, "Discharge", String(tech), week, hour, u_prod[tech, week, hour]))
        end

        u_prod = simulations[sim][t][:u_unmet]
        for week in weeks, hour in hours
            push!(df_prod, (year, "Unmet", "/", week, hour, u_prod[week, hour]))
        end

        for week in weeks, hour in hours
            dem = x_demand_mult * first(data.weeks[(data.weeks[!, "typical_week"] .== week) .& (data.weeks[!, "hour"] .== hour)  .& (data.weeks[!, "year"] .== (year))  .& (data.weeks[!, "scenario_price"] .== price), "Load Profile"])
            push!(df_prod, (year, "Demand", "/", week, hour, dem))
        end
        
    end
    show(df_prod)
    CSV.write(joinpath(output_dir, "operational_sim1.csv"), df_prod)

    df_ope_overview = DataFrame(
        year = Int[],
        
        policy = String[],
        price = String[],
        prev_policy = String[],
        prev_price = String[],

        demand_mult = Float64[],
        prev_demand_mult = Float64[],
        
        simulation = Int[],
        technology = String[],
        
        volume = Float64[]
        # CO2 = Float64[],
        # cost = Float64[]
    )

    df_design_overview = DataFrame(
        year = Int[],
        
        policy = String[],
        price = String[],
        prev_policy = String[],
        prev_price = String[],

        demand_mult = Float64[],
        prev_demand_mult = Float64[],
        
        simulation = Int[],
        technology = String[],
        
        installed = Float64[]
        # CO2 = Float64[],
        # cost = Float64[]
    )

    for (index_sim,sim) in enumerate(simulations)
        prev_policy, prev_price = "/","/"
        prev_demand_mult = 1.0
        for t in ope_stages
            state = sim[t][:node_index][2]
            policy, price = state2keys[state]
            year, phase = stage2year_phase[t]
            X_tech = sim[t][:X_tech]
            x_demand_mult = sim[t][:x_demand_mult].in
            


            # techs, lives = axes(X_tech)
            for tech in keys(params.tech_dict)
                installed = sum(X_tech[tech,lives].in for lives in params.tech_dict[tech]["lifetime_new"])
                push!(df_design_overview, (year, 
                                policy,
                                price, 
                                prev_policy,
                                prev_price,
                                
                                x_demand_mult,
                                prev_demand_mult,

                                Int(index_sim),
                                params.tech_dict[tech]["full_name"],

                                installed,
                                ))
            end
            prev_policy, prev_price = policy, price
            prev_demand_mult = x_demand_mult

        end

    end
    show(df_design_overview)
    CSV.write(joinpath(output_dir, "design_overview.csv"), df_design_overview)


    for (index_sim,sim) in enumerate(simulations)
        prev_policy, prev_price = "/","/"
        prev_demand_mult = 1.0
        for t in ope_stages
            state = sim[t][:node_index][2]
            policy, price = state2keys[state]
            year, phase = stage2year_phase[t]
            u_prod = sim[t][:u_production]
            x_demand_mult = sim[t][:x_demand_mult].in
            


            techs, weeks, hours = axes(u_prod)
            for tech in techs
                prod = sum(u_prod[tech, week, hour] * 
                            first(data.week_weights[(data.week_weights[!,"year"] .== (year)) .& (data.week_weights[!,"scenario_price"] .== price), string(week)]) 
                            for week in weeks, hour in hours
                            )
                push!(df_ope_overview, (year, 
                                policy,
                                price, 
                                prev_policy,
                                prev_price,
                                
                                x_demand_mult,
                                prev_demand_mult,

                                Int(index_sim),
                                params.tech_dict[tech]["full_name"],

                                prod,
                                ))
            end
    
            u_prod = sim[t][:u_unmet]
            prod = sum(u_prod[week, hour] * 
                first(data.week_weights[(data.week_weights[!,"year"] .== (year)) .& (data.week_weights[!,"scenario_price"] .== price), string(week)]) 
                for week in weeks, hour in hours
                )
            push!(df_ope_overview, (year, 
                            policy,
                            price, 
                            prev_policy,
                            prev_price,
                            
                            x_demand_mult,
                            prev_demand_mult,

                            index_sim,
                            "unmet",

                            prod,
                            ))

            prev_policy, prev_price = policy, price
            prev_demand_mult = x_demand_mult

        end

    end
    show(df_ope_overview)
    CSV.write(joinpath(output_dir, "operational_overview.csv"), df_ope_overview)


            # u_prod = simulations[sim][t][:u_charge]
            # techs, _, _ = axes(u_prod)
            # for tech in techs, week in weeks, hour in hours
            #     push!(df_prod, (year, "Charge", String(tech), week, hour, u_prod[tech, week, hour]))
            # end
    
            # u_prod = simulations[sim][t][:u_discharge]
            # techs, _, _ = axes(u_prod)
            # for tech in techs, week in weeks, hour in hours
            #     push!(df_prod, (year, "Discharge", String(tech), week, hour, u_prod[tech, week, hour]))
            # end
    
            # u_prod = simulations[sim][t][:u_unmet]
            # for week in weeks, hour in hours
            #     push!(df_prod, (year, "Unmet", "/", week, hour, u_prod[week, hour]))
            # end
    
            
    # Add the 
end
    # # 3. Load Duration Curve for Sample Simulation
    # println("Generating load duration curves...")

    # # Calculate total hours across all weeks for visualization
    # total_viz_hours = data.n_weeks * data.hours_per_week

    # # For each operational stage
    # for t_year in 1:params.T
    #     t = 2 * t_year  # Operational stages are even
    #     sp = simulations[n_sim][t]

    #     # Prepare data for stacked area plot
    #     y_data = zeros(total_viz_hours, length(params.technologies) + 1)  # +1 for storage

    #     for (tech_idx, tech) in enumerate(params.technologies)
    #         hour_idx = 1
    #         for week in 1:data.n_weeks
    #             for hour in 1:data.hours_per_week
    #                 y_data[hour_idx, tech_idx] = value(sp[:u_production][tech, week, hour])
    #                 hour_idx += 1
    #             end
    #         end
    #     end

    #     # Add storage discharge
    #     hour_idx = 1
    #     for week in 1:data.n_weeks
    #         for hour in 1:data.hours_per_week
    #             y_data[hour_idx, length(params.technologies) + 1] = value(sp[:u_discharge][week, hour])
    #             hour_idx += 1
    #         end
    #     end

    #     # Sort for load duration curve
    #     for col in 1:size(y_data, 2)
    #         y_data[:, col] = sort(y_data[:, col], rev=true)
    #     end

    #     # Create subplot for this year
    #     year_label = 2010 + t_year * 10
    #     p = plot(title="Load Duration Curve - Year $year_label")
    #     labels = [String(tech) for tech in params.technologies]
    #     push!(labels, "Storage")

    #     # Stacked area plot
    #     areaplot!(1:total_viz_hours, y_data, label=reshape(labels, 1, :),
    #              fillalpha=0.7, legend=:topright)
    #     xlabel!("Hours (sorted)")
    #     ylabel!("Heat Generation [MWh_th]")

    #     savefig(joinpath(output_dir, "LoadDurationCurve_Year$(year_label).png"))
    # end

    # # 4. Violin Plots - Production vs Demand
    # println("Generating violin plots...")

    # xlabels_years = ["2020", "2030", "2040", "2050"]
    # ope_var_demand = zeros(length(simulations), params.T)

    # for (ope_stage, stage) in enumerate(2:2:2*params.T)
    #     for sim in 1:length(simulations)
    #         ope_var_demand[sim, ope_stage] = value(simulations[sim][stage][:x_demand_mult].out) * params.base_annual_demand
    #     end
    # end

    # # Create combined violin plot for each technology
    # for tech in params.technologies
    #     ope_var_prod = zeros(length(simulations), params.T)

    #     for (ope_stage, stage) in enumerate(2:2:2*params.T)
    #         for sim in 1:length(simulations)
    #             # Sum production across all weeks and hours, weighted by week occurrence
    #             total_prod = 0.0
    #             for week in 1:data.n_weeks
    #                 week_prod = sum(value(simulations[sim][stage][:u_production][tech, week, hour])
    #                                for hour in 1:data.hours_per_week)
    #                 total_prod += week_prod * data.week_weights_normalized[week]
    #             end
    #             ope_var_prod[sim, ope_stage] = total_prod
    #         end
    #     end

    #     p = plot_combined_violins(ope_var_prod, ope_var_demand, String(tech),
    #                               xlabels_years, "Annual Energy [MWh]", :best)
    #     savefig(joinpath(output_dir, "ViolinPlot_$(tech).png"))
    # end

    # # 5. Storage Operation Violin Plot
    # println("Generating storage operation plot...")
    # ope_var_storage = zeros(length(simulations), params.T)

    # for (ope_stage, stage) in enumerate(2:2:2*params.T)
    #     for sim in 1:length(simulations)
    #         # Sum storage discharge across all weeks and hours, weighted
    #         total_discharge = 0.0
    #         for week in 1:data.n_weeks
    #             week_discharge = sum(value(simulations[sim][stage][:u_discharge][week, hour])
    #                                for hour in 1:data.hours_per_week)
    #             total_discharge += week_discharge * data.week_weights_normalized[week]
    #         end
    #         ope_var_storage[sim, ope_stage] = total_discharge
    #     end
    # end

    # p = violin(repeat(xlabels_years, inner=length(simulations)), vec(ope_var_storage),
    #           xlabel="Years", title="Storage Discharge",
    #           ylabel="Annual Discharge [MWh]",
    #           legend=false, alpha=0.5, c=:blue)
    # savefig(joinpath(output_dir, "ViolinPlot_Storage.png"))

    # # 6. Spaghetti Plot using SDDP's built-in functionality
    # println("Generating spaghetti plots...")
    # plt = SDDP.SpaghettiPlot(simulations)

    # for tech in params.technologies
    #     SDDP.add_spaghetti(plt; title="Expansion_$tech") do data2
    #         return data2[:u_expansion_tech][tech]
    #     end
    #     println(data)
    #     SDDP.add_spaghetti(plt; title="Production_$tech") do data2
    #         # Sum across all weeks and hours
    #         total = 0.0
    #         for week in 1:data.n_weeks
    #             total += sum(data2[:u_production][tech, week, :])  * data.week_weights_normalized[week]
    #         end
    #         return total
    #     end
    # end

    # SDDP.add_spaghetti(plt; title="Storage_Expansion") do data2
    #     return data2[:u_expansion_storage]
    # end

    # SDDP.add_spaghetti(plt; title="Storage_Discharge") do data2
    #     total = 0.0
    #     for week in 1:data.n_weeks
    #         total += sum(data2[:u_discharge][week, :]) * data.week_weights_normalized[week]
    #     end
    #     return total
    # end

    # SDDP.add_spaghetti(plt; title="Demand_Multiplier") do data2
    #     return data2[:x_demand_mult].out
    # end

    # SDDP.add_spaghetti(plt; title="Unmet_Demand") do data2
    #     total = 0.0
    #     for week in 1:data.n_weeks
    #         total += sum(data2[:u_unmet][week, :]) * data.week_weights_normalized[week]
    #     end
    #     return total
    # end

    # SDDP.plot(plt, joinpath(output_dir, "spaghetti_plot.html"))

    # println("Visualization complete! Check generated plots in '$output_dir'")
# end
