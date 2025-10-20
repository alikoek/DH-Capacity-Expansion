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

    # 1. Investment Decisions Plot
    println("Generating investment plots...")
    for tech in params.technologies
        inv_stages = collect(1:2:(params.T*2))
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
        savefig(joinpath(output_dir, "Investments_$tech.png"))
    end

    # 2. Storage Investment Plot
    println("Generating storage investment plot...")
    stor_invs = zeros(length(simulations), 4)
    for sim in 1:length(simulations)
        counter = 1
        for t in collect(1:2:(params.T*2))
            stor_invs[sim, counter] = value(simulations[sim][t][:u_expansion_storage])
            counter += 1
        end
    end

    stages_display = [2020, 2030, 2040, 2050]
    fig = plot_bands(stor_invs, "Storage Investment", 1:4, (1:4, stages_display),
                    "Investment Year", "Investment [MWh]", :topleft)
    savefig(joinpath(output_dir, "Investments_Storage.png"))

    # 3. Load Duration Curve for Sample Simulation
    println("Generating load duration curves...")
    n_sim = 1  # Use first simulation

    # Calculate total hours across all weeks for visualization
    total_viz_hours = data.n_weeks * data.hours_per_week

    # For each operational stage
    for t_year in 1:params.T
        t = 2 * t_year  # Operational stages are even
        sp = simulations[n_sim][t]

        # Prepare data for stacked area plot
        y_data = zeros(total_viz_hours, length(params.technologies) + 1)  # +1 for storage

        for (tech_idx, tech) in enumerate(params.technologies)
            hour_idx = 1
            for week in 1:data.n_weeks
                for hour in 1:data.hours_per_week
                    y_data[hour_idx, tech_idx] = value(sp[:u_production][tech, week, hour])
                    hour_idx += 1
                end
            end
        end

        # Add storage discharge
        hour_idx = 1
        for week in 1:data.n_weeks
            for hour in 1:data.hours_per_week
                y_data[hour_idx, length(params.technologies) + 1] = value(sp[:u_discharge][week, hour])
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
        labels = [String(tech) for tech in params.technologies]
        push!(labels, "Storage")

        # Stacked area plot
        areaplot!(1:total_viz_hours, y_data, label=reshape(labels, 1, :),
                 fillalpha=0.7, legend=:topright)
        xlabel!("Hours (sorted)")
        ylabel!("Heat Generation [MWh_th]")

        savefig(joinpath(output_dir, "LoadDurationCurve_Year$(year_label).png"))
    end

    # 4. Violin Plots - Production vs Demand
    println("Generating violin plots...")

    xlabels_years = ["2020", "2030", "2040", "2050"]
    ope_var_demand = zeros(length(simulations), params.T)

    for (ope_stage, stage) in enumerate(2:2:2*params.T)
        for sim in 1:length(simulations)
            ope_var_demand[sim, ope_stage] = value(simulations[sim][stage][:x_demand_mult].out) * params.base_annual_demand
        end
    end

    # Create combined violin plot for each technology
    for tech in params.technologies
        ope_var_prod = zeros(length(simulations), params.T)

        for (ope_stage, stage) in enumerate(2:2:2*params.T)
            for sim in 1:length(simulations)
                # Sum production across all weeks and hours, weighted by week occurrence
                total_prod = 0.0
                for week in 1:data.n_weeks
                    week_prod = sum(value(simulations[sim][stage][:u_production][tech, week, hour])
                                   for hour in 1:data.hours_per_week)
                    total_prod += week_prod * data.week_weights_normalized[week]
                end
                ope_var_prod[sim, ope_stage] = total_prod
            end
        end

        p = plot_combined_violins(ope_var_prod, ope_var_demand, String(tech),
                                  xlabels_years, "Annual Energy [MWh]", :best)
        savefig(joinpath(output_dir, "ViolinPlot_$(tech).png"))
    end

    # 5. Storage Operation Violin Plot
    println("Generating storage operation plot...")
    ope_var_storage = zeros(length(simulations), params.T)

    for (ope_stage, stage) in enumerate(2:2:2*params.T)
        for sim in 1:length(simulations)
            # Sum storage discharge across all weeks and hours, weighted
            total_discharge = 0.0
            for week in 1:data.n_weeks
                week_discharge = sum(value(simulations[sim][stage][:u_discharge][week, hour])
                                   for hour in 1:data.hours_per_week)
                total_discharge += week_discharge * data.week_weights_normalized[week]
            end
            ope_var_storage[sim, ope_stage] = total_discharge
        end
    end

    p = violin(repeat(xlabels_years, inner=length(simulations)), vec(ope_var_storage),
              xlabel="Years", title="Storage Discharge",
              ylabel="Annual Discharge [MWh]",
              legend=false, alpha=0.5, c=:blue)
    savefig(joinpath(output_dir, "ViolinPlot_Storage.png"))

    # 6. Spaghetti Plot using SDDP's built-in functionality
    println("Generating spaghetti plots...")
    plt = SDDP.SpaghettiPlot(simulations)

    for tech in params.technologies
        SDDP.add_spaghetti(plt; title="Expansion_$tech") do sp
            return sp[:u_expansion_tech][tech]
        end

        SDDP.add_spaghetti(plt; title="Production_$tech") do sp
            # Sum across all weeks and hours
            total = 0.0
            for week in 1:data.n_weeks
                total += sum(sp[:u_production][tech, week, :]) * data.week_weights_normalized[week]
            end
            return total
        end
    end

    SDDP.add_spaghetti(plt; title="Storage_Expansion") do sp
        return sp[:u_expansion_storage]
    end

    SDDP.add_spaghetti(plt; title="Storage_Discharge") do sp
        total = 0.0
        for week in 1:data.n_weeks
            total += sum(sp[:u_discharge][week, :]) * data.week_weights_normalized[week]
        end
        return total
    end

    SDDP.add_spaghetti(plt; title="Demand_Multiplier") do sp
        return sp[:x_demand_mult].out
    end

    SDDP.add_spaghetti(plt; title="Unmet_Demand") do sp
        total = 0.0
        for week in 1:data.n_weeks
            total += sum(sp[:u_unmet][week, :]) * data.week_weights_normalized[week]
        end
        return total
    end

    SDDP.plot(plt, joinpath(output_dir, "spaghetti_plot.html"))

    println("Visualization complete! Check generated plots in '$output_dir'")
end
