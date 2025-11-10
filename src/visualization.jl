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
    plot_bands_separate(data, title, x_values, xticks_vals, xlabel_str, ylabel_str, legend_pos)

Create a band plot with TWO SEPARATE VISIBLE BANDS showing percentile ranges.
This is the old visualization style where both bands are drawn explicitly.

# Arguments
- `data`: Matrix where each column represents data for one x-value
- `title`: Plot title
- `x_values`: X-axis values
- `xticks_vals`: X-axis tick positions and labels
- `xlabel_str`: X-axis label
- `ylabel_str`: Y-axis label
- `legend_pos`: Legend position
"""
function plot_bands_separate(data, title, x_values, xticks_vals, xlabel_str, ylabel_str, legend_pos)
    # Compute statistics for each column
    q5 = [quantile(data[:, i], 0.05) for i in 1:size(data, 2)]
    q25 = [quantile(data[:, i], 0.25) for i in 1:size(data, 2)]
    q50 = [quantile(data[:, i], 0.50) for i in 1:size(data, 2)]
    q75 = [quantile(data[:, i], 0.75) for i in 1:size(data, 2)]
    q95 = [quantile(data[:, i], 0.95) for i in 1:size(data, 2)]

    # Create plot with TWO SEPARATE BANDS
    # First band: 5-95 percentile (bottom to top)
    p = plot(x_values, q5, fillrange=q95, fillalpha=0.15, c=2, label="5-95 percentile",
             legend=legend_pos, lw=0, xticks=xticks_vals)
    # Second band: 25-75 percentile (bottom to top)
    plot!(x_values, q25, fillrange=q75, fillalpha=0.35, c=2, label="25-75 percentile", lw=0)
    # Draw the outer band edges explicitly (this creates the separate visible bands)
    plot!(x_values, q95, fillalpha=0.0, c=2, label="", lw=0)
    plot!(x_values, q75, fillalpha=0.0, c=2, label="", lw=0)
    # Median line
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

    # 1. Investment Decisions Plot (Bar Charts with Error Bars)
    println("Generating investment plots...")
    years = [2023, 2030, 2040, 2050]
    inv_stages = collect(1:2:(params.T*2))

    # Collect investment data for all technologies
    all_inv_data = Dict{Symbol, Matrix{Float64}}()

    for tech in params.technologies
        u_invs = zeros(length(simulations), length(inv_stages))
        for sim in 1:length(simulations)
            for (idx, t) in enumerate(inv_stages)
                u_invs[sim, idx] = value(simulations[sim][t][:u_expansion_tech][tech])
            end
        end
        all_inv_data[tech] = u_invs

        # Create individual bar chart for each technology
        medians = [median(u_invs[:, i]) for i in 1:params.T]
        q25 = [quantile(u_invs[:, i], 0.25) for i in 1:params.T]
        q75 = [quantile(u_invs[:, i], 0.75) for i in 1:params.T]

        err_low = medians .- q25
        err_high = q75 .- medians

        p = bar(1:params.T, medians,
               xlabel="Year",
               ylabel="Investment [MW]",
               title="$tech - Investment Decisions",
               xticks=(1:params.T, years),
               legend=false,
               color=:steelblue,
               alpha=0.8,
               grid=true,
               size=(800, 500),
               yerror=(err_low, err_high),
               linecolor=:black,
               bar_width=0.6)

        savefig(p, joinpath(output_dir, "Investments_$tech.png"))
    end

    # Create multi-panel figure for key technologies
    println("Generating multi-panel investment figure...")
    key_techs = [:Waste_CHP, :Electric_Boiler, :SeaWater_HeatPump,
                 :WoodChip_CHP, :BioOil_Boiler, :BioPellet_CHP]
    # Filter to only existing techs
    key_techs = filter(t -> t in params.technologies, key_techs)

    n_techs = min(length(key_techs), 6)
    layout = n_techs <= 4 ? (2, 2) : (2, 3)

    p_multi = plot(layout=layout, size=(1200, 800), plot_title="Investment Decisions - Key Technologies")

    for (idx, tech) in enumerate(key_techs[1:n_techs])
        u_invs = all_inv_data[tech]
        medians = [median(u_invs[:, i]) for i in 1:params.T]
        q25 = [quantile(u_invs[:, i], 0.25) for i in 1:params.T]
        q75 = [quantile(u_invs[:, i], 0.75) for i in 1:params.T]

        err_low = medians .- q25
        err_high = q75 .- medians

        bar!(p_multi, 1:params.T, medians,
            subplot=idx,
            title=String(tech),
            xlabel=idx > n_techs-2 ? "Year" : "",
            ylabel=idx % layout[2] == 1 ? "Investment [MW]" : "",
            xticks=(1:params.T, years),
            legend=false,
            color=:steelblue,
            alpha=0.8,
            grid=true,
            yerror=(err_low, err_high),
            linecolor=:black,
            bar_width=0.6)
    end

    savefig(p_multi, joinpath(output_dir, "Investments_MultiPanel.png"))

    # 2. Storage Investment Plot (Bar Chart)
    println("Generating storage investment plot...")
    stor_invs = zeros(length(simulations), params.T)
    for sim in 1:length(simulations)
        for (idx, t) in enumerate(inv_stages)
            stor_invs[sim, idx] = value(simulations[sim][t][:u_expansion_storage])
        end
    end

    stor_medians = [median(stor_invs[:, i]) for i in 1:params.T]
    stor_q25 = [quantile(stor_invs[:, i], 0.25) for i in 1:params.T]
    stor_q75 = [quantile(stor_invs[:, i], 0.75) for i in 1:params.T]

    stor_err_low = stor_medians .- stor_q25
    stor_err_high = stor_q75 .- stor_medians

    p_stor = bar(1:params.T, stor_medians,
                xlabel="Year",
                ylabel="Investment [MWh]",
                title="Storage Investment Decisions",
                xticks=(1:params.T, years),
                legend=false,
                color=:darkorange,
                alpha=0.8,
                grid=true,
                size=(800, 500),
                yerror=(stor_err_low, stor_err_high),
                linecolor=:black,
                bar_width=0.6)

    savefig(p_stor, joinpath(output_dir, "Investments_Storage.png"))

    # 2b. Generate Band-Style Investment Plots (separate bands visualization)
    println("Generating band-style investment plots...")
    generate_investment_plots_separate_bands(simulations, params, data; output_dir=output_dir)

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
            # Deterministic demand: always use base_annual_demand
            ope_var_demand[sim, ope_stage] = params.base_annual_demand
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

    # Note: Demand is deterministic (no x_demand_mult variable)
    # Removed Demand_Multiplier spaghetti plot

    SDDP.add_spaghetti(plt; title="Unmet_Demand") do sp
        total = 0.0
        for week in 1:data.n_weeks
            total += sum(sp[:u_unmet][week, :]) * data.week_weights_normalized[week]
        end
        return total
    end

    SDDP.plot(plt, joinpath(output_dir, "spaghetti_plot.html"))

    # 7. Capacity Evolution with Vintage Tracking
    println("Generating capacity evolution plots...")
    plot_capacity_evolution_by_technology(simulations, params, data, output_dir)

    println("Visualization complete! Check generated plots in '$output_dir'")
end

"""
    generate_investment_plots_separate_bands(simulations, params::ModelParameters, data::ProcessedData; output_dir="output/")

Generate investment plots using the OLD two-band visualization style.
This creates separate plots for each technology showing investment decisions with two distinct bands.

# Arguments
- `simulations`: Simulation results from SDDP.simulate
- `params::ModelParameters`: Model parameters structure
- `data::ProcessedData`: Processed data structure
- `output_dir::String`: Directory to save plots (default: "output/")
"""
function generate_investment_plots_separate_bands(simulations, params::ModelParameters, data::ProcessedData; output_dir="output/")
    println("Generating investment plots with separate bands...")

    inv_stages = collect(1:2:(params.T*2))
    stages_display = [2020, 2030, 2040, 2050]

    # Technology investment plots
    for tech in params.technologies
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

        fig = plot_bands_separate(u_invs, "$tech Investment", 1:4, (1:4, stages_display),
                        "Investment Year", "Investment [MW_th]", legend_pos)
        savefig(joinpath(output_dir, "Investments_SeparateBands_$tech.png"))
    end

    # Storage investment plot
    println("Generating storage investment plot with separate bands...")
    stor_invs = zeros(length(simulations), length(inv_stages))
    for sim in 1:length(simulations)
        counter = 1
        for t in inv_stages
            stor_invs[sim, counter] = value(simulations[sim][t][:u_expansion_storage])
            counter += 1
        end
    end

    fig = plot_bands_separate(stor_invs, "Storage Investment", 1:4, (1:4, stages_display),
                    "Investment Year", "Investment [MWh]", :topleft)
    savefig(joinpath(output_dir, "Investments_SeparateBands_Storage.png"))

    println("Separate-band investment plots complete!")
end

"""
    plot_capacity_evolution_by_technology(simulations, params::ModelParameters, data::ProcessedData, output_dir::String)

Create capacity evolution plots using bar charts with error bars for each technology.

Creates two versions:
1. Total capacity (cumulative): Shows total available capacity with uncertainty
2. Vintage breakdown: Shows individual vintage contributions

# Arguments
- `simulations`: Simulation results from SDDP.simulate
- `params::ModelParameters`: Model parameters structure
- `data::ProcessedData`: Processed data structure
- `output_dir::String`: Directory to save plots
"""
function plot_capacity_evolution_by_technology(simulations, params::ModelParameters, data::ProcessedData, output_dir::String)
    n_sims = length(simulations)
    years = [2023, 2030, 2040, 2050]  # Display years
    model_years = 1:params.T
    investment_stages = params.investment_stages  # [1, 3, 5, 7]

    # Define colors
    color_existing = :gray
    color_new = :steelblue
    vintage_colors = Dict(
        0 => :gray,
        1 => :steelblue,
        3 => :seagreen,
        5 => :darkorange,
        7 => :firebrick
    )

    for tech in params.technologies
        println("  Processing $tech...")

        # Collect data across all simulations
        capacity_data = Dict{Int, Dict{Int, Vector{Float64}}}()

        for year in model_years
            capacity_data[year] = Dict{Int, Vector{Float64}}()
            capacity_data[year][0] = Float64[]  # Existing
            for stage in investment_stages
                capacity_data[year][stage] = Float64[]
            end
        end

        for sim in 1:n_sims
            for year in model_years
                operational_stage = 2 * year

                # Existing capacity
                if haskey(params.c_existing_capacity_schedule, tech)
                    existing_cap = params.c_existing_capacity_schedule[tech][year]
                else
                    existing_cap = 0.0
                end
                push!(capacity_data[year][0], existing_cap)

                # NEW investment vintages
                for stage in investment_stages
                    if stage <= length(simulations[sim])
                        inv_amount = value(simulations[sim][stage][:u_expansion_tech][tech])
                        if is_alive(stage, operational_stage, params.c_lifetime_new, tech)
                            push!(capacity_data[year][stage], inv_amount)
                        else
                            push!(capacity_data[year][stage], 0.0)
                        end
                    else
                        push!(capacity_data[year][stage], 0.0)
                    end
                end
            end
        end

        # Calculate statistics
        stats = Dict{Int, Dict{Symbol, Dict{Int, Float64}}}()
        for year in model_years
            stats[year] = Dict(:median => Dict{Int, Float64}(),
                             :q25 => Dict{Int, Float64}(),
                             :q75 => Dict{Int, Float64}())
            for vintage in [0; investment_stages]
                data_vec = capacity_data[year][vintage]
                stats[year][:median][vintage] = median(data_vec)
                stats[year][:q25][vintage] = quantile(data_vec, 0.25)
                stats[year][:q75][vintage] = quantile(data_vec, 0.75)
            end
        end

        # === PLOT 1: TOTAL CAPACITY (Cumulative) ===
        existing_caps = [stats[year][:median][0] for year in model_years]
        new_median = [sum(stats[year][:median][s] for s in investment_stages) for year in model_years]
        new_q25 = [sum(stats[year][:q25][s] for s in investment_stages) for year in model_years]
        new_q75 = [sum(stats[year][:q75][s] for s in investment_stages) for year in model_years]

        total_median = existing_caps .+ new_median

        # Create stacked bar chart manually
        p1 = bar(collect(model_years), existing_caps,
                bar_position=:stack,
                xlabel="Year",
                ylabel="Capacity [MW]",
                title="$tech - Total Capacity Evolution",
                label="Existing",
                color=color_existing,
                xticks=(model_years, years),
                legend=:best,
                size=(800, 600),
                grid=true,
                alpha=0.8)

        bar!(p1, collect(model_years), new_median,
            bar_position=:stack,
            label="New Investments",
            color=color_new,
            alpha=0.8)

        # Add error bars for NEW capacity only
        for (i, year) in enumerate(model_years)
            err_low = new_median[i] - new_q25[i]
            err_high = new_q75[i] - new_median[i]
            plot!(p1, [year], [existing_caps[i] + new_median[i]],
                 yerror=([err_low], [err_high]),
                 color=:black, linewidth=1.5, label="",
                 markershape=:none)
        end

        savefig(p1, joinpath(output_dir, "CapacityEvolution_Total_$(tech).png"))

        # === PLOT 2: VINTAGE BREAKDOWN ===
        p2 = plot(
            xlabel="Year",
            ylabel="Capacity [MW]",
            title="$tech - Capacity by Vintage",
            xticks=(model_years, years),
            legend=:best,
            size=(900, 600),
            grid=true
        )

        vintage_labels = Dict(
            0 => "Existing",
            1 => "2023 Inv",
            3 => "2030 Inv",
            5 => "2040 Inv",
            7 => "2050 Inv"
        )

        bar_width = 0.15
        vintages_to_show = [0; investment_stages]

        for (v_idx, vintage) in enumerate(vintages_to_show)
            vintage_median = [stats[year][:median][vintage] for year in model_years]
            vintage_q25 = [stats[year][:q25][vintage] for year in model_years]
            vintage_q75 = [stats[year][:q75][vintage] for year in model_years]

            # Skip if zero everywhere
            if maximum(vintage_median) < 0.001
                continue
            end

            # Offset bars for grouping
            x_offset = (v_idx - (length(vintages_to_show)+1)/2) * bar_width
            x_positions = model_years .+ x_offset

            bar!(p2, x_positions, vintage_median,
                bar_width=bar_width,
                color=vintage_colors[vintage],
                label=vintage_labels[vintage],
                alpha=0.8)

            # Error bars for NEW vintages
            if vintage != 0
                for (i, year) in enumerate(model_years)
                    if vintage_median[i] > 0.001
                        err_low = vintage_median[i] - vintage_q25[i]
                        err_high = vintage_q75[i] - vintage_median[i]
                        plot!(p2, [x_positions[i]], [vintage_median[i]],
                             yerror=([err_low], [err_high]),
                             color=:black, linewidth=1, label="",
                             markershape=:none)
                    end
                end
            end
        end

        savefig(p2, joinpath(output_dir, "CapacityEvolution_Vintage_$(tech).png"))
    end

    println("  Capacity evolution plots complete!")
end
