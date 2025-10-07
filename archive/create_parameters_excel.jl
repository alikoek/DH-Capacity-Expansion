using XLSX

# Create Excel file with all parameters
filepath = joinpath(@__DIR__, "data", "model_parameters.xlsx")

XLSX.openxlsx(filepath, mode="w") do xf

    # Sheet 1: ModelConfig
    sheet = xf[1]
    XLSX.rename!(sheet, "ModelConfig")
    sheet["A1"] = "parameter"
    sheet["B1"] = "value"
    sheet["A2"] = "T"
    sheet["B2"] = 4
    sheet["A3"] = "T_years"
    sheet["B3"] = 10
    sheet["A4"] = "discount_rate"
    sheet["B4"] = 0.05
    sheet["A5"] = "base_annual_demand"
    sheet["B5"] = 2000000
    sheet["A6"] = "salvage_fraction"
    sheet["B6"] = 1
    sheet["A7"] = "c_penalty"
    sheet["B7"] = 100000
    sheet["A8"] = "num_price_scenarios"
    sheet["B8"] = 4
    sheet["A9"] = "mean_price"
    sheet["B9"] = 37.0
    sheet["A10"] = "price_volatility"
    sheet["B10"] = 10.0

    # Sheet 2: Technologies
    XLSX.addsheet!(xf, "Technologies")
    tech_sheet = xf["Technologies"]
    tech_sheet["A1"] = "technology"
    tech_sheet["B1"] = "initial_capacity"
    tech_sheet["C1"] = "max_additional_capacity"
    tech_sheet["D1"] = "investment_cost"
    tech_sheet["E1"] = "fixed_om"
    tech_sheet["F1"] = "variable_om"
    tech_sheet["G1"] = "efficiency_th"
    tech_sheet["H1"] = "efficiency_el"
    tech_sheet["I1"] = "energy_carrier"
    tech_sheet["J1"] = "lifetime_new"
    tech_sheet["K1"] = "lifetime_initial"

    # CHP
    tech_sheet["A2"] = "CHP"
    tech_sheet["B2"] = 500
    tech_sheet["C2"] = 500
    tech_sheet["D2"] = 1100000
    tech_sheet["E2"] = 15000
    tech_sheet["F2"] = 3.3
    tech_sheet["G2"] = 0.44
    tech_sheet["H2"] = 0.43
    tech_sheet["I2"] = "nat_gas"
    tech_sheet["J2"] = 3
    tech_sheet["K2"] = 2

    # Boiler
    tech_sheet["A3"] = "Boiler"
    tech_sheet["B3"] = 350
    tech_sheet["C3"] = 500
    tech_sheet["D3"] = 100000
    tech_sheet["E3"] = 2000
    tech_sheet["F3"] = 0.75
    tech_sheet["G3"] = 0.9
    tech_sheet["H3"] = 0.0
    tech_sheet["I3"] = "nat_gas"
    tech_sheet["J3"] = 3
    tech_sheet["K3"] = 1

    # HeatPump
    tech_sheet["A4"] = "HeatPump"
    tech_sheet["B4"] = 0
    tech_sheet["C4"] = 250
    tech_sheet["D4"] = 591052
    tech_sheet["E4"] = 3000
    tech_sheet["F4"] = 1.1
    tech_sheet["G4"] = 3.0
    tech_sheet["H4"] = 0.0
    tech_sheet["I4"] = "elec"
    tech_sheet["J4"] = 2
    tech_sheet["K4"] = 2

    # Geothermal
    tech_sheet["A5"] = "Geothermal"
    tech_sheet["B5"] = 0
    tech_sheet["C5"] = 100
    tech_sheet["D5"] = 1000000
    tech_sheet["E5"] = 2000
    tech_sheet["F5"] = 0.5
    tech_sheet["G5"] = 1.0
    tech_sheet["H5"] = 0.0
    tech_sheet["I5"] = "geothermal"
    tech_sheet["J5"] = 3
    tech_sheet["K5"] = 0

    # Sheet 3: Storage
    XLSX.addsheet!(xf, "Storage")
    stor_sheet = xf["Storage"]
    stor_sheet["A1"] = "parameter"
    stor_sheet["B1"] = "value"
    stor_sheet["A2"] = "capacity_cost"
    stor_sheet["B2"] = 25000
    stor_sheet["A3"] = "fixed_om"
    stor_sheet["B3"] = 500
    stor_sheet["A4"] = "variable_om"
    stor_sheet["B4"] = 0.1
    stor_sheet["A5"] = "efficiency"
    stor_sheet["B5"] = 0.95
    stor_sheet["A6"] = "loss_rate"
    stor_sheet["B6"] = 0.02
    stor_sheet["A7"] = "max_charge_rate"
    stor_sheet["B7"] = 0.25
    stor_sheet["A8"] = "max_discharge_rate"
    stor_sheet["B8"] = 0.25
    stor_sheet["A9"] = "lifetime"
    stor_sheet["B9"] = 4
    stor_sheet["A10"] = "max_capacity"
    stor_sheet["B10"] = 1000
    stor_sheet["A11"] = "initial_capacity"
    stor_sheet["B11"] = 0

    # Sheet 4: EnergyCarriers
    XLSX.addsheet!(xf, "EnergyCarriers")
    carrier_sheet = xf["EnergyCarriers"]
    carrier_sheet["A1"] = "carrier"
    carrier_sheet["B1"] = "emission_factor"
    carrier_sheet["A2"] = "nat_gas"
    carrier_sheet["B2"] = 0.2
    carrier_sheet["A3"] = "elec"
    carrier_sheet["B3"] = 0.0
    carrier_sheet["A4"] = "geothermal"
    carrier_sheet["B4"] = 0.0

    # Sheet 5: CarbonPrice
    XLSX.addsheet!(xf, "CarbonPrice")
    carbon_sheet = xf["CarbonPrice"]
    carbon_sheet["A1"] = "year"
    carbon_sheet["B1"] = "carbon_price"
    carbon_sheet["A2"] = 1
    carbon_sheet["B2"] = 50
    carbon_sheet["A3"] = 2
    carbon_sheet["B3"] = 150
    carbon_sheet["A4"] = 3
    carbon_sheet["B4"] = 250
    carbon_sheet["A5"] = 4
    carbon_sheet["B5"] = 350

    # Sheet 6: DemandMultipliers
    XLSX.addsheet!(xf, "DemandMultipliers")
    demand_sheet = xf["DemandMultipliers"]
    demand_sheet["A1"] = "state"
    demand_sheet["B1"] = "multiplier"
    demand_sheet["A2"] = 1
    demand_sheet["B2"] = 1.1
    demand_sheet["A3"] = 2
    demand_sheet["B3"] = 1.0
    demand_sheet["A4"] = 3
    demand_sheet["B4"] = 0.9
end

println("Excel file created successfully at: $filepath")
