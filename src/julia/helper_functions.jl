"""
Helper functions for the District Heating Capacity Expansion model
"""

"""
    discount_factor(t::Int, T_years::Int, discount_rate::Float64)

Calculate the discount factor for a given stage.

# Arguments
- `t::Int`: Current stage number
- `T_years::Int`: Number of years represented by each model year
- `discount_rate::Float64`: Annual discount rate
"""
function discount_factor(year_start, year_evaluate, discount_rate::Float64)
    exponent = year_evaluate - year_start
    return 1.0 / (1.0 + discount_rate)^exponent
end

"""
    is_alive(s_invest::Int, s_current::Int, lifetime::Dict, tech::Symbol)

Check if a technology investment is still operational at the current stage.

# Arguments
- `s_invest::Int`: Investment stage
- `s_current::Int`: Current stage
- `lifetime::Dict`: Dictionary mapping technology symbols to lifetimes
- `tech::Symbol`: Technology symbol
"""
function is_alive(s_invest::Int, s_current::Int, lifetime::Dict, tech::Symbol)
    year_invest = ceil(s_invest / 2)
    year_current = ceil(s_current / 2)
    return (1 <= (year_current - year_invest)) && ((year_current - year_invest) <= lifetime[tech])
end

"""
    is_storage_alive(s_invest::Int, s_current::Int, storage_lifetime::Int)

Check if a storage investment is still operational at the current stage.

# Arguments
- `s_invest::Int`: Investment stage
- `s_current::Int`: Current stage
- `storage_lifetime::Int`: Storage lifetime in model years
"""
function is_storage_alive(s_invest::Int, s_current::Int, storage_lifetime::Int)
    year_invest = ceil(s_invest / 2)
    year_current = ceil(s_current / 2)
    return (1 <= (year_current - year_invest)) && ((year_current - year_invest) <= storage_lifetime)
end

# former build_dictionnaries
function get_encoder_decoder(policies_transition, prices_transition, temp_transition, rep_years)

    # state -> policy, price, temperature
    i = 0
    state2keys = Dict()
    keys2state = Dict()
    for temp in eachrow(temp_transition)
        nest = Dict()
        for policy in eachrow(policies_transition)
            nest2 = Dict()
            for price in eachrow(prices_transition)  # Fixed: use prices_transition
                nest2[price["price"]] = i
                i = i+1
                state2keys[i] = (policy["policy"], price["price"],temp["temp"])
            end
            nest[policy["policy"]] = copy(nest2)
        end
        keys2state[temp["temp"]] = copy(nest)
    end

    # stages -> year, investment
    stage2year_phase = Dict()
    year_phase2stage = Dict()
    for i in 1:length(rep_years)
        year = rep_years[i]
        stage2year_phase[2*i-1] = (year,"investment")
        stage2year_phase[2*i] = (year,"operations")

        year_phase2stage[year,"investment"] = 2*i-1
        year_phase2stage[year,"operations"] = 2*i
    end
    return sort(state2keys), sort(keys2state), sort(stage2year_phase), sort(year_phase2stage)
end