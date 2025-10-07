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
function discount_factor(t::Int, T_years::Int, discount_rate::Float64)
    exponent = T_years * (ceil(t / 2) - 1)
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
