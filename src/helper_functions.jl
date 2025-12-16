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

"""
    decode_markov_state(t::Int, markov_state::Int; late_temp_branching::Bool=false)

Decode node index into (energy_state, temp_scenario).

# Arguments
- `t::Int`: Stage number
- `markov_state::Int`: Markov state index from policy graph
- `late_temp_branching::Bool`: If true, use late branching structure (temp at stage 2→3)

# Returns
- `(energy_state, temp_scenario)`: Tuple of energy state (1-3) and temperature scenario (1-2)

# Details
## Early branching (default, late_temp_branching=false):
- Stage 1: temperature branching only (2 nodes)
- Stages 2+: 6 states (energy × temp)

## Late branching (late_temp_branching=true):
- Stage 1: single node (no branching yet)
- Stage 2: energy branching only (3 nodes)
- Stages 3+: 6 states (energy × temp)

Node ordering for 6-state stages: (e1,t1), (e2,t1), (e3,t1), (e1,t2), (e2,t2), (e3,t2).
"""
function decode_markov_state(t::Int, markov_state::Int; late_temp_branching::Bool=false)
    if late_temp_branching
        # Late branching: temp revealed at stage 2→3
        if t == 1
            # Stage 1: single investment node, no scenario info
            # Return Medium energy (2) as default, temp=1 as placeholder
            return 2, 1
        elseif t == 2
            # Stage 2: 3 energy states only, no temp info yet
            # markov_state ∈ {1, 2, 3} = energy state
            # temp=1 as placeholder (Year 1 effects are identical for both temps)
            return markov_state, 1
        else
            # Stages 3+: 6 states representing (energy, temp) combinations
            temp_scenario = div(markov_state - 1, 3) + 1
            energy_state = mod(markov_state - 1, 3) + 1
            return energy_state, temp_scenario
        end
    else
        # Early branching (current default): temp revealed at root→stage 1
        if t == 1
            # Stage 1: System temperature branching only (2 nodes)
            return 1, markov_state  # energy_state=1 (default), temp_scenario∈{1,2}
        else
            # Stages 2+: 6 states representing (energy, temp) combinations
            temp_scenario = div(markov_state - 1, 3) + 1
            energy_state = mod(markov_state - 1, 3) + 1
            return energy_state, temp_scenario
        end
    end
end
