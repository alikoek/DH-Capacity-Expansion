# Fix environment
using Pkg
Pkg.activate(".")
Pkg.add(["SDDP", "Gurobi", "JuMP"])
Pkg.instantiate()
println("Environment fixed!")