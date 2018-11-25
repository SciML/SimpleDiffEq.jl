__precompile__()

module SimpleDiffEq

using Reexport
@reexport using DiffEqBase
using StaticArrays

include("functionmap.jl")
include("euler_maruyama.jl")
include("simple_tsit5_integrator/tsit5_integrator.jl")
end # module
