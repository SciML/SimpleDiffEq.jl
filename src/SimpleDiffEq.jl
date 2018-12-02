__precompile__()

module SimpleDiffEq

using Reexport
@reexport using DiffEqBase
using StaticArrays

include("functionmap.jl")
include("euler_maruyama.jl")
include("tsit5.jl")

end # module
