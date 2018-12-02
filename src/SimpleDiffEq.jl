__precompile__()

module SimpleDiffEq

using Reexport
@reexport using DiffEqBase
using StaticArrays

include("functionmap.jl")
include("euler_maruyama.jl")
include("tsit5/tsit5.jl")
include("tsit5/atsit5.jl")

end # module
