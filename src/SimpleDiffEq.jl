__precompile__()

module SimpleDiffEq

using Reexport
@reexport using DiffEqBase
using StaticArrays

include("functionmap.jl")
include("euler_maruyama.jl")
include("tsit5/atsit5_cache.jl")
include("tsit5/tsit5.jl")
include("tsit5/atsit5.jl")
include("tsit5/gpuatsit5.jl")

end # module
