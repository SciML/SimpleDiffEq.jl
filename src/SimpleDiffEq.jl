__precompile__()

module SimpleDiffEq

using Reexport, MuladdMacro
@reexport using DiffEqBase
using StaticArrays
using RecursiveArrayTools

@inline _copy(a::SArray) = a
@inline _copy(a) = copy(a)

include("functionmap.jl")
include("euler_maruyama.jl")
include("tsit5/atsit5_cache.jl")
include("tsit5/tsit5.jl")
include("tsit5/atsit5.jl")
include("tsit5/gpuatsit5.jl")

end # module
