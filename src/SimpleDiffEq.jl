__precompile__()

module SimpleDiffEq

using Reexport, MuladdMacro
@reexport using DiffEqBase
using StaticArrays
using RecursiveArrayTools
const ^ = DiffEqBase.fastpow

@inline _copy(a::SArray) = a
@inline _copy(a) = copy(a)

include("functionmap.jl")
include("euler_maruyama.jl")
include("rk4/rk4.jl")
include("rk4/gpurk4.jl")
include("rk4/looprk4.jl")
include("tsit5/atsit5_cache.jl")
include("tsit5/tsit5.jl")
include("tsit5/atsit5.jl")
include("tsit5/gpuatsit5.jl")

end # module
