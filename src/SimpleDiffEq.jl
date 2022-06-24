__precompile__()

module SimpleDiffEq

using Reexport, MuladdMacro
@reexport using DiffEqBase
using StaticArrays
using RecursiveArrayTools

@inline _copy(a::SArray) = a
@inline _copy(a) = copy(a)

abstract type AbstractSimpleDiffEqODEAlgorithm <: SciMLBase.AbstractODEAlgorithm end
SciMLBase.isautodifferentiable(alg::AbstractSimpleDiffEqODEAlgorithm) = true
SciMLBase.allows_arbitrary_number_types(alg::AbstractSimpleDiffEqODEAlgorithm) = true
SciMLBase.allowscomplex(alg::AbstractSimpleDiffEqODEAlgorithm) = true
SciMLBase.isadaptive(alg::AbstractSimpleDiffEqODEAlgorithm) = false # except 2, handled individually

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
