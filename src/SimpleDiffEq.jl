__precompile__()

module SimpleDiffEq

# Explicit imports
using Reexport: @reexport
using MuladdMacro: @muladd
@reexport using DiffEqBase
using DiffEqBase: DiffEqBase
using SciMLBase: SciMLBase, ODEProblem, SDEProblem, DiscreteProblem, isinplace,
                 reinit!, u_modified!
using CommonSolve: step!
using StaticArrays: StaticArrays, SArray, MVector, SVector
using RecursiveArrayTools: RecursiveArrayTools, recursivecopy!
using LinearAlgebra: LinearAlgebra, mul!
using Parameters: Parameters
using UnPack: @unpack

# Explicit imports from Base
import Base: /, convert

@inline _copy(a::SArray) = a
@inline _copy(a) = copy(a)

abstract type AbstractSimpleDiffEqODEAlgorithm <: SciMLBase.AbstractODEAlgorithm end
SciMLBase.isautodifferentiable(alg::AbstractSimpleDiffEqODEAlgorithm) = true
SciMLBase.allows_arbitrary_number_types(alg::AbstractSimpleDiffEqODEAlgorithm) = true
SciMLBase.allowscomplex(alg::AbstractSimpleDiffEqODEAlgorithm) = true
SciMLBase.isadaptive(alg::AbstractSimpleDiffEqODEAlgorithm) = false # except 2, handled individually

function build_adaptive_controller_cache(::Type{T}) where {T}
    beta1 = T(7 / 50)
    beta2 = T(2 / 25)
    qmax = T(10.0)
    qmin = T(1 / 5)
    gamma = T(9 / 10)
    qoldinit = T(1e-4)
    qold = qoldinit

    return beta1, beta2, qmax, qmin, gamma, qoldinit, qold
end

include("functionmap.jl")
include("euler_maruyama.jl")
include("rk4/rk4.jl")
include("rk4/gpurk4.jl")
include("rk4/looprk4.jl")
include("euler/euler.jl")
include("euler/gpueuler.jl")
include("euler/loopeuler.jl")
include("tsit5/atsit5_cache.jl")
include("tsit5/tsit5.jl")
include("tsit5/atsit5.jl")
include("tsit5/gpuatsit5.jl")
include("verner/verner_tableaus.jl")
include("verner/gpuvern7.jl")
include("verner/gpuvern9.jl")

end # module
