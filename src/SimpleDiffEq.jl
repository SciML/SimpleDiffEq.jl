__precompile__()

module SimpleDiffEq

    using Reexport: @reexport
    using MuladdMacro: @muladd
    @reexport using DiffEqBase: DiffEqBase, ODEProblem, SDEProblem, DiscreteProblem,
        isinplace, reinit!, u_modified!, ODE_DEFAULT_NORM,
        set_t!, solve, step!, init, DESolution, @..,
        AbstractODEIntegrator, DEIntegrator, ConstantInterpolation,
        __init, __solve, build_solution, has_analytic,
        calculate_solution_errors!, is_diagonal_noise,
        AbstractSDEAlgorithm, AbstractODEAlgorithm, isdiscrete, SciMLBase
    import DiffEqBase.SciMLBase: allows_arbitrary_number_types, allowscomplex, isautodifferentiable, isadaptive
    using StaticArrays: SArray, SVector, MVector
    using RecursiveArrayTools: recursivecopy!
    using LinearAlgebra: mul!
    using Parameters: @unpack

    @inline _copy(a::SArray) = a
    @inline _copy(a) = copy(a)

    abstract type AbstractSimpleDiffEqODEAlgorithm <: AbstractODEAlgorithm end
    isautodifferentiable(alg::AbstractSimpleDiffEqODEAlgorithm) = true
    allows_arbitrary_number_types(alg::AbstractSimpleDiffEqODEAlgorithm) = true
    allowscomplex(alg::AbstractSimpleDiffEqODEAlgorithm) = true
    isadaptive(alg::AbstractSimpleDiffEqODEAlgorithm) = false # except 2, handled individually

    function build_adaptive_controller_cache(::Type{T}) where {T}
        beta1 = T(7 / 50)
        beta2 = T(2 / 25)
        qmax = T(10.0)
        qmin = T(1 / 5)
        gamma = T(9 / 10)
        qoldinit = T(1.0e-4)
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
