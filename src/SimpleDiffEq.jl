__precompile__()

module SimpleDiffEq

    using Reexport: @reexport
    using MuladdMacro: @muladd
    @reexport using FastBroadcast: @..
    @reexport using DiffEqBase: DiffEqBase, ODEProblem, SDEProblem, DiscreteProblem,
        isinplace, reinit!, ODE_DEFAULT_NORM,
        set_t!, solve, step!, init, isdiscrete
    @reexport using SciMLBase: SciMLBase, build_solution, is_diagonal_noise,
        AbstractSDEAlgorithm, AbstractODEAlgorithm,
        AbstractODEIntegrator, DEIntegrator, ConstantInterpolation,
        __init, __solve, has_analytic, calculate_solution_errors!
    import SciMLBase: allows_arbitrary_number_types, allowscomplex, isautodifferentiable, isadaptive
    # `derivative_discontinuity!` was introduced in DiffEqBase v7 / SciMLBase v3,
    # replacing the older `u_modified!`. Support both branches so the package can
    # be used with either DiffEqBase v6 or v7.
    @static if isdefined(DiffEqBase, :derivative_discontinuity!)
        using DiffEqBase: derivative_discontinuity!
        export derivative_discontinuity!
    else
        const derivative_discontinuity! = DiffEqBase.u_modified!
        export derivative_discontinuity!
    end
    # Keep `u_modified!` re-exported for backwards compatibility with any user
    # code that still calls it.
    @static if isdefined(DiffEqBase, :u_modified!)
        using DiffEqBase: u_modified!
        export u_modified!
        @doc """
            u_modified!(integrator, modified::Bool)

        Mark whether an integrator state was externally modified.

        # Arguments

        - `integrator`: A SciML integrator that supports the mutation hook.
        - `modified::Bool`: Whether the current state should be treated as modified.

        # Returns

        Returns the implementation-defined result of the integrator hook.

        # Notes

        This is the legacy DiffEqBase compatibility name for
        `derivative_discontinuity!`. SimpleDiffEq reexports it when the
        installed DiffEqBase/SciMLBase stack still provides the binding.
        """
        SciMLBase.u_modified!
    end
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
