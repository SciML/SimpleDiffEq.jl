# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Runge-Kutta 4th order solver
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

struct SimpleRK4 end
export SimpleRK4

mutable struct SimpleRK4Integrator{IIP, S, T, P, F} <: DiffEqBase.AbstractODEIntegrator{SimpleRK4, IIP, S, T}
    f::F             # ..................................... Equations of motion
    uprev::S         # .......................................... Previous state
    u::S             # ........................................... Current state
    tmp::S           #  Auxiliary variable similar to state to avoid allocations
    tprev::T         # ...................................... Previous time step
    t::T             # ....................................... Current time step
    t0::T            # ........... Initial time step, only for re-initialization
    dt::T            # ............................................... Step size
    tdir::T          # ........................................ Not used for RK4
    p::P             # .................................... Parameters container
    u_modified::Bool # ..... If `true`, then the input of last step was modified
    ks::Vector{S}    # ........................... Interpolants of the algorithm
end

const SRK4 = SimpleRK4Integrator

# If `true`, then the equation of motion format is `f!(du,u,p,t)` instead of
# `du = f(u,p,t)`.
DiffEqBase.isinplace(::SRK4{IIP}) where {IIP} = IIP

################################################################################
#                                Initialization
################################################################################

function DiffEqBase.__init(prob::ODEProblem, alg::SimpleRK4;
                           dt = error("dt is required for this algorithm"))
    simplerk4_init(prob.f,
                   DiffEqBase.isinplace(prob),
                   prob.u0,
                   prob.tspan[1],
                   dt,
                   prob.p)
end

function DiffEqBase.__solve(prob::ODEProblem, alg::SimpleRK4;
                            dt = error("dt is required for this algorithm"))
    u0    = prob.u0
    tspan = prob.tspan
    ts    = Array(tspan[1]:dt:tspan[2])
    n     = length(ts)
    us    = Vector{typeof(u0)}(undef, n)

    @inbounds us[1] = _copy(u0)

    integ = simplerk4_init(prob.f, DiffEqBase.isinplace(prob), prob.u0,
                           prob.tspan[1], dt, prob.p)

    # FSAL
    for i = 1:n-1
        step!(integ)
        us[i+1] = _copy(integ.u)
    end

    sol = DiffEqBase.build_solution(prob, alg, ts, us, calculate_error = false)

    DiffEqBase.has_analytic(prob.f) &&
        DiffEqBase.calculate_solution_errors!(sol;
                                              timeseries_errors = true,
                                              dense_errors = false)

    return sol
end

@inline function simplerk4_init(f::F, IIP::Bool, u0::S, t0::T, dt::T, p::P) where
    {F, P, T, S<:AbstractArray{T}}

    # Allocate the vector with the interpolants. For RK4, we need 5.
    ks = [zero(u0) for _ = 1:5]

    integ = SRK4{IIP, S, T, P, F}(f,
                                  _copy(u0),
                                  _copy(u0),
                                  _copy(u0),
                                  t0,
                                  t0,
                                  t0,
                                  dt,
                                  sign(dt),
                                  p,
                                  true,
                                  ks)

    return integ
end

################################################################################
#                                   Stepping
################################################################################

@inline function DiffEqBase.step!(integ::SRK4{true, S, T}) where {T, S}
    integ.uprev       .= integ.u
    tmp                = integ.tmp
    f!                 = integ.f
    p                  = integ.p
    t                  = integ.t
    dt                 = integ.dt
    uprev              = integ.uprev
    u                  = integ.u
    k₁, k₂, k₃, k₄, k₅ = integ.ks

    # If the input was modified, then we need to recompute the initial pass of
    # this step.
    if integ.u_modified
        f!(k₁, uprev, p, t)
        integ.u_modified = false
    else
        k₁ .= k₅
    end

    # This algorithm is faster than the one using broadcasts.
    @inbounds begin
        L = length(u)

        for i in 1:L
            tmp[i] = uprev[i] + dt*k₁[i]/2
        end
        f!(k₂, tmp, p, t + dt/2)

        for i in 1:L
            tmp[i] = uprev[i] + dt*k₂[i]/2
        end
        f!(k₃, tmp, p, t + dt/2)

        for i in 1:L
            tmp[i] = uprev[i] + dt*k₃[i]
        end
        f!(k₄, tmp, p, t + dt)

        for i = 1:L
            u[i] = uprev[i] + (dt/6)*( 2*(k₂[i] + k₃[i]) + (k₁[i] + k₄[i]) )
        end

        f!(k₅, u, p, t + dt)
    end

    integ.tprev = t
    integ.t += dt

    return nothing
end

@inline function DiffEqBase.step!(integ::SRK4{false, S, T}) where {T, S}
    integ.uprev = integ.u
    f           = integ.f
    p           = integ.p
    t           = integ.t
    dt          = integ.dt
    uprev       = integ.uprev

    # If the input was modified, then we need to recompute the initial pass of
    # this step.
    if integ.u_modified
        k₁ = f(uprev, p, t)
        integ.u_modified = false
    else
        @inbounds k₁ = integ.ks[5]
    end

    tmp = uprev + dt*k₁/2
    k₂ = f(tmp, p, t + dt/2)

    tmp = uprev + dt*k₂/2
    k₃ = f(tmp, p, t + dt/2)

    tmp = uprev + dt*k₃
    k₄ = f(tmp, p, t + dt)

    integ.u = uprev + (dt/6)*( 2*(k₂ + k₃) + (k₁ + k₄) )
    k₅ = f(integ.u, p, t + dt)

    # Update the interpolants in the integrator. This is necessary for the
    # interpolation.
    @inbounds begin
        integ.ks[1] = k₁
        integ.ks[2] = k₂
        integ.ks[3] = k₃
        integ.ks[4] = k₄
        integ.ks[5] = k₅
    end

    integ.tprev = t
    integ.t += dt

    return nothing
end

################################################################################
#                                Interpolation
################################################################################

function (integ::SRK4)(t::T) where T
    t₁, t₀, dt = integ.t, integ.tprev, integ.dt

    y₀ = integ.uprev
    y₁ = integ.u
    ks = integ.ks
    Θ  = (t - t₀)/dt

    # Hermite interpolation.
    @inbounds if !isinplace(integ)
        u = (1-Θ)*y₀ + Θ*y₁ + Θ*(Θ-1)*( (1-2Θ)*(y₁-y₀) +
                                        (Θ-1)*dt*ks[1] +
                                        Θ*dt*ks[5])
        return u
    else
        u = similar(y₁)
        for i in 1:length(u)
            u[i] = (1-Θ)*y₀[i] + Θ*y₁[i] + Θ*(Θ-1)*( (1-2Θ)*(y₁[i]-y₀[i])+
                                                     (Θ-1)*dt*ks[1][i] +
                                                     Θ*dt*ks[5][i])
        end

        return u
    end
end
