"""
    SimpleEM()

Construct a fixed-step Euler-Maruyama algorithm for `SDEProblem`s.

`SimpleEM` applies the Euler-Maruyama method to stochastic differential
equations. It supports in-place and out-of-place drift/diffusion functions, and
handles diagonal and non-diagonal noise.

# Arguments

No positional arguments are accepted.

# Keywords

No constructor keywords are accepted. Pass `dt` to `solve`.

# Returns

A `SimpleEM` algorithm object for use with `SDEProblem`.

# Example

```julia
using SimpleDiffEq

f(u, p, t) = 0.1u
g(u, p, t) = 0.2u

u0 = 1.0
tspan = (0.0, 1.0)
prob = SDEProblem(f, g, u0, tspan)

sol = solve(prob, SimpleEM(), dt = 0.01)
```

# Notes

`dt` is required. The method has strong order `1/2` for general SDEs.

# See Also

`SDEProblem`
"""
struct SimpleEM <: SciMLBase.AbstractSDEAlgorithm end
export SimpleEM

@muladd function DiffEqBase.solve(
        prob::SDEProblem{uType, tType, false}, alg::SimpleEM,
        args...;
        dt = error("dt required for SimpleEM"),
        kwargs...
    ) where {
        uType,
        tType,
    }
    f = prob.f
    g = prob.g
    u0 = prob.u0
    tspan = prob.tspan
    p = prob.p

    is_diagonal_noise = SciMLBase.is_diagonal_noise(prob)

    @inbounds begin
        n = Int((tspan[2] - tspan[1]) / dt) + 1
        u = [u0 for i in 1:n]
        t = [tspan[1] + i * dt for i in 0:(n - 1)]
        sqdt = sqrt(dt)
    end

    @inbounds for i in 2:n
        uprev = u[i - 1]
        tprev = t[i - 1]

        if is_diagonal_noise
            if u0 isa Number
                u[i] = uprev + f(uprev, p, tprev) * dt +
                    sqdt * g(uprev, p, tprev) * randn(typeof(u0))
            else
                u[i] = uprev + f(uprev, p, tprev) * dt +
                    sqdt * g(uprev, p, tprev) .* randn(typeof(u0))
            end
        else
            u[i] = uprev + f(uprev, p, tprev) * dt +
                sqdt * g(uprev, p, tprev) * randn(size(prob.noise_rate_prototype, 2))
        end
    end

    sol = SciMLBase.build_solution(
        prob, alg, t, u,
        calculate_error = false
    )
end

@muladd function DiffEqBase.solve(
        prob::SDEProblem{uType, tType, true}, alg::SimpleEM,
        args...;
        dt = error("dt required for SimpleEM"),
        kwargs...
    ) where {
        uType,
        tType,
    }
    f = prob.f
    g = prob.g
    u0 = prob.u0
    tspan = prob.tspan
    p = prob.p
    ftmp = zero(u0)
    gtmp = SciMLBase.is_diagonal_noise(prob) ? zero(u0) : zero(prob.noise_rate_prototype)
    gtmp2 = SciMLBase.is_diagonal_noise(prob) ? nothing : zero(u0)
    dW = SciMLBase.is_diagonal_noise(prob) ? zero(u0) :
        false .* prob.noise_rate_prototype[1, :]

    @inbounds begin
        n = Int((tspan[2] - tspan[1]) / dt) + 1
        u = [copy(u0) for i in 1:n]
        t = [tspan[1] + i * dt for i in 0:(n - 1)]
        sqdt = sqrt(dt)
    end

    @inbounds for i in 2:n
        uprev = u[i - 1]
        tprev = t[i - 1]
        f(ftmp, uprev, p, tprev)
        g(gtmp, uprev, p, tprev)
        @. dW = randn(eltype(dW))

        if SciMLBase.is_diagonal_noise(prob)
            @.. u[i] = uprev + ftmp * dt + sqdt * gtmp * dW
        else
            mul!(gtmp2, gtmp, dW)
            @.. u[i] = uprev + ftmp * dt + sqdt * gtmp2
        end
    end

    sol = SciMLBase.build_solution(
        prob, alg, t, u,
        calculate_error = false
    )
end
