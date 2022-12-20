struct SimpleEM <: DiffEqBase.AbstractSDEAlgorithm end
export SimpleEM

@muladd function DiffEqBase.solve(prob::SDEProblem{uType, tType, false}, alg::SimpleEM,
                                  args...;
                                  dt = error("dt required for SimpleEM")) where {uType,
                                                                                 tType}
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

    sol = DiffEqBase.build_solution(prob, alg, t, u,
                                    calculate_error = false)
end

@muladd function DiffEqBase.solve(prob::SDEProblem{uType, tType, true}, alg::SimpleEM,
                                  args...;
                                  dt = error("dt required for SimpleEM")) where {uType,
                                                                                 tType}
    f = prob.f
    g = prob.g
    u0 = prob.u0
    tspan = prob.tspan
    p = prob.p
    ftmp = zero(u0)
    gtmp = DiffEqBase.is_diagonal_noise(prob) ? zero(u0) : zero(prob.noise_rate_prototype)
    gtmp2 = DiffEqBase.is_diagonal_noise(prob) ? nothing : zero(u0)
    dW = DiffEqBase.is_diagonal_noise(prob) ? zero(u0) :
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

        if DiffEqBase.is_diagonal_noise(prob)
            DiffEqBase.@.. u[i] = uprev + ftmp * dt + sqdt * gtmp * dW
        else
            mul!(gtmp2, gtmp, dW)
            DiffEqBase.@.. u[i] = uprev + ftmp * dt + sqdt * gtmp2
        end
    end

    sol = DiffEqBase.build_solution(prob, alg, t, u,
                                    calculate_error = false)
end
