using SimpleDiffEq, Test, StaticArrays

# dX_t = 2u dt + dW_t
f(u, p, t) = 2u
g(u, p, t) = 1
u0 = 0.5
tspan = (0.0, 1.0)
prob = SDEProblem(f, g, u0, tspan)

sol = solve(prob, SimpleEM(), dt = 0.25)

@test sol.t == collect(0:0.25:1.0)
@test length(sol.u) == 5
@test typeof(sol) <: DESolution

u0 = @SVector [0.1, 0.2]
prob = SDEProblem(f, g, u0, tspan)
sol = solve(prob, SimpleEM(), dt = 0.25)

@test typeof(sol.u) <: Vector{SVector{2, Float64}}

f(du, u, p, t) = du .= 2.0 * u
g(du, u, p, t) = du .= 1
u0 = 0.5ones(4)
tspan = (0.0, 1.0)
prob = SDEProblem(f, g, u0, tspan)
sol = solve(prob, SimpleEM(), dt = 0.25)

@info "Non-Diagonal Noise OOP"

function f_oop(u, p, t)
    return 1.01 .* u
end

function g_oop(u, p, t)
    du1_1 = 0.3u[1]
    du1_2 = 0.6u[1]
    du1_3 = 0.9u[1]
    du1_4 = 0.12u[1]
    du2_1 = 1.2u[2]
    du2_2 = 0.2u[2]
    du2_3 = 0.3u[2]
    du2_4 = 1.8u[2]
    return [du1_1 du1_2 du1_3 du1_4; du2_1 du2_2 du2_3 du2_4]
end

u0 = ones(2)
noise_rate_prototype = zeros(2, 4)
prob = SDEProblem(f_oop, g_oop, u0, (0.0, 1.0), noise_rate_prototype = noise_rate_prototype)

sol = solve(prob, SimpleEM(), dt = 0.25)

@test length(sol.u) == 5
@test typeof(sol) <: DESolution

@info "Non-Diagonal Noise IIP"

f_oop(du, u, p, t) = du .= 1.01u
function g_oop(du, u, p, t)
    du[1, 1] = 0.3u[1]
    du[1, 2] = 0.6u[1]
    du[1, 3] = 0.9u[1]
    du[1, 4] = 0.12u[1]
    du[2, 1] = 1.2u[2]
    du[2, 2] = 0.2u[2]
    du[2, 3] = 0.3u[2]
    return du[2, 4] = 1.8u[2]
end
prob = SDEProblem(f_oop, g_oop, ones(2), (0.0, 1.0), noise_rate_prototype = zeros(2, 4))

sol = solve(prob, SimpleEM(), dt = 0.25)

@test length(sol.u) == 5
@test typeof(sol) <: DESolution
