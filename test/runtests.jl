using SimpleDiffEq
using Base.Test

tic()
@time @testset "Discrete Tests" begin include("discrete_tests.jl") end
toc()
