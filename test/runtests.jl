using SimpleDiffEq, Test

@time begin
    @time @testset "Discrete Tests" include("discrete_tests.jl")
    @time @testset "SimpleEM Tests" include("simpleem_tests.jl")
    @time @testset "SimpleTsit5 Tests" include("simpletsit5_tests.jl")
    @time @testset "SimpleATsit5 Tests" include("simpleatsit5_tests.jl")
    @time @testset "GPUSimpleATsit5 Tests" include("gpusimpleatsit5_tests.jl")
    @time @testset "SimpleRK4 Tests" include("simplerk4_tests.jl")
    @time @testset "SimpleEuler Tests" include("simpleeuler_tests.jl")
    @time @testset "GPU Compatible ODE Tests" include("gpu_ode_regression.jl")
end
