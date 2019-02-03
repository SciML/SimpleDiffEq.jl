using SimpleDiffEq, Test

@time begin
  @time @testset "Discrete Tests" begin include("discrete_tests.jl") end
  @time @testset "SimpleEM Tests" begin include("simpleem_tests.jl") end
  @time @testset "SimpleTsit5 Tests" begin include("simpletsit5_tests.jl") end
  @time @testset "SimpleATsit5 Tests" begin include("simpleatsit5_tests.jl") end
  @time @testset "GPUSimpleATsit5 Tests" begin include("gpusimpleatsit5_tests.jl") end
end
