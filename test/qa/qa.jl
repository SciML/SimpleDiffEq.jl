using SciMLTesting, SimpleDiffEq, Test

const REEXPORTS = (
    :DiscreteProblem, :ODEProblem, :SDEProblem, :init, :reinit!, :solve, :step!,
)

run_qa(SimpleDiffEq; reexports_allow = REEXPORTS)

@testset "Reexport surface" begin
    @testset "$name" for name in REEXPORTS
        @test name in names(SimpleDiffEq)
        @test isdefined(@__MODULE__, name)
    end
end
