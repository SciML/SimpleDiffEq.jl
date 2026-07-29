# SimpleDiffEq.jl

SimpleDiffEq.jl provides compact differential equation solvers that implement
the SciML common solve interface without the broader feature set of the full
solver packages.

## Algorithms

```@docs
SimpleEuler
LoopEuler
SimpleRK4
LoopRK4
SimpleTsit5
SimpleATsit5
GPUSimpleEuler
GPUSimpleRK4
GPUSimpleTsit5
GPUSimpleATsit5
GPUSimpleVern7
GPUSimpleAVern7
GPUSimpleVern9
GPUSimpleAVern9
SimpleEM
SimpleFunctionMap
```

## SciML Interface

SimpleDiffEq algorithms are used through the public SciMLBase problem and
solve interfaces. Construct an `ODEProblem`, `SDEProblem`, or
`DiscreteProblem` with SciMLBase, then pass one of the algorithms above to
`solve`. For manual ODE stepping, use the documented SciMLBase `init` and
`step!` interface. The extension and mutation rules for those shared
interfaces are defined by [SciMLBase](https://docs.sciml.ai/SciMLBase/stable/).
