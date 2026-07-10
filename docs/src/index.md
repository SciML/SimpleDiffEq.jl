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

## Compatibility Hooks

```@docs
u_modified!
```

## Reexported SciML Interface

SimpleDiffEq also reexports these upstream SciML interface names for
convenience:

<!-- public-api-reexports-start -->
- `@..`
- `AbstractODEAlgorithm`
- `AbstractODEIntegrator`
- `AbstractSDEAlgorithm`
- `ConstantInterpolation`
- `DEIntegrator`
- `DiffEqBase`
- `DiscreteProblem`
- `ODEProblem`
- `ODE_DEFAULT_NORM`
- `SDEProblem`
- `SciMLBase`
- `__init`
- `__solve`
- `build_solution`
- `calculate_solution_errors!`
- `derivative_discontinuity!`
- `has_analytic`
- `init`
- `is_diagonal_noise`
- `isdiscrete`
- `isinplace`
- `reinit!`
- `set_t!`
- `solve`
- `step!`
<!-- public-api-reexports-end -->
