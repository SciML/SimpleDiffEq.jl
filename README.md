# SimpleDiffEq.jl

[![Build Status](https://github.com/SciML/SimpleDiffEq.jl/workflows/CI/badge.svg)](https://github.com/SciML/SimpleDiffEq.jl/actions?query=workflow%3ACI)
[![Coverage Status](https://coveralls.io/repos/SciML/SimpleDiffEq.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/SciML/SimpleDiffEq.jl?branch=master)
[![codecov.io](http://codecov.io/github/SciML/SimpleDiffEq.jl/coverage.svg?branch=master)](http://codecov.io/github/SciML/SimpleDiffEq.jl?branch=master)

SimpleDiffEq.jl is a library of basic differential equation solvers. They are
the "no-cruft" versions of the solvers which don't and won't ever support
any fancy features like events. They are self-contained. This library exists
for a few purposes. For one, it can be a nice way to teach "how to write a
solver for X" in Julia by having a simple yet optimized version. Secondly,
since it's hooked onto the common interface, these algorithms can serve as
benchmarks to test the overhead of the full integrators on the simplest case.
Lastly, these can be used to test correctness of the more complicated
implementations.

## Installation

```julia
using Pkg
Pkg.add("SimpleDiffEq")
```

## Available Algorithms

### ODE Solvers

| Algorithm | Description |
|-----------|-------------|
| `SimpleEuler` | Forward Euler method |
| `LoopEuler` | Loop-based Euler (optimized for teaching/benchmarking) |
| `SimpleRK4` | Classic Runge-Kutta 4th order method |
| `LoopRK4` | Loop-based RK4 (optimized for teaching/benchmarking) |
| `SimpleTsit5` | Tsitouras 5th order method (fixed step) |
| `SimpleATsit5` | Adaptive Tsitouras 5th order method |

### GPU-Compatible ODE Solvers

These solvers are designed for GPU compatibility (out-of-place only):

| Algorithm | Description |
|-----------|-------------|
| `GPUSimpleRK4` | GPU-compatible RK4 |
| `GPUSimpleTsit5` | GPU-compatible Tsit5 (fixed step) |
| `GPUSimpleATsit5` | GPU-compatible adaptive Tsit5 |
| `GPUSimpleVern7` | GPU-compatible Verner 7th order |
| `GPUSimpleAVern7` | GPU-compatible adaptive Verner 7th order |
| `GPUSimpleVern9` | GPU-compatible Verner 9th order |
| `GPUSimpleAVern9` | GPU-compatible adaptive Verner 9th order |

### SDE Solvers

| Algorithm | Description |
|-----------|-------------|
| `SimpleEM` | Euler-Maruyama method for SDEs |

### Discrete Solvers

| Algorithm | Description |
|-----------|-------------|
| `SimpleFunctionMap` | Simple function iteration for discrete problems |

## Usage Examples

### Solving an ODE with SimpleRK4

```julia
using SimpleDiffEq, StaticArrays

# Define the Lorenz system (out-of-place form)
function lorenz(u, p, t)
    σ, ρ, β = p
    du1 = σ * (u[2] - u[1])
    du2 = u[1] * (ρ - u[3]) - u[2]
    du3 = u[1] * u[2] - β * u[3]
    return SVector(du1, du2, du3)
end

u0 = SVector(1.0, 0.0, 0.0)
tspan = (0.0, 100.0)
p = (10.0, 28.0, 8/3)

prob = ODEProblem(lorenz, u0, tspan, p)
sol = solve(prob, SimpleRK4(), dt = 0.01)
```

### Using the Adaptive Tsit5 Solver

```julia
using SimpleDiffEq

function f(u, p, t)
    return 1.01 * u
end

u0 = 0.5
tspan = (0.0, 1.0)
prob = ODEProblem(f, u0, tspan)

# Adaptive stepping with SimpleATsit5
sol = solve(prob, SimpleATsit5(), dt = 0.1, abstol = 1e-6, reltol = 1e-3)
```

### In-Place Form (for mutable arrays)

```julia
using SimpleDiffEq

function lorenz!(du, u, p, t)
    σ, ρ, β = p
    du[1] = σ * (u[2] - u[1])
    du[2] = u[1] * (ρ - u[3]) - u[2]
    du[3] = u[1] * u[2] - β * u[3]
    return nothing
end

u0 = [1.0, 0.0, 0.0]
tspan = (0.0, 100.0)
p = (10.0, 28.0, 8/3)

prob = ODEProblem(lorenz!, u0, tspan, p)
sol = solve(prob, SimpleRK4(), dt = 0.01)
```

### Using the Integrator Interface

```julia
using SimpleDiffEq

prob = ODEProblem((u, p, t) -> 1.01 * u, 0.5, (0.0, 1.0))
integrator = init(prob, SimpleRK4(), dt = 0.1)

# Step through the solution
step!(integrator)
step!(integrator)

# Access current state
integrator.u  # current solution
integrator.t  # current time
```

## Notes

- All solvers require a fixed `dt` parameter (no automatic step size selection for non-adaptive methods)
- GPU-compatible solvers only support out-of-place problem definitions
- These solvers integrate with the SciML common interface but do not support advanced features like callbacks or events

For full-featured differential equation solving, see [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl) and the [DifferentialEquations.jl](https://docs.sciml.ai/DiffEqDocs/stable/) documentation.
