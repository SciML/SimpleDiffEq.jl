# Reexported SciML common interface

`using SimpleDiffEq` also brings in the parts of the SciML common interface used to
construct and solve the problems supported by this package. These names are owned and
documented by [SciMLBase](https://docs.sciml.ai/SciMLBase/stable/):

  - Problems: `ODEProblem`, `SDEProblem`, and `DiscreteProblem`
  - Solving: `solve`, `init`, and `step!`
  - Integrator interface: `reinit!`

Anything else from SciMLBase must be imported from SciMLBase directly.
