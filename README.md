# SimpleDiffEq.jl

[![Build Status](https://travis-ci.org/JuliaDiffEq/SimpleDiffEq.jl.svg?branch=master)](https://travis-ci.org/JuliaDiffEq/SimpleDiffEq.jl)
[![Coverage Status](https://coveralls.io/repos/ChrisRackauckas/SimpleDiffEq.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/ChrisRackauckas/SimpleDiffEq.jl?branch=master)
[![codecov.io](http://codecov.io/github/ChrisRackauckas/SimpleDiffEq.jl/coverage.svg?branch=master)](http://codecov.io/github/ChrisRackauckas/SimpleDiffEq.jl?branch=master)

SimpleDiffEq.jl is a library of basic differential equation solvers. They are
the "no-cruft" versions of the solvers which don't and won't ever support
any fancy features like events. They are self-contained. This library exists
for a few purposes. For one, it can be a nice way to teach "how to write a
solver for X" in Julia by having a simple yet optimized version. Secondly,
since it's hooked onto the common interface, these algorithms can serve as
benchmarks to test the overhead of the full integrators on the simplest case.
Lastly, these can be used to test correctness of the more complicated
implementations.
