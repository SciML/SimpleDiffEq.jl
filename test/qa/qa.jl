using SciMLTesting, SimpleDiffEq, Test
using JET

# JET `report_package(; mode = :typo)` is clean on Julia 1.10/1.11 (JET 0.9.x) but
# surfaces 36 reports on Julia 1.12 (JET 0.11.x): one genuine latent bug
# (src/euler/euler.jl:199, an inplace-interpolation path that reads `length(u)`
# before `u` is assigned) and ~35 `may be undefined` reports on correlated-branch
# control flow (`cur_t` in the tsit5 `solve`/`gpuatsit5`/`gpuvern` loops, `q11` in
# the adaptive `step!` error-estimate blocks) that the earlier JET could prove safe
# but JET 0.11 cannot. The source is byte-identical to master, so these predate this
# QA conversion. Mark the JET check known-broken on 1.12+ (tracked in
# SciML/SimpleDiffEq.jl#123); the marker auto-flips to an Unexpected Pass when the
# reports are resolved. Gated by VERSION so the still-clean 1.10/1.11 lanes keep
# running the hard `test_package` check (a blanket `jet_broken` would record an
# Unexpected-Pass error there).
const jet_broken = VERSION >= v"1.12"

run_qa(
    SimpleDiffEq;
    explicit_imports = true,
    jet_broken = jet_broken,
)
