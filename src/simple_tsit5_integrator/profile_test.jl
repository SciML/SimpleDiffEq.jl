function loop(u, p, t)
    @inbounds begin
        σ = p[1]; ρ = p[2]; β = p[3]
        du1 = σ*(u[2]-u[1])
        du2 = u[1]*(ρ-u[3]) - u[2]
        du3 = u[1]*u[2] - β*u[3]
        return SVector{3}(du1, du2, du3)
    end
end
function liip(du, u, p, t)
    σ = p[1]; ρ = p[2]; β = p[3]
    du[1] = σ*(u[2]-u[1])
    du[2] = u[1]*(ρ-u[3]) - u[2]
    du[3] = u[1]*u[2] - β*u[3]
    return nothing
end


# %%
begin
    u0 = 10ones(3)

    oop = init(MinimalTsit5(), loop, false, SVector{3}(u0), 0.0, 0.0001, [10, 28, 8/3])
    step!(oop)
    for i in 1:10000;
        step!(oop);
        if isnan(oop.u[1]) || isnan(oop.u[2]) || isnan(oop.u[3])
            error("oop nan")
        end
    end

    iip = init(MinimalTsit5(), liip, true, copy(u0), 0.0, 0.0001, [10, 28, 8/3])
    step!(iip)
    for i in 1:10000;
        step!(iip);
        if isnan(iip.u[1]) || isnan(iip.u[2]) || isnan(iip.u[3])
            error("iip nan")
        end
    end

    @profiler for i in 1:1000000; step!(iip); end

end

# @profiler for i in 1:1000000; step!(integ); end
# using PyPlot
#
# for integ in (iip, oop)
#     N = 10000
#     xs = zeros(N); ys = copy(xs); zs = copy(xs)
#
#     for i in 1:N
#         step!(integ)
#         xs[i], ys[i], zs[i] = integ.u
#     end
#
#     plot3D(xs, ys, zs)
# end
