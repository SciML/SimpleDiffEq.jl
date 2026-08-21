@setup_workload begin
    f(u, p, t) = 1.01 * u
    prob = ODEProblem(f, 0.5, (0.0, 1.0))

    @compile_workload begin
        SciMLBase.solve(prob, SimpleRK4(); dt = 0.1)
    end
end
