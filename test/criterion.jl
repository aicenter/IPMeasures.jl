using Plots, Distributions
using IPMeasures
using IPMeasures: crit_mmd2_var, crit_mxy_over_mltpl, crit_mxy_over_sum
plotlyjs()

function sample_blobs(n, ratio, rows=5, cols=5, sep=10)
    correlation = (ratio - 1) / (ratio + 1)
    # generate within-blob variation
    X = rand(Distributions.Normal(0,1), 2, n)

    corr_sigma = [1  correlation;  correlation 1]
    Y = rand(Distributions.MultivariateNormal([0,0], corr_sigma), n)

    # assign to blobs
    X[1, :] .+= rand(1:rows,n) .* sep
    X[2, :] .+= rand(1:cols,n) * sep
    Y[1, :] .+= rand(1:rows,n) * sep
    Y[2, :] .+= rand(1:cols,n) * sep

    (X, Y)
end

@testset "Criteria comparison" begin
    γs = -3:0.1:2
    x = randn(2,1000);
    y = rand(2,1000) .- [0.5,0.5];
    y = randn(2,1000) .* [0.01,0.01];
    x, y = sample_blobs(1000, 6.0, 5, 5, 10)
    plot(γs, [crit_mmd2_var(IPMeasures.IMQKernel(10.0^γ), x, y) for γ in γs], label = "MMD2 / √VAR")
    plot!(γs, [crit_mxy_over_mltpl(IPMeasures.IMQKernel(10.0^γ), x, y) for γ in γs], label = "M(X,Y) / SQRT(M(X,X) * M(Y,Y)")
    p = plot!(γs, [crit_mxy_over_sum(IPMeasures.IMQKernel(10.0^γ), x, y) for γ in γs], label = "M(X,Y) / (M(X,X) + M(Y,Y)")
    @test p != nothing
end
