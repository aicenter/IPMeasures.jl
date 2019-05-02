using IPMeasures, StatsBase, Distributions
using IPMeasures: mmd, GaussianKernel, IMQKernel
using Plots
plotly()



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

function split2(x)
	n = size(x,2)
	x[:,1:div(n,2)],x[:,div(n,2)+1:end]
end

criterion(k, x, y) = mmd(k, x, y) / sqrt(abs(mmd(k, split2(x)...)*mmd(k, split2(y)...)))


γs = -3:0.1:2
x = randn(2,1000);
y = rand(2,1000) .- [0.5,0.5];
y = randn(2,1000) .* [0.01,0.01];
x, y = sample_blobs(1000, 6.0, 5, 5, 10)
plot(γs, [criterion(IPMeasures.IMQKernel(10.0^γ), x, y) for γ in γs])

using Flux
plot(γs, [sum(abs.(Flux.data(Flux.Tracker.gradient(z -> mmd(IPMeasures.IMQKernel(10.0^γ), x, z), y)[1]))) for γ in γs])