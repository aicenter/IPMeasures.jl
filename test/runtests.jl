using Test
using Distances
using Distributions
using ConditionalDists
using IPMeasures
using CUDA
using Flux


using IPMeasures: mapsum, mmd, mmdfromdist, mmd2_and_variance,
    GaussianKernel, RQKernel, IMQKernel, null_distribution, samplecolumns,
    split2, pairwisel2

@testset "samplecolumns" begin
    xs = randn(2,5)
    ys = samplecolumns(xs, 2)
    @test sum(all(xs .≈ ys[:,1], dims=1)) == 1
    @test sum(all(xs .≈ ys[:,2], dims=1)) == 1

    ys = samplecolumns(xs, 12)
    @test all(xs .== ys)
end

@testset "split2" begin
    x = randn(2,10)
    (a,b) = split2(x)
    @test all(x[:,1:5] .== a)
    @test all(x[:,6:10] .== b)
end

@testset "pairwisel2" begin
    n = 20
    x = rand(Float32, n, 3) .* 45
    y = rand(Float32, n, 3) .* -120

    dcpu = pairwisel2(x,y)
    dgpu = pairwisel2(gpu(x), gpu(y))
    @test dgpu isa typeof(gpu(x))
    @test dcpu ≈ cpu(dgpu) rtol=1e-2
end

@testset "mapsum" begin
	d = randn(5,5)
	@test mapsum(exp, d , [1,2]) ≈ sum(exp.(d[1:2,1:2]))
	@test mapsum(exp, d , [1,5]) ≈ sum(exp.(d[[1,5],[1,5]]))
	@test mapsum(exp, d , [1,2],[3,5]) ≈ sum(exp.(d[1:2,[3,5]]))
end

@testset "kernels" begin
	x = randn(3,10)
    d = IPMeasures.pairwisel2(x,x)
	i, j = [1,2,3], [4,5,6]
	for k in [GaussianKernel(1.0), RQKernel(1.0), IMQKernel(1.0)]
		@test mmdfromdist(k, d, i, j) .≈ mmd(k, x[:,i],x[:,j])
		@test mmd(k, randn(1,1000), randn(1,1000)) < 1e-2
	end
end

@testset "Null distribution" begin
    k = GaussianKernel(1.0)
    x = randn(100, 100)
    n = 100
    d = null_distribution(k,x,n)
    @test size(d) == (100,)
end

include("mmd.jl")
include("kl_divergence.jl")