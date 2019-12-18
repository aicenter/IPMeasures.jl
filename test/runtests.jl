using IPMeasures, Test
using IPMeasures: mapsum, pairwisel2, mahalanobis, mmd, GaussianKernel, RQKernel, IMQKernel, Mahalanobis, mmdfromdist, SumOfKernels

using IPMeasures: mapsum, mmd, mmdfromdist, mmd2_and_variance,
    GaussianKernel, RQKernel, IMQKernel

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
include("mmd.jl")
include("kl_divergence.jl")