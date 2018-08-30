using IPMeasures, Test
using IPMeasures: mapsum, pairwisel2, mmd, GaussianKernel

@testset "mapsum" begin
	d = randn(5,5)
	@test mapsum(exp, d , [1,2]) ≈ sum(exp.(d[1:2,1:2]))
	@test mapsum(exp, d , [1,5]) ≈ sum(exp.(d[[1,5],[1,5]]))
	@test mapsum(exp, d , [1,2],[3,5]) ≈ sum(exp.(d[1:2,[3,5]]))
end


@testset "Gaussian MMD" begin
	x = randn(3,10)
	d = pairwisel2(x)
	k = GaussianKernel(1.0)
	i, j = [1,2,3], [4,5,6]
	@test mmd(k, d, i, j) ≈ mmd(k, x[:,i],x[:,j])
	@test mmd(k, randn(1,1000), randn(1,1000)) < 1e-2
end

