using IPMeasures, Test
using IPMeasures: mapsum, pairwisel2, mahalanobis, mmd, GaussianKernel, RQKernel, IMQKernel

function distances(Σ,x)
	d = fill(0.0,size(x,2),size(x,2))
	for i in 1:size(x,2)
		for j in 1:size(x,2)
			δ = x[:,i] - x[:,j];
			d[i,j] = δ'*Σ*δ
		end 
	end
	d
end

@testset "distances" begin
	Σ = randn(4,4);Σ = Σ + Σ';
	x = randn(4,10)
	@test distances(Σ,x) ≈ mahalanobis(Σ,x)
end




@testset "mapsum" begin
	d = randn(5,5)
	@test mapsum(exp, d , [1,2]) ≈ sum(exp.(d[1:2,1:2]))
	@test mapsum(exp, d , [1,5]) ≈ sum(exp.(d[[1,5],[1,5]]))
	@test mapsum(exp, d , [1,2],[3,5]) ≈ sum(exp.(d[1:2,[3,5]]))
end


@testset "kernels" begin
	x = randn(3,10)
	d = pairwisel2(x)
	i, j = [1,2,3], [4,5,6]
	for k in [GaussianKernel(1.0), RQKernel(1.0), IMQKernel(1.0)]
		@test mmd(k, d, i, j) ≈ mmd(k, x[:,i],x[:,j])
		@test mmd(k, randn(1,1000), randn(1,1000)) < 1e-2
	end
end

