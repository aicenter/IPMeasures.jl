using IPMeasures, Test
using IPMeasures: mapsum, pairwisel2, mahalanobis, mmd, GaussianKernel, RQKernel, IMQKernel, Mahalanobis, mmdfromdist, mmd2_and_variance

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

function distances(Σ, x, y)
	d = fill(0.0,size(x,2),size(y,2))
	for i in 1:size(x,2)
		for j in 1:size(y,2)
			δ = x[:,i] - y[:,j];
			d[i,j] = δ'*Σ*δ
		end 
	end
	d
end

@testset "distances" begin
	Σ = randn(4,4);Σ = Σ + Σ';
	x = randn(4,10)
	y = randn(4,9)
	@test distances(Σ,x) ≈ mahalanobis(Σ,x)
	@test distances(Σ, x, y) ≈ mahalanobis(Σ,x, y)
	m = Mahalanobis(Σ)
	@test distances(Σ,x) ≈ m(x)
	@test distances(Σ, x, y) ≈ m(x, y)
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
		@test mmdfromdist(k, d, i, j) .≈ mmd(k, x[:,i],x[:,j])
		@test mmd(k, randn(1,1000), randn(1,1000)) < 1e-2
	end
end

@testset "mmd2 and variance" begin
    o = ones(Float64, 3, 3)
    z = zeros(Float64, 3, 3)
    @test mmd2_and_variance(o, o, o) == (0, 0)
    @test mmd2_and_variance(z, z, z) == (0, 0)
    @test mmd2_and_variance(o, z, z) == (1, 0)
    @test mmd2_and_variance(z, z, o) == (1, 0)
    @test mmd2_and_variance(o, o, z) == (-1, 0)
    @test mmd2_and_variance(o, z, z) == (1, 0)
    @test mmd2_and_variance(z, o, z) == (-2, 0)

    a = [1 2 3; 4 5 6; 7 8 9]
    b = (a .+ 1)'
    c = (a .+ 2)'
    a = a'
    m, v = mmd2_and_variance(a, b, c)
    @test m ≈ 0
    @test v ≈ -3.1111111

    a = [1 1 1; 1 1 1; 0 1 0]'
    b = [1 1 1; 1 0 1; 0 1 0]'
    c = [0 1 1; 1 0 1; 0 1 0]'
    m, v = mmd2_and_variance(a, b, c)
    @test m ≈ 0.33333333
    @test v ≈ -0.0185185185185

    a = [0.31067363 0.17995522 0.69170826
        0.10956897 0.43168305 0.13072019
        0.98321756 0.66711701 0.4277054]'

    b = [0.6308643 0.72541977 0.26667574
        0.24141601 0.56088218 0.2872887 
        0.12801543 0.43956629 0.0703913]'

    c = [0.68779085 0.68050291 0.47736484
        0.49306397 0.07206057 0.10580517
        0.99297355 0.90194579 0.32241892]'

    m, v = mmd2_and_variance(a, b, c)
    @test m ≈ 0.32443064
    @test v ≈ 0.33610499
end

include("criterion.jl")

