"""
	pairwisel2(x,y)
	pairwisel2(x)

	Calculates pairwise distances with L2 distance between `x` and `y` or `x` and `x`

"""
pairwisel2(x,y) = -2 .* x' * y .+ sum(x.^2, dims = 1)' .+ sum(y.^2,dims = 1)
pairwisel2(x) = -2 .* x' * x .+ sum(x.^2, dims = 1)' .+ sum(x.^2,dims = 1)

pairwisecos(x, y) = 1 .- (x' * y / norm(x) / norm(y))
pairwisecos(x) = pairwisecos(x, x)

struct Mahalanobis{T}
	Σ::T
end

(m::Mahalanobis)(x) = mahalanobis(m.Σ,x)
(m::Mahalanobis)(x, y) = mahalanobis(m.Σ,x, y)

function mahalanobis(Σ,x)
	Σx = Σ*x
	nx = sum( x.*Σx, dims = 1)
	nx .+ nx' - 2*x'*Σx
end

function mahalanobis(Σ, x, y)
	Σx, Σy = Σ*x, Σ*y
	sum(x.*Σx, dims = 1)' .+ sum(y.*Σy, dims = 1) - x'*Σy - (y'*Σx)'
end
