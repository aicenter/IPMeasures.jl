const CuMatrix = CuArray{<:Float32,2}

"""
	pairwisel2(x,y)
	pairwisel2(x)

Calculates pairwise squared euclidean distances the columns of `x` and `y` or
`x` and `x`. The dispatches for CuArrays are necessary until
https://github.com/JuliaStats/Distances.jl/pull/142 is merged.
"""
pairwisel2(x::Matrix, y::Matrix) = pairwise(SqEuclidean(), x, y, dims=2)
pairwisel2(x::CuMatrix, y::CuMatrix) =
    -2 .* x' * y .+ sum(x.^2, dims=1)' .+ sum(y.^2,dims=1)
pairwisel2(x::AbstractMatrix) = pairwisel2(x,x)

pairwisecos(x::Matrix, y::Matrix) = pairwise(CosineDist(), x, y, dims=2)
pairwisecos(x::CuMatrix, y::CuMatrix) = 1 .- (x' * y / norm(x) / norm(y))
pairwisecos(x::AbstractMatrix) = pairwisecos(x,x)
