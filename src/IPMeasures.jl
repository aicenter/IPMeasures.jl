module IPMeasures

using StatsBase
using LinearAlgebra
using Distances
using CUDA

using Distributions
using DistributionsAD
using ConditionalDists

const MetricOrFun = Union{PreMetric,Function}
const CuMatrix = CuArray{<:Float32,2}


"""
    samplecolumns(x::AbstractMatrix, n::Int)

Sample n columns from a matrix. Returns x if the matrix has less than n columns.
"""
function samplecolumns(x::AbstractMatrix, n::Int)
    (size(x,2) > n) ? x[:,sample(1:size(x,2), n, replace=false)] : x
end

"""
    split2(x::AbstractMatrix)

Splits a matrix into halves.
"""
function split2(x::AbstractMatrix)
	n = size(x,2)
	x[:,1:div(n,2)], x[:,div(n,2)+1:end]
end


"""
	pairwisel2(x,y)
	pairwisel2(x)

Calculates pairwise squared euclidean distances of the columns of `x` and `y`
or `x` and `x`. The dispatches for CuArrays are necessary until
https://github.com/JuliaStats/Distances.jl/pull/142 is merged.
"""
pairwisel2(x::Matrix, y::Matrix) = pairwise(SqEuclidean(), x, y, dims=2)
pairwisel2(x::CuMatrix, y::CuMatrix) =
    -2 .* x' * y .+ sum(x.^2, dims=1)' .+ sum(y.^2,dims=1)
pairwisel2(x::AbstractMatrix) = pairwisel2(x,x)

"""
	pairwisel2(x,y)
	pairwisel2(x)

Calculates pairwise cosine distances of the columns of `x` and `y` or
`x` and `x`. The dispatches for CuArrays are necessary until
https://github.com/JuliaStats/Distances.jl/pull/142 is merged.
"""
pairwisecos(x::Matrix, y::Matrix) = pairwise(CosineDist(), x, y, dims=2)
pairwisecos(x::CuMatrix, y::CuMatrix) = 1 .- (x' * y / norm(x) / norm(y))
pairwisecos(x::AbstractMatrix) = pairwisecos(x,x)


include("kernels.jl")
include("mmd.jl")
include("kl_divergence.jl")

 
"""
	null_distribution(k::AbstractKernel, x, n, l = div(size(x,2),2))

Estimates the null distribution of samples `x` from `n` random draws of subsets of size `l`
"""
null_distribution(k::AbstractKernel, x, n, l = div(size(x,2),2)) =
    _null_distribution(k, pairwisel2(x,x), n, l )

function _null_distribution(k, d, n, l)
	ns = size(d,2)
	map(1:n) do _
		idx = sample(1:ns, l, replace = false)
		cidx = setdiff(1:ns, idx)
		cidx = sample(cidx, min(l, length(cidx)), replace = false) 
		mmdfromdist(k, d, idx, cidx)
	end
end

end # module
