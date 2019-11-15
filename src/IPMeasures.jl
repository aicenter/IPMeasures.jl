module IPMeasures

using StatsBase, LinearAlgebra, Distances
MetricOrFun = Union{PreMetric,Function}

"""
    samplecolumns(x::AbstractMatrix, n::Int)

Sample n columns from a matrix. Returns x if the matrix has less than n columns.
"""
function samplecolumns(x::AbstractMatrix, n::Int)
    (size(x,2) > n) ? x[:,sample(1:size(x,2), n, replace=false)] : x
end

"""
    split2(x)

Splits a vector into halves.
"""
function split2(x)
	n = size(x,2)
	x[:,1:div(n,2)], x[:,div(n,2)+1:end]
end

pairwisel2(x,y) = pairwise(SqEuclidean(), x, y)

include("kernels.jl")
include("mmd.jl")
 
"""
	null_distribution(k::AbstractKernel, x, n, l = div(size(x,2),2))

Estimates the null distribution of samples `x` from `n` random draws of subsets of size `l`
"""
null_distribution(k::AbstractKernel, x, n, l = div(size(x,2),2)) =
    _null_distribution(k, pairwisel2(x), n, l )

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
