module IPMeasures
using StatsBase
include("kernels.jl")
include("distances.jl")

samplecolumns(x,n) =  (size(x,2) > n) ? x[:,sample(1:size(x,2),n,replace = false)] : x

"""
		k_gaussian(x,y,γ)
		k_gaussian(x,γ)
		k_gaussian(d::M, γ, idx, cidx)

		kernel matrix corresponding to the gaussian kernel with `γ` on diagonal
"""
kernelsum(k::AbstractKernel, x, y, distfun) = sum(k.(distfun(x,y)))/(size(x,2) * size(y,2))
kernelsum(k::AbstractKernel, x::T, distfun) where {T<:AbstractMatrix} = (l = size(x,2); (sum(k.(distfun(x,x))) - l*k(0.0))/(l^2 - l)) 
kernelsum(k::AbstractKernel, x::T, distfun) where {T<:AbstractVector} = zero(eltype(x)) 


"""
		mmd(AbstractKernel(γ), x, y)
		mmd(AbstractKernel(γ), x, y, n)

		mmd with gaussian kernel of bandwidth `γ` using at most `n` samples
"""
mmd(k::AbstractKernel, x, y, distfun = pairwisel2) = kernelsum(k, x, distfun) + kernelsum(k, y, distfun) - 2*kernelsum(k, x, y, distfun)
mmd(k::AbstractKernel, x, y, n::Int, distfun = pairwisel2) = mmd(k, samplecolumns(x,n), samplecolumns(y,n), distfun)
function mmdfromdist(k::AbstractKernel, d, idx, cidx)
	li, lc = length(idx), length(cidx)
	kxx = (mapsum(k, d, idx) - li*k(0.0))/(li^2 - li)
	kyy = (mapsum(k, d, cidx) - lc*k(0.0))/(lc^2 - lc)
	kyx = mapsum(k, d, idx, cidx)/ (lc*li)
	kxx + kyy -2kyx
end


mapsum(f, d, idx) = (s = 0.0; @inbounds for j in idx, i in idx s+= f(d[i,j]) end; s)
mapsum(f, d, rowidx, colidx) = (s = 0.0; @inbounds for i in rowidx, j in colidx s+= f(d[i,j]) end; s)

"""
		null_distribution(k::AbstractKernel, x, n, l = div(size(x,2),2)) = _null_distribution(k, pairwisel2(x), n, l )

		estimates the null distribution of samples `x` from `n` random draws of subsets of size `l`
"""
null_distribution(k::AbstractKernel, x, n, l = div(size(x,2),2)) = _null_distribution(k, pairwisel2(x), n, l )
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
