module IPMeasures
using StatsBase


samplecolumns(x,n) =  (size(x,2) > n) ? x[:,sample(1:size(x,2),n,replace = false)] : x

"""
	pairwisel2(x,y)
	pairwisel2(x)

	Calculates pairwise distances with L2 distance between `x` and `y` or `x` and `x`

"""
pairwisel2(x,y) = -2 .* x' * y .+ sum(x.^2,1)' .+ sum(y.^2,1)
pairwisel2(x) = -2 .* x' * x .+ sum(x.^2,1)' .+ sum(x.^2,1)

"""
		k_gaussian(x,y,γ)
		k_gaussian(x,γ)

		kernel matrix corresponding to the gaussian kernel with `γ` on diagonal
"""
k_gaussian(x,y,γ) = sum(exp.(-γ .* pairwisel2(x,y)))/(size(x,2) * size(y,2))
k_gaussian(x::T,γ) where {T<:AbstractMatrix} = sum(exp.(-γ .* pairwisel2(x)))/(size(x,2) * (size(x,2) -1 )) 
k_gaussian(x::T,γ) where {T<:AbstractVector} = zero(eltype(x)) 

"""
		mmdg(x,y,γ)
		mmdg(x,y,γ,n)

		mmd with gaussian kernel of bandwidth `γ` using at most `n` samples
"""
mmdg(x,y,γ) = k_gaussian(x,γ) + k_gaussian(y,γ) - 2*k_gaussian(x,y,γ)
mmdg(x,y,γ,n) = mmdg(samplecolumns(x,n), samplecolumns(y,n), γ)


"""
		k_imq(x,c,γ)
		k_imq(x,c)

		kernel matrix corresponding to the inverse multi-quadratic kernel ``imq = \\frac{C}{C + \\|X - Y\\|}`` kernel with `c` on diagonal
"""
k_imq(x,y,c) = sum( c./ (c .+ pairwisel2(x,y)))/(size(x,2) * size(y,2))
k_imq(x::T,c) where {T<:AbstractMatrix} = sum(c ./(c .+ pairwisel2(x)))/(size(x,2) * (size(x,2) -1 )) 
k_imq(x::T,c) where {T<:AbstractVector} = zero(eltype(x)) 

"""
		mmd_imq(x,y,γ)

		mmd with inverse polynomial kernel ``\\frac{C}{C + \\|X - Y\\|}`` used
		in Tolstikhin, Ilya, et al. "Wasserstein Auto-Encoders." arXiv preprint arXiv:1711.01558 (2017)

		`c` being equal to 2d σ2z, which is expected squared distance between two samples drawn from p(x).

"""
mmd_imq(x,y,c) = k_imq(x,c) + k_imq(y,c) - 2*k_imq(x,y,c)
mmd_imq(x,y,c,n) = mmd_imq(samplecolumns(x,n), samplecolumns(y,n), c)

"""

		rqfun(x)

		kernel from 
"""
rqfun(x,α) = (1 + (x + eltype(x)(1e-6))/2α)^-α 

"""
		k_rqh(x,y,α)
		k_rqh(x,α)

		kernel matrix with rq kernel
"""
k_rqh(x,y,α) = sum( 1 .+ rqfun.(pairwisel2(x,y),α))/((size(x,2) - 1) * size(y,2))
k_rqh(x::T,α) where {T<:AbstractMatrix} = sum( 1 .+ rqfun.(pairwisel2(x),α)) /(size(x,2) * (size(x,2) -1 )) 
k_rqh(x::T,α) where {T<:AbstractVector} = zero(eltype(x)) 

"""

	mmdrq(x,y,α)

	mmd with rq kernel

"""
mmdrq(x,y,α) = k_rqh(x,α) + k_rqh(y,α) - 2*k_rqh(x,y,α)
mmdrq(x,y,α,n) = mmdrq(samplecolumns(x,n), samplecolumns(y,n), α)
end # module
