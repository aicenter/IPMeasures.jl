export GaussianKernel, IMQKernel, RQKernel

abstract type AbstractKernel end

function kernelsum(k::AbstractKernel, x::AbstractMatrix, y::AbstractMatrix, dist::MetricOrFun)
    sum(k(dist(x,y))) / (size(x,2) * size(y,2))
end

function kernelsum(k::AbstractKernel, x::AbstractMatrix{T}, dist::MetricOrFun) where T
    l = size(x,2)
    (sum(k(dist(x,x))) - l*k(T(0)))/(l^2 - l)
end

kernelsum(k::AbstractKernel, x::AbstractVector, dist::MetricOrFun) = zero(eltype(x))

"""
	GaussianKernel(γ<:Number)

	implements the standard Gaussian kernel ``exp(-γ * x)
"""
struct GaussianKernel{T<:Number} <: AbstractKernel
	γ::T
end

(m::GaussianKernel)(x::Number) = exp(-m.γ * x)
(m::GaussianKernel)(x::AbstractArray) = exp.(-m.γ .* x)


"""
	ImqKernel(c)

Inverse polynomial kernel ``\\frac{C}{C + x}`` used in Tolstikhin, Ilya, et al.
"Wasserstein Auto-Encoders." arXiv preprint arXiv:1711.01558 (2017)

`c` being equal to 2d σ2z, which is expected squared distance between two
samples drawn from p(x).
"""
struct IMQKernel{T<:Number} <: AbstractKernel
	c::T
end

(m::IMQKernel)(x::Number) = m.c/ (m.c + x)
(m::IMQKernel)(x::AbstractArray) = m.c ./ (m.c .+ x)

"""
	RQ kernel ``\\frac{C}{C + \\|X - Y\\|}`` used
	in Tolstikhin, Ilya, et al. "Wasserstein Auto-Encoders." arXiv preprint arXiv:1711.01558 (2017)

	`c` being equal to 2d σ2z, which is expected squared distance between two samples drawn from p(x).

"""
struct RQKernel{T<:Number} <: AbstractKernel
	α::T
end

(m::RQKernel)(x::Number) = (1 + (x + eps(x))/2m.α)^-m.α 
(m::RQKernel)(x::AbstractArray) = (1 .+(x .+ eps(eltype(x))) ./ (2m.α) ).^-m.α 
