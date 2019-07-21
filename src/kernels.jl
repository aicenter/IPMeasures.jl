abstract type AbstractKernel end;

"""
	GaussianKernel(γ)

	implements the standard Gaussian kernel ``exp(-γ * x)
"""
struct GaussianKernel{T} <: AbstractKernel
	γ::T
end

(m::GaussianKernel)(x::Number) = exp(-m.γ * x)
(m::GaussianKernel)(x::AbstractArray) = exp.(-m.γ .* x)


"""
	ImqKernel(c)

	inverse polynomial kernel ``\\frac{C}{C + x}`` used
	in Tolstikhin, Ilya, et al. "Wasserstein Auto-Encoders." arXiv preprint arXiv:1711.01558 (2017)

	`c` being equal to 2d σ2z, which is expected squared distance between two samples drawn from p(x).

"""
struct IMQKernel{T} <: AbstractKernel
	c::T
end

(m::IMQKernel)(x::Number) = m.c/ (m.c + x)
(m::IMQKernel)(x::AbstractArray) = m.c ./ (m.c .+ x)

"""
	RQ kernel ``\\frac{C}{C + \\|X - Y\\|}`` used
	in Tolstikhin, Ilya, et al. "Wasserstein Auto-Encoders." arXiv preprint arXiv:1711.01558 (2017)

	`c` being equal to 2d σ2z, which is expected squared distance between two samples drawn from p(x).

"""
struct RQKernel{T} <: AbstractKernel
	α::T
end

(m::RQKernel)(x::Number) = (1 + (x + eps(x))/2m.α)^-m.α 
(m::RQKernel)(x::AbstractArray) = (1 .+(x .+ eps(eltype(x))) ./ (2m.α) ).^-m.α 


struct SumOfKernels{T<:Tuple} <: AbstractKernel
	ks::T 
end
(m::SumOfKernels)(x) = sum(k(x) for k in m.ks)