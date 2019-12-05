export KLDivergence, kl_divergence
import Distances: KLDivergence, kl_divergence

function _kld_gaussian(μ1::AbstractArray, σ1::AbstractArray, μ2::AbstractArray, σ2::AbstractArray)
    k  = size(μ1, 1)
    m1 = sum(σ1 ./ σ2, dims=1)
    m2 = sum((μ2 .- μ1).^2 ./ σ2, dims=1)
    m3 = sum(log.(σ2 ./ σ1), dims=1)
    (m1 .+ m2 .+ m3 .- k) ./ 2
end

function (m::KLDivergence)(p::Gaussian, q::Gaussian)
    (μ1, σ1) = mean_var(p)
    (μ2, σ2) = mean_var(q)
    _kld_gaussian(μ1, σ1, μ2, σ2)
end

kl_divergence(p::Gaussian, q::Gaussian) = KLDivergence()(p, q)

function (m::KLDivergence)(p::AbstractCGaussian{T}, q::Gaussian{T}, z::AbstractArray{T}) where T
    (μ1, σ1) = mean_var(p, z)
    (μ2, σ2) = mean_var(q)
    _kld_gaussian(μ1, σ1, μ2, σ2)
end

kl_divergence(p::AbstractCGaussian, q::Gaussian, z::AbstractArray) = KLDivergence()(p,q,z)
