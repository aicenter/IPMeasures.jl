@testset "kl_divergence" begin
    p = Gaussian(zeros(2), ones(2))
    q = Gaussian(zeros(2), ones(2))
    @test all(kl_divergence(p, q) .≈ 0.0)

    cp = CMeanGaussian{Float64,DiagVar}(identity, ones(2))

    kld = kl_divergence(cp, q, zeros(2,10))
    @test size(kld) == (1,10)
    @test all(kld .≈ 0.0)
end
