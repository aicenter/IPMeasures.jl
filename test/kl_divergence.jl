@testset "kl_divergence" begin
    p = MvNormal(zeros(2), ones(2))
    q = MvNormal(zeros(2), ones(2))
    @test all(kl_divergence(p, q) .≈ 0.0)

    cp = ConditionalMvNormal(identity)

    kld = kl_divergence(cp, q, zeros(2,10))
    @test size(kld) == (1,10)
    @test all(kld .≈ 0.0)
end
