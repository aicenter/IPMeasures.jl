@testset "mmd2 and variance" begin
    o = ones(Float64, 3, 3)
    z = zeros(Float64, 3, 3)
    @test mmd2_and_variance(o, o, o) == (0, 0)
    @test mmd2_and_variance(z, z, z) == (0, 0)
    @test mmd2_and_variance(o, z, z) == (1, 0)
    @test mmd2_and_variance(z, z, o) == (1, 0)
    @test mmd2_and_variance(o, o, z) == (-1, 0)
    @test mmd2_and_variance(o, z, z) == (1, 0)
    @test mmd2_and_variance(z, o, z) == (-2, 0)

    a = [1 2 3; 4 5 6; 7 8 9]
    b = (a .+ 1)'
    c = (a .+ 2)'
    a = a'
    m, v = mmd2_and_variance(a, b, c)
    @test m ≈ 0
    @test v ≈ -3.1111111

    a = [1 1 1; 1 1 1; 0 1 0]'
    b = [1 1 1; 1 0 1; 0 1 0]'
    c = [0 1 1; 1 0 1; 0 1 0]'
    m, v = mmd2_and_variance(a, b, c)
    @test m ≈ 0.33333333
    @test v ≈ -0.0185185185185

    a = [0.31067363 0.17995522 0.69170826
        0.10956897 0.43168305 0.13072019
        0.98321756 0.66711701 0.4277054]'

    b = [0.6308643 0.72541977 0.26667574
        0.24141601 0.56088218 0.2872887 
        0.12801543 0.43956629 0.0703913]'

    c = [0.68779085 0.68050291 0.47736484
        0.49306397 0.07206057 0.10580517
        0.99297355 0.90194579 0.32241892]'

    m, v = mmd2_and_variance(a, b, c)
    @test m ≈ 0.32443064
    @test v ≈ 0.33610499
end
