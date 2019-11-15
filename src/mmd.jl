export MMD, mmd, mmd2_and_variance

struct MMD <: PreMetric
    kernel::AbstractKernel
    dist::MetricOrFun
end


function (m::MMD)(x::AbstractArray, y::AbstractArray)
    xx = kernelsum(m.kernel, x, m.dist)
    yy = kernelsum(m.kernel, y, m.dist)
    xy = kernelsum(m.kernel, x, y, m.dist)
    xx + yy - 2xy
end


pairwisel2(x,y) = pairwise(SqEuclidean(), x, y)

"""
	mmd(AbstractKernel(γ), x, y)
	mmd(AbstractKernel(γ), x, y, n)

MMD with Gaussian kernel of bandwidth `γ` using at most `n` samples
"""
mmd(k::AbstractKernel, x::AbstractArray, y::AbstractArray, dist=pairwisel2) =
    MMD(k, dist)(x, y)
mmd(k::AbstractKernel, x::AbstractArray, y::AbstractArray, n::Int, dist=pairwisel2) =
    mmd(k, samplecolumns(x,n), samplecolumns(y,n), dist)


"""
    mmd2_and_variance(K_XX, K_XY, K_YY, unit_diagonal=false, biased=false)

calculates the mmd and variance according to https://arxiv.org/pdf/1611.04488.pdf
"""
function mmd2_and_variance(K_XX, K_XY, K_YY, unit_diagonal=false, biased=false)
    @assert size(K_XX, 2) == size(K_XY, 2) == size(K_XX, 2)
    m = size(K_XX, 2)  # Assumes X, Y are same shape

    ### Get the various sums of kernels that we'll use
    # Kts drop the diagonal, but we don't need to compute them explicitly
    diag_X = diag_Y = 1
    sum_diag_X = sum_diag_Y = m
    sum_diag2_X = sum_diag2_Y = m
    if !unit_diagonal
        diag_X = diag(K_XX)
        diag_Y = diag(K_YY)

        sum_diag_X = sum(diag_X)
        sum_diag_Y = sum(diag_Y)

        sum_diag2_X = dot(diag_X,diag_X)
        sum_diag2_Y = dot(diag_Y,diag_Y)
    end

    Kt_XX_sums = sum(K_XX, dims = 1)[:] - diag_X
    Kt_YY_sums = sum(K_YY, dims = 1)[:] - diag_Y
    K_XY_sums_0 = sum(K_XY; dims = 2)
    K_XY_sums_1 = sum(K_XY, dims = 1)[:]

    Kt_XX_sum = sum(Kt_XX_sums)
    Kt_YY_sum = sum(Kt_YY_sums)
    K_XY_sum = sum(K_XY_sums_0)

    Kt_XX_2_sum = sum(K_XX.^2) - sum_diag2_X
    Kt_YY_2_sum = sum(K_YY.^2) - sum_diag2_Y
    K_XY_2_sum  = sum(K_XY.^2)

    mmd2 = if biased
        ((Kt_XX_sum + sum_diag_X) / (m * m)
              + (Kt_YY_sum + sum_diag_Y) / (m * m)
              - 2 * K_XY_sum / (m * m))
    else
        (Kt_XX_sum / (m * (m-1))
              + Kt_YY_sum / (m * (m-1))
              - 2 * K_XY_sum / (m * m))
    end

    var_est = (
          2 / (m^2 * (m-1)^2) * (
              2 * dot(Kt_XX_sums,Kt_XX_sums) - Kt_XX_2_sum
            + 2 * dot(Kt_YY_sums, Kt_YY_sums) - Kt_YY_2_sum)
        - (4*m-6) / (m^3 * (m-1)^3) * (Kt_XX_sum^2 + Kt_YY_sum^2)
        + 4*(m-2) / (m^3 * (m-1)^2) * (
              dot(K_XY_sums_1, K_XY_sums_1)
            + dot(K_XY_sums_0, K_XY_sums_0))
        - 4 * (m-3) / (m^3 * (m-1)^2) * K_XY_2_sum
        - (8*m - 12) / (m^5 * (m-1)) * K_XY_sum^2
        + 8 / (m^3 * (m-1)) * (
              1/m * (Kt_XX_sum + Kt_YY_sum) * K_XY_sum
            - dot(Kt_XX_sums, K_XY_sums_1)
            - dot(Kt_YY_sums, K_XY_sums_0))
    )

    (mmd2, var_est)
end

"""
    crit_mmd2_var(k, X, Y)
    crit_mmd2_var(k, X, Y, n)

Calculates the criterion for selection of the width of the kernel based on
https://arxiv.org/pdf/1611.04488.pdf using at most `n` samples.
"""
crit_mmd2_var(k::GaussianKernel, X, Y, n::Int, dist = pairwisel2) =
    crit_mmd2_var(k, samplecolumns(X, n), samplecolumns(Y, n), dist)

function crit_mmd2_var(k::GaussianKernel, X, Y, dist = pairwisel2)
    KXX = k.(dist(X,X))
    KXY = k.(dist(X,Y))
    KYY = k.(dist(Y,Y))
    m, v = mmd2_and_variance(KXX, KXY, KYY, true)
    m / sqrt(max(v, eps(Float64)))
end

crit_mmd2_var(k::IMQKernel, X, Y, n::Int, dist = pairwisel2) =
    crit_mmd2_var(k, samplecolumns(X, n), samplecolumns(Y, n), dist)

function crit_mmd2_var(k::IMQKernel, X, Y, dist = pairwisel2)
    KXX = k.(dist(X,X))
    KXY = k.(dist(X,Y))
    KYY = k.(dist(Y,Y))
    m, v = mmd2_and_variance(KXX, KXY, KYY, false)
    m / sqrt(max(v, eps(Float64)))
end


"""
    crit_mxy_over_mltpl(k, x, y)

Calculates criterion for selection of the kernel width as MMD(X, Y) / SQRT(MMD(X,X) * MMD(Y, Y))
"""
crit_mxy_over_mltpl(k, x, y, dist = pairwisel2) =
    mmd(k, x, y, dist) / sqrt(abs(mmd(k, split2(x)..., dist) * mmd(k, split2(y)..., dist)))

"""
    crit_mxy_over_sum(k, x, y)

Calculates criterion for selection of the kernel width as MMD(X, Y) / (MMD(X,X) + MMD(Y, Y))
"""
crit_mxy_over_sum(k, x, y, dist = pairwisel2) =
    mmd(k, x, y, dist) / (abs(mmd(k, split2(x)..., dist)) + abs(mmd(k, split2(y)..., dist)))

function mmdfromdist(k::AbstractKernel, d, idx, cidx)
	li, lc = length(idx), length(cidx)
	kxx = (mapsum(k, d, idx) - li*k(0.0))/(li^2 - li)
	kyy = (mapsum(k, d, cidx) - lc*k(0.0))/(lc^2 - lc)
	kyx = mapsum(k, d, idx, cidx)/ (lc*li)
	kxx + kyy -2kyx
end


mapsum(f, d, idx) = (s = 0.0; @inbounds for j in idx, i in idx s+= f(d[i,j]) end; s)
mapsum(f, d, rowidx, colidx) =
    (s = 0.0; @inbounds for i in rowidx, j in colidx s+= f(d[i,j]) end; s)
