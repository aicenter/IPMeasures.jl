using IPMeasures, StatsBase, Distributions, LinearAlgebra
using IPMeasures: mmd, GaussianKernel, IMQKernel, pairwisel2

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

function crit_mmd2_var(k::GaussianKernel, X, Y)
    KXX = k.(pairwisel2(X,X))
    KXY = k.(pairwisel2(X,Y))
    KYY = k.(pairwisel2(Y,Y))
    m, v = mmd2_and_variance(KXX, KXY, KYY, true)
    m / sqrt(v)
end

function crit_mmd2_var(k::IMQKernel, X, Y)
    KXX = k.(pairwisel2(X,X))
    KXY = k.(pairwisel2(X,Y))
    KYY = k.(pairwisel2(Y,Y))
    m, v = mmd2_and_variance(KXX, KXY, KYY, false)
    m / sqrt(v)
end

function split2(x)
	n = size(x,2)
	x[:,1:div(n,2)], x[:,div(n,2)+1:end]
end

crit_mxy_over_mltpl(k, x, y) = mmd(k, x, y) / sqrt(abs(mmd(k, split2(x)...)*mmd(k, split2(y)...)))
crit_mxy_over_sum(k, x, y) = mmd(k, x, y) / (abs(mmd(k, split2(x)...)) + abs(mmd(k, split2(y)...)))