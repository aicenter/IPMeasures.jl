[![Build Status](https://travis-ci.com/aicenter/IPMeasures.jl.svg?branch=master)](https://travis-ci.com/aicenter/IPMeasures.jl)
[![codecov](https://codecov.io/gh/aicenter/IPMeasures.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/aicenter/IPMeasures.jl)

# IPMeasures.jl

Implements Integral Probability Measures, such as Maximum Mean Discrepancy
(MMD) with Gaussian, RQ, and IPM kernels, as well as the KL divergence of
conditional Gaussians (based on 
[ConditionalDists.jl](https://github.com/aicenter/ConditionalDists.jl)). The
package is compatible with Flux.jl and uses the Distances.jl interface.

## Examples

Maximum Mean Discrepancy between `x` and `y` using gaussian kernel of bandwidth `γ`

```julia
using IPMeasures: mmd, GaussianKernel

x = randn(2,100)
y = randn(2,100)
γ = 1.0
mmd(GaussianKernel(γ),x,y)
0.012
```

`IMQKernel(c)` inverse multi-quadratic kernel ``k(d) = C/(C+d)`` with `d` being a
distance as used in [Tolstikhin, Ilya, et al. "Wasserstein
Auto-Encoders." (2017)](arXiv preprint arXiv:1711.01558)

```julia
using IPMeasures
import IPMeasures: mmd, IMQKernel
mmd(IMQKernel(1.0),randn(2,100),randn(2,100))
0.026
```

`RQKernel(α)` Maximum Mean Discrepancy between `x` and `y`  rq kernel from
Bińkowski, Mikołaj, et al. "Demystifying MMD GANs." (2018).

```julia
using IPMeasures
import IPMeasures: mmd, RQKernel
mmd(RQKernel(1.0),randn(2,100),randn(2,100))
0.026
```

Furthermore, we have estimation of Null Hypothesis of kernel `k` of samples `x`
from `n` random draws of subsets of size `l`
```
null_distribution(k::AbstractKernel, x, n, l)
```
estimates the null distribution 
