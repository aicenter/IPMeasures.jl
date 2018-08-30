# IPMeasures

Implements Integral Probability Measures, at the moment Maximum Mean Discrepancy with Gaussian, RQ, and IPM kernel. The package is made such that it is compatible with Flux.

`mmd(GaussianKernel(γ),x,y,γ,n)` Maximum Mean Discrepancy between `x` and `y` using gaussian kernel of bandwidth `γ`

Example
```
using IPMeasures
import IPMeasures: mmd, GaussianKernel
mmd(GaussianKernel(1.0),randn(2,100),randn(2,100))
0.012
```


`IMQKernel(c)` inverse multi-quadratic kernel ``k(d) = \frac{C}{C + d}`` with d being a distance used in Tolstikhin, Ilya, et al. "Wasserstein Auto-Encoders." arXiv preprint arXiv:1711.01558 (2017)

Example
```
using IPMeasures
import IPMeasures: mmd, IMQKernel
mmd(IMQKernel(1.0),randn(2,100),randn(2,100))
0.026
```

`RQKernel(α)` Maximum Mean Discrepancy between `x` and `y`  rq kernel from Bińkowski, Mikołaj, et al. "Demystifying MMD GANs." (2018).

Example
```
using IPMeasures
import IPMeasures: mmd, RQKernel
mmd(RQKernel(1.0),randn(2,100),randn(2,100))
0.026
```


Furthermore, we ha estimation of Null Hypothesis of kernel `k` of samples `x` from `n` random draws of subsets of size `l`
```
null_distribution(k::AbstractKernel, x, n, l)
```

		estimates the null distribution 
