# IPMeasures

Implements Integral Probability Measures, at the moment Maximum Mean Discrepancy with Gaussian, RQ, and IPM kernel. The package is made such that it is compatible with Flux.

`mmdg(x,y,γ,n)` Maximum Mean Discrepancy between `x` and `y` using gaussian kernel of bandwidth `γ`

Example
```
using IPMeasures
import IPMeasures: mmdg
mmdg(randn(2,100),randn(2,100), 1.0)
0.012
```


`mmd_imq(x,y,c,n)` Maximum Mean Discrepancy between `x` and `y`  inverse multi-quadratic kernel k(x,y) = \frac{C}{C + \|X - Y\|} used in Tolstikhin, Ilya, et al. "Wasserstein Auto-Encoders." arXiv preprint arXiv:1711.01558 (2017)

Example
```
using IPMeasures
import IPMeasures: mmd_imq
mmd_imq(randn(2,100),randn(2,100), 1.0)
0.026
```

`mmd_rq(x,y,c,n)` Maximum Mean Discrepancy between `x` and `y`  rq kernel from Bińkowski, Mikołaj, et al. "Demystifying MMD GANs." (2018).

Example
```
using IPMeasures
import IPMeasures: mmd_rq
mmdrq(randn(2,100),randn(2,100), 1.0)
0.026
```
