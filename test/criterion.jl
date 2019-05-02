using IPMeasures, StatsBase
using IPMeasures: mmd, GaussianKernel, IMQKernel
using Plots
plotly()
k = IPMeasures.GaussianKernel(0.1)
x = randn(2,100);
y = rand(2,100);

function split2(x)
	n = size(x,2)
	x[:,1:div(n,2)],x[:,div(n,2)+1:end]
end

criterion(k, x, y) = mmd(k, x, y) / sqrt(abs(mmd(k, split2(x)...)) + abs(mmd(k, split2(y)...)))


using Plots
plotly()
k = IPMeasures.GaussianKernel(0.1)