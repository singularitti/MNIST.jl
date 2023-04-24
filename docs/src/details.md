# Some details

Note that in the
[constructor of `Network`](https://github.com/singularitti/MNIST.jl/blob/eb836c46bd5cd0bc3dfd5d2dfaed1f36ce736666/src/types.jl#L19-L22),
we use `randn` instead of `rand`. This generates random numbers from a normal distribution
ranging from negative infinity to positive infinity. If we use `rand`, it will only generate
values from `0` to `1`. However, since the activation function used here is the sigmoid
function, the resulting `z` (which is pretty large) in the first hidden layer will be mapped
close to `1`, making the network converge very slowly.
