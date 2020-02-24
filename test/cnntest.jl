include("../src/includes.jl")

x = rand(1:255, 10,10,3,10) #10x10x3 images 10 of them
Y = rand(0:4, 10)
Y = oneHot(Y; classes = [0,1,2,3,4])

####create some Layers

X = Conv2D(10, (3,3))(x)
X = Activation(:relu)(X)
X = Conv2D(20, (3,3))(X)
X = Conv2D(20, (1,1))(X)
#### test initialization

deepInitWB!(x, X; dtype=Float64)
