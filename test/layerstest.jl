include("../src/includes.jl")

# using .NumNN

x = rand(64*64,1000);

Y = rand(0:4, 1000);
Y = oneHot(Y, classes = [0,1,2,3,4])
#
X = chain(x, [FCLayer(120, :relu),  FCLayer(5, :softmax)])
# X = FCLayer(120, :relu)(x)
# X = FCLayer(20, :relu)(X)
# # X = FCLayer(20, :tanh)(X1)
# # X = AddLayer()([X, X2])
# X = FCLayer(5, :softmax)(X)


# X1 = FCLayer(20, :relu)(x)
# X = FCLayer(20, :tanh)(X1)
# X = AddLayer()([X,X1])
# X = FCLayer(5, :softmax)(X)
# deepInitWB!(x, X)

# resetCount!(X, :forwCount)

# deepInitVS!(X, :adam)

mo = Model(x, Y, X, 0.01; optimizer = :adam, lossFun = :categoricalCrossentropy)

# Ŷ_pred = predict(mo, x, Y)
#
# chainBackProp!(x,Y,mo)

costs = train(x, Y, mo, 100, batchSize=64, useProgBar=true, printCostsInterval = 500)

using Plots

gr()

plot(1:length(costs), costs)

Ŷ_pred = predict(mo, x, Y)

Ŷ = chainForProp(x, X)

x1 = randn(64*64, 5);

ŷ1 = chainForProp(x1, X);

X = FCLayer(120, :relu)(x)

X = FCLayer(5, :softmax)(X)

mo1 = Model(x, Y, X, 0.01; optimizer = :adam)

ŷ_oneLayer = chainForProp(x, X);

ŷ_oneLayer_pred = predict(mo1, x, Y)
