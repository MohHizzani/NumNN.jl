
# using Images#, ImageView
using MLDatasets
# using .NumNN
include("../src/includes.jl")
# using ProgressMeter
# gr()

X_train_org, Y_train_org = FashionMNIST.traindata(Integer)
X_train1, X_train2, X_train_m = size(X_train_org)
X_test_org, Y_test_org = FashionMNIST.testdata(Integer)
X_test1, X_test2, X_test_m = size(X_test_org)
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

X_train = reshape(X_train_org, (X_train1*X_train2, X_train_m))
println(size(X_train))
m = size(X_train)[2]
X_train = X_train ./ 255
X_test = reshape(X_test_org, (X_test1*X_test2, X_test_m))
X_test = X_test ./ 255
Y_train = oneHot(Y_train_org)
Y_train = eltype(X_train).(Y_train)
Y_test = oneHot(Y_test_org)
Y_test = eltype(X_test).(Y_test)
outLayer = chain(X_train, [FCLayer(128, :relu), FCLayer(10, :softmax)])
# W, B = deepInitWB(X_train, Y_train, layers)
model = Model(X_train, Y_train, outLayer, 0.001;
               lossFun=:categoricalCrossentropy,
               regulization=0,
               optimizer = :adam,
               λ=1.0);


trcache = train(X_train, Y_train, model, 10, batchSize = 256, useProgBar = true)

pred = predict(model, X_train, Y_train)

using Plots
gr()
plot(1:length(trcache), trcache)
