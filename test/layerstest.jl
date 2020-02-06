using NumNN

x = rand(20,5);

Y = rand(0:4, 5);
Y = oneHot(Y, classes = [0,1,2,3,4])

X1 = NumNN.FCLayer(20, :relu)(x)
X = FCLayer(20, :tanh)(X1)
X = AddLayer(X, X1)
X = FCLayer(5, :softmax)(X)

mo = Model(x, Y, X, 0.01; optimizer = :adam)
