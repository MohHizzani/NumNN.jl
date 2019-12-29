
using ProgressMeter
using Random
using LinearAlgebra

"""
    perform the forward propagation using

    input:
        x := (n0, m) matrix
        y := (c,  m) matrix where c is the number of classes

    return cache of A, Z, Yhat, Cost
"""
function forwardProp(X::Matrix{T},
                     Y::Matrix{T},
                     model::Model) where {T}

    W::AbstractArray{Matrix{T},1},
    B::AbstractArray{Matrix{T},1},
    layers::AbstractArray{Layer,1},
    lossFun = model.W, model.B, model.layers, model.lossFun
    regulization = model.regulization
    λ = model.λ
    m = size(X)[2]
    c = size(Y)[1]
    L = length(layers)
    A = Vector{Matrix{eltype(X)}}()
    Z = Vector{Matrix{eltype(X)}}()
    push!(Z, W[1]*X .+ B[1])
    actFun = layers[1].actFun
    push!(A, eval(:($actFun.($Z[1]))))
    for l=2:L
        push!(Z, W[l]*A[l-1] .+ B[l])
        actFun = layers[l].actFun

        if isequal(actFun, :softmax)
            a = Matrix{eltype(X)}(undef, c, 0)
            for i=1:m
                zCol = Z[l][:,i]
                a = hcat(a, eval(:($(actFun)($zCol))))
            end
            push!(A, a)
        else
            push!(A, eval(:($(actFun).($Z[$l]))))
        end
    end

    elCost = 0
    for i=1:size(Y)[2]
        a = A[L][:,i]
        y = Y[:,i]
        elCost += eval(:(sum($lossFun($a, $y)))) / c
    end
    cost = elCost/m

    #add the cost of regulization
    if regulization > 0
        cost += (λ/2m) * sum([norm(w, regulization) for w in W])
    end

    return Dict("A"=>A,
                "Z"=>Z,
                "Yhat"=>A[L],
                "Cost"=>cost)
end #forwardProp

export forwardProp


"""
    predict Y using the model and the input X and the labels Y
    inputs:
        model := the trained model
        X := the input matrix
        Y := the input labels to compare with

    output:
        a Dict of
        "Yhat" := the predicted values
        "Yhat_bools" := the predicted labels
        "accuracy" := the accuracy of the predicted labels
"""
function predict(model::Model, X, Y)
    Ŷ = forwardProp(X, Y, model)["Yhat"]
    T = eltype(Ŷ)
    layers = model.layers
    c, m = size(Y)
    # if isbool(Y)
    acc = 0
    if isequal(layers[end].actFun, :softmax)
        Ŷ_bool = BitArray(undef, c, 0)
        for v in eachcol(Ŷ)
            Ŷ_bool = hcat(Ŷ_bool, v .== maximum(v))
        end

        acc = sum([Ŷ_bool[:,i] == Y[:,i] for i=1:size(Y)[2]])/m
        println("Accuracy is = $acc")
    end

    if isequal(layers[end].actFun, :σ)
        Ŷ_bool = BitArray(undef, c, 0)
        for v in eachcol(Ŷ)
            Ŷ_bool = hcat(Ŷ_bool, v .> T(0.5))
        end

        acc = sum(Ŷ_bool .== Y)/(c*m)
        println("Accuracy is = $acc")
    end
    return Dict("Yhat"=>Ŷ,
                "Yhat_bool"=>Ŷ_bool,
                "accuracy"=>acc)
end #predict

export predict

"""
    return true if the array values are boolean (ones and zeros)
"""
function isbool(y::Array{T}) where {T}
    return iszero(y[y .!= one(T)])
end



"""

    inputs:
    X := is a (nx, m) matrix
    model contains these parameters
        W := is a Vector of Matrices of (n[l], n[l-1])
        B := is a Vector of Matrices of (n[l], 1)

"""
function backProp(X,Y,
                  model::Model,
                  cache::Dict{})

    layers::AbstractArray{Layer, 1} = model.layers
    lossFun = model.lossFun
    m = size(X)[2]
    L = length(layers)
    A, Z = cache["A"], cache["Z"]
    W, B, regulization, λ = model.W, model.B, model.regulization, model.λ

    D = [rand(size(A[i])...) .< layers[i].keepProb for i=1:L]

    if layers[L].keepProb < 1.0 #to save time of multiplication in case keepProb was one
        A = [A[i] .* D[i] for i=1:L]
        A = [A[i] ./ layers[i].keepProb for i=1:L]
    end
    # init all Arrays
    dA = Vector{Matrix{eltype(A[1])}}([similar(mat) for mat in A])
    dZ = Vector{Matrix{eltype(Z[1])}}([similar(mat) for mat in Z])
    dW = Vector{Matrix{eltype(W[1])}}([similar(mat) for mat in W])
    dB = Vector{Matrix{eltype(B[1])}}([similar(mat) for mat in B])

    dlossFun = Symbol("d",lossFun)
    dA[L] = eval(:($dlossFun.($A[$L], $Y))) #.* eval(:($dActFun.(Z[L])))

    if layers[L].keepProb < 1.0 #to save time of multiplication in case keepProb was one
        dA[L] = dA[L] .* D[L]
        dA[L] = dA[L] ./ layers[L].keepProb
    end
    for l=L:-1:2
        actFun = layers[l].actFun
        dActFun = Symbol("d",actFun)
        dZ[l] = dA[l] .* eval(:($dActFun.($Z[$l])))

        dW[l] = dZ[l]*A[l-1]' ./m

        if regulization > 0
            if regulization==1
                dW[l] .+= (λ/m)
            else
                dW[l] .+= (λ/m) .* W[l]
            end
        end
        dB[l] = 1/m .* sum(dZ[l], dims=2)
        dA[l-1] = W[l]'dZ[l]
        if layers[l-1].keepProb < 1.0 #to save time of multiplication in case keepProb was one
            dA[l-1] = dA[l-1] .* D[l-1]
            dA[l-1] = dA[l-1] ./ layers[l-1].keepProb
        end
    end

    l=1 #shortcut cause I just copied from the for loop
    actFun = layers[l].actFun
    dActFun = Symbol("d",actFun)
    dZ[l] = dA[l] .* eval(:($dActFun.($Z[$l])))

    dW[l] = 1/m .* dZ[l]*X' #where X = A[0]

    if regulization > 0
        if regulization==1
            dW[l] .+= (λ/m)
        else
            dW[l] .+= (λ/m) .* W[l]
        end
    end
    dB[l] = 1/m .* sum(dZ[l], dims=2)

    grads = Dict("dW"=>dW,
                 "dB"=>dB)
    return grads
end #backProp

export backProp

function updateParams(W, B, grads::Dict, α)
    dW, dB = grads["dW"], grads["dB"]
    W .-= (α .* dW)
    B .-= (α .* dB)
    return W, B
end #updateParams

export updateParams

"""
    Repeat the trainging for a single preceptron


    returns:
        W := Array of matrices of size (n[l], n[l-1])
        B := Array of matrices of size (n[l], 1)
"""
function train(X_train,
               Y_train,
               model::Model,
               epochs;
               batchSize = 64,
               printCostsInterval = 0,
               useProgBar = false,
               ϵ=10^-6)
    layers, lossFun, α, W, B = model.layers, model.lossFun, model.α, model.W, model.B
    Costs = []

    m = size(X_train)[2]
    nB = m ÷ batchSize
    shufInd = randperm(m)

    if useProgBar
        p = Progress(epochs, 1)
    end

    for i=1:epochs
        minCosts = [] #the costs of all mini-batches
        for j=1:nB
            downInd = (j-1)*batchSize+1
            upInd   = j * batchSize
            batchInd = shufInd[downInd:upInd]
            X = X_train[:, batchInd]
            Y = Y_train[:, batchInd]

            cache = forwardProp(X,
                                Y,
                                model)
            push!(minCosts, cache["Cost"])
            grads = backProp(X,Y,
                             model,
                             cache)

            model.W, model.B = updateParams(W, B, grads, α)
        end

        if m%batchSize != 0
            downInd = (nB)*batchSize+1
            batchInd = shufInd[downInd:end]
            X = X_train[:, batchInd]
            Y = Y_train[:, batchInd]

            cache = forwardProp(X,
                                Y,
                                model)
            push!(minCosts, cache["Cost"])
            grads = backProp(X,Y,
                             model,
                             cache)

            model.W, model.B = updateParams(W, B, grads, α)
        end

        push!(Costs, sum(minCosts)/length(minCosts))
        if printCostsInterval>0 && i%printCostsInterval==0
            println("N = $i, Cost = $(Costs[end])")
        end

        if useProgBar
            next!(p)
        end
    end

    # model.W, model.B = W, B
    return Dict("model"=>model,
                "Costs"=>Costs)
end #train

export train
