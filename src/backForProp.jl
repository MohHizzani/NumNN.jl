
using ProgressMeter
using Random
using LinearAlgebra


"""
    perform the chained forward propagation using recursive calls

    input:
    X := input of the forward propagation
    oLayer := output layer
    cnt := an internal counter used to cache the layers was performed
           not to redo it again

    returns:
    A := the output of the last layer

    for internal use, it set again the values of Z and A in each layer
        to be used later in back propagation and add one to the layer
        forwCount value when pass through it
"""
function chainForProp(X, oLayer::Layer, cnt::Integer=-1)
    if cnt==-1
        cnt = oLayer.forwCount+1
    end

    if typeof(oLayer)!=AddLayer
        if oLayer.prevLayer==nothing
            actFun = oLayer.actFun
            W, B = oLayer.W, oLayer.B
            if oLayer.forwCount < cnt
                oLayer.forwCount += 1
                Z = W*X .+ B
                oLayer.Z = Z
                A = eval(:($actFun($Z)))
                oLayer.A = A
            else #if oLayer.forwCount < cnt
                A = oLayer.A
            end #if oLayer.forwCount < cnt
            return A
        else #if oLayer.prevLayer==nothing
            actFun = oLayer.actFun
            W, B = oLayer.W, oLayer.B
            prevLayer = oLayer.prevLayer
            if oLayer.forwCount < cnt
                oLayer.forwCount += 1
                Z = W*chainForProp(X, prevLayer, oLayer.forwCount) .+ B
                oLayer.Z = Z
                A = eval(:($actFun($Z)))
                oLayer.A = A
            else #if oLayer.forwCount < cnt
                A = oLayer.A
            end #if oLayer.forwCount < cnt
            return A
        end #if oLayer.prevLayer!=nothing
    else #if typeof(oLayer)!=AddLayer
        if oLayer.forwCount < cnt
            oLayer.forwCount += 1
            A = chainForProp(X, oLayer.prevLayer, oLayer.forwCount) .+
                chainForProp(X, oLayer.l2, oLayer.forwCount)
            oLayer.A = A
        else #if oLayer.forwCount < cnt
            A = oLayer.A
        end #if oLayer.forwCount < cnt

        return A
    end #if typeof(oLayer)!=AddLayer

end #function chainForProp

export chainForProp



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
    Ŷ = chainForProp(X, model.outLayer)
    T = eltype(Ŷ)
    c, m = size(Y)
    outLayer = model.outLayer
    # if isbool(Y)
    acc = 0
    if isequal(outLayer.actFun, :softmax)
        # Ŷ_bool = BitArray(undef, c, 0)
        maximums = maximum(Ŷ, dims=1)
        Ŷ_bool = Ŷ .== maximums
        T = eltype(Y)
        # Ŷ_bool = T.(Ŷ_bool)
        for i=1:m
            acc += (Ŷ_bool[:,i] == Y[:,i]) ? 1 : 0
        end
        acc /= m
        println("Accuracy is = $acc")
    end

    if isequal(outLayer.actFun, :σ)
        Ŷ_bool = Ŷ .> T(0.5)
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
    actFun = layers[L].actFun
    if ! isequal(actFun, :softmax) && !(actFun==:σ &&
                                        lossFun==:binaryCrossentropy)
        dA[L] = eval(:($dlossFun.($A[$L], $Y))) #.* eval(:($dActFun.(Z[L])))
    end
    if layers[L].keepProb < 1.0 #to save time of multiplication in case keepProb was one
        dA[L] = dA[L] .* D[L]
        dA[L] = dA[L] ./ layers[L].keepProb
    end
    for l=L:-1:2
        actFun = layers[l].actFun
        dActFun = Symbol("d",actFun)
        if l==L && (isequal(actFun, :softmax) ||
                    (actFun==:σ && lossFun==:binaryCrossentropy))
            dZ[l] = A[l] .- Y
        else
            dZ[l] = dA[l] .* eval(:($dActFun.($Z[$l])))
        end

        dW[l] = dZ[l]*A[l-1]' ./m

        if regulization > 0
            if regulization==1
                dW[l] .+= (λ/2m)
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
            dW[l] .+= (λ/2m)
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

function updateParams(model::Model, grads::Dict, tMiniBatch::Integer)
    W, B, V, S, layers = model.W, model.B, model.V, model.S, model.layers
    optimizer = model.optimizer
    dW, dB = grads["dW"], grads["dB"]
    L = length(layers)
    α = model.α
    β1, β2, ϵAdam = model.β1, model.β2, model.ϵAdam

    #initialize the needed variables to hold the corrected values
    #it is being init here cause these are not needed elsewhere
    VCorrected, SCorrected = deepInitVS(W,B,optimizer)
    if optimizer==:adam || optimizer==:momentum
        for i=1:L
            V[:dw][i] .= β1 .* V[:dw][i] .+ (1-β1) .* dW[i]
            V[:db][i] .= β1 .* V[:db][i] .+ (1-β1) .* dB[i]

            ##correcting
            VCorrected[:dw][i] .= V[:dw][i] ./ (1-β1^tMiniBatch)
            VCorrected[:db][i] .= V[:db][i] ./ (1-β1^tMiniBatch)

            if optimizer==:adam
                S[:dw][i] .= β2 .* S[:dw][i] .+ (1-β2) .* (dW[i].^2)
                S[:db][i] .= β2 .* S[:db][i] .+ (1-β2) .* (dB[i].^2)

                ##correcting
                SCorrected[:dw][i] .= S[:dw][i] ./ (1-β2^tMiniBatch)
                SCorrected[:db][i] .= S[:db][i] ./ (1-β2^tMiniBatch)

                ##update parameters with adam
                W[i] .-= (α .* (VCorrected[:dw][i] ./ (sqrt.(SCorrected[:dw][i]) .+ ϵAdam)))
                B[i] .-= (α .* (VCorrected[:db][i] ./ (sqrt.(SCorrected[:db][i]) .+ ϵAdam)))
            else#if optimizer==:momentum
                W[i] .-= (α .* VCorrected[:dw][i])
                B[i] .-= (α .* VCorrected[:db][i])
            end #if optimizer==:adam
        end #for i=1:L
    else
        W .-= (α .* dW)
        B .-= (α .* dB)
    end #if optimizer==:adam || optimizer==:momentum
    return W, B
end #updateParams

function updateParams!(model::Model, grads::Dict, tMiniBatch::Integer)
    W, B, V, S, layers = model.W, model.B, model.V, model.S, model.layers
    optimizer = model.optimizer
    dW, dB = grads["dW"], grads["dB"]
    L = length(layers)
    α = model.α
    β1, β2, ϵAdam = model.β1, model.β2, model.ϵAdam

    #initialize the needed variables to hold the corrected values
    #it is being init here cause these are not needed elsewhere
    VCorrected, SCorrected = deepInitVS(W,B,optimizer)
    if optimizer==:adam || optimizer==:momentum
        for i=1:L
            V[:dw][i] .= β1 .* V[:dw][i] .+ (1-β1) .* dW[i]
            V[:db][i] .= β1 .* V[:db][i] .+ (1-β1) .* dB[i]

            ##correcting
            VCorrected[:dw][i] .= V[:dw][i] ./ (1-β1^tMiniBatch)
            VCorrected[:db][i] .= V[:db][i] ./ (1-β1^tMiniBatch)

            if optimizer==:adam
                S[:dw][i] .= β2 .* S[:dw][i] .+ (1-β2) .* (dW[i].^2)
                S[:db][i] .= β2 .* S[:db][i] .+ (1-β2) .* (dB[i].^2)

                ##correcting
                SCorrected[:dw][i] .= S[:dw][i] ./ (1-β2^tMiniBatch)
                SCorrected[:db][i] .= S[:db][i] ./ (1-β2^tMiniBatch)

                ##update parameters with adam
                W[i] .-= (α .* (VCorrected[:dw][i] ./ (sqrt.(SCorrected[:dw][i]) .+ ϵAdam)))
                B[i] .-= (α .* (VCorrected[:db][i] ./ (sqrt.(SCorrected[:db][i]) .+ ϵAdam)))
            else#if optimizer==:momentum
                W[i] .-= (α .* VCorrected[:dw][i])
                B[i] .-= (α .* VCorrected[:db][i])
            end #if optimizer==:adam
        end #for i=1:L
    else
        W .-= (α .* dW)
        B .-= (α .* dB)
    end #if optimizer==:adam || optimizer==:momentum
    model.W, model.B = W, B
    return
end #updateParams!

export updateParams, updateParams!

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

            updateParams!(model, grads, j)
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

            updateParams!(model, grads, nB+1)
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
