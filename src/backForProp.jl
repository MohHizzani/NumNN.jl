
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
    if cnt<0
        cnt = oLayer.forwCount+1
    end

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
    elseif isa(oLayer, AddLayer) #if typeof(oLayer)==AddLayer
        if oLayer.forwCount < cnt
            oLayer.forwCount += 1
            A = chainForProp(X, oLayer.prevLayer[1], oLayer.forwCount)
            for prevLayer in oLayer.prevLayer[2:end]
                A .+=  chainForProp(X, prevLayer, oLayer.forwCount)
            end
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
    do the back propagation for the output layer
"""
function outBackProp!(model::Model, Y, cnt::Integer)
    outLayer = model.outLayer
    prevLayer = outLayer.prevLayer
    lossFun = model.lossFun
    m = size(Y)[2]
    # A, Z = cache["A"], cache["Z"]
    A, Z = outLayer.A, outLayer.Z
    # dZ, dW, dB = outLayer.dZ, outLayer.dW, outLayer.dB
    regulization, λ = model.regulization, model.λ

    if outLayer.backCount < cnt
        outLayer.backCount += 1

        if outLayer.keepProb < 1.0 #to save memory and time
            D = rand(outLayer.numNodes,1) .< outLayer.keepProb
            outLayer.A = outLayer.A .* D
            outLayer.A = outLayer.A ./ outLayer.keepProb
        end
        # init all Arrays
        # dA = zeros(eltype(A), size(A)...)
        # dZ = zeros(eltype(A), size(A)...)

        dlossFun = Symbol("d",lossFun)
        actFun = outLayer.actFun

        outLayer.dZ = eval(:($dlossFun($A, $Y))) #.* eval(:($dActFun.(Z[L])))

        # if outLayer.keepProb < 1.0 #to save time of multiplication in case keepProb was one
        #     dA = dA .* D
        #     dA = dA ./ outLayer.keepProb
        # end

        outLayer.dW = outLayer.dZ*outLayer.A' ./m

        if regulization > 0
            if regulization==1
                outLayer.dW .+= (λ/2m)
            else
                outLayer.dW .+= (λ/m) .* outLayer.W
            end
        end
        outLayer.dB = 1/m .* sum(outLayer.dZ, dims=2)
    end #if outLayer.backCount < cnt

    return nothing
end #function outBackProp!

export outBackProp!


function backProp!(X::Array,
                   model::Model,
                   cLayer::Layer,
                   cnt::Integer)

    outLayer = model.outLayer
    prevLayer = cLayer.prevLayer
    lossFun = model.lossFun
    try
        global keepProb = cLayer.keepProb
    catch

    end
    m = size(X)[2]
    # A, Z = cache["A"], cache["Z"]
    try
        global A, Z = cLayer.A, cLayer.Z
    catch

    end
    # dZ, dW, dB = outLayer.dZ, outLayer.dW, outLayer.dB
    regulization, λ = model.regulization, model.λ

    ## in case nothing has finished yet of the next layer(s)
    if cLayer.backCount >= cnt
        return nothing
    end

    if all(i->(i.backCount==cLayer.nextLayers[1].backCount), cLayer.nextLayers)
        if cLayer isa AddLayer
            for nextLayer in cLayer.nextLayers
                try
                    cLayer.dZ .+= nextLayer.W'nextLayer.dZ
                catch e
                    cLayer.dZ = nextLayer.W'nextLayer.dZ
                end #tyr/catch
            end #for
            cLayer.backCount += 1
            return nothing
        end #if cLayer isa AddLayer
        for nextLayer in cLayer.nextLayers
            try
                dA .+= nextLayer.W'nextLayer.dZ
            catch e
                try
                    dA = nextLayer.W'nextLayer.dZ
                catch e1
                    dA = nextLayer.dZ
                end #try/catch
            end #try/catch
        end #for
        if keepProb < 1.0 #to save time of multiplication in case keepProb was one
           D = rand(cLayer.numNodes,1) .< keepProb
           dA = dA .* D
           dA = dA ./ keepProb
        end
    dActFun = Symbol("d",cLayer.actFun)

    cLayer.dZ = dA .* eval(:($dActFun($Z)))

    try ##in case it is the input layer
        cLayer.dW = cLayer.dZ*cLayer.prevLayer.A' ./m
    catch
        cLayer.dW = cLayer.dZ*X' ./m
    end #try/cathc

    if regulization > 0
        if regulization==1
            cLayer.dW .+= (λ/2m)
        else
            cLayer.dW .+= (λ/m) .* cLayer.W
        end
    end

    cLayer.dB = 1/m .* sum(cLayer.dZ, dims=2)

    cLayer.backCount += 1
    end #if all(i->(i.backCount==cLayer.nextLayers[1].backCount), cLayer.nextLayers)
    return nothing
end

export backProp!

"""

    inputs:
    X := is a (nx, m) matrix
    model contains these parameters
        W := is a Vector of Matrices of (n[l], n[l-1])
        B := is a Vector of Matrices of (n[l], 1)

"""
function chainBackProp!(X,Y,
                       model::Model,
                       cLayer::L=nothing,
                       cnt = -1) where {L<:Union{Layer,Nothing}}
    if cnt < 0
        cnt = model.outLayer.backCount+1
    end

    if cLayer==nothing
        outBackProp!(model, Y, cnt)
        chainBackProp!(X,Y,model, model.outLayer.prevLayer, model.outLayer.backCount)

    elseif cLayer isa AddLayer
        backProp!(X, model, cLayer, cnt)
        for prevLayer in cLayer.prevLayer
            chainBackProp!(X,Y,model, prevLayer, cLayer.backCount)
        end #for
    else #if cLayer==nothing
        backProp!(X,model,cLayer, cnt)
        if cLayer.prevLayer != nothing
            chainBackProp!(X,Y,model, cLayer.prevLayer, cLayer.backCount)
        end #if cLayer.prevLayer == nothing
    end #if cLayer==nothing


end #backProp

export chainBackProp!



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

export updateParams!

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
