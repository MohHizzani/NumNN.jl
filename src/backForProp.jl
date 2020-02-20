
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


#TODO remove all tMiniBatch from update functions



"""
    do the back propagation for the output layer
"""
function outBackProp!(model::Model, Y, cnt::Integer; tMiniBatch::Integer=1)
    outLayer = model.outLayer
    prevLayer = outLayer.prevLayer
    lossFun = model.lossFun
    m = size(Y)[2]

    A, Z = outLayer.A, outLayer.Z

    regulization, λ = model.regulization, model.λ

    if outLayer.backCount < cnt
        outLayer.backCount += 1

        if outLayer.keepProb < 1.0 #to save memory and time
            D = rand(outLayer.numNodes,1) .< outLayer.keepProb
            outLayer.A = outLayer.A .* D
            outLayer.A = outLayer.A ./ outLayer.keepProb
        end


        dlossFun = Symbol("d",lossFun)
        actFun = outLayer.actFun

        outLayer.dZ = eval(:($dlossFun($A, $Y)))


        outLayer.dW = outLayer.dZ*prevLayer.A' ./m

        if regulization > 0
            if regulization==1
                outLayer.dW .+= (λ/2m)
            else
                outLayer.dW .+= (λ/m) .* outLayer.W
            end
        end




        outLayer.dB = 1/m .* sum(outLayer.dZ, dims=2)

        # updateParams!(model,
        #               outLayer,
        #               tMiniBatch)

    end #if outLayer.backCount < cnt



    return nothing
end #function outBackProp!

export outBackProp!


function backProp!(X::Array,
                   model::Model,
                   cLayer::Layer,
                   cnt::Integer;
                   tMiniBatch::Integer)

    outLayer = model.outLayer
    prevLayer = cLayer.prevLayer
    lossFun = model.lossFun
    keepProb = 1.0
    try
        keepProb = cLayer.keepProb
    catch

    end
    m = size(X)[2]
    A, Z = [], []
    try
        A, Z = cLayer.A, cLayer.Z
    catch

    end

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
        dA = []
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
    catch e
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

    # updateParams!(model,
    #               cLayer,
    #               tMiniBatch)

    end #if all(i->(i.backCount==cLayer.nextLayers[1].backCount), cLayer.nextLayers)

    return nothing
end

export backProp!

"""

    inputs:
    X := is a (nx, m) matrix
    Y := is a (c,  m) matrix
    model := is the model to perform the back propagation on
    cLayer := is an internal variable to hold the current layer
    cnt := is an internal variable to count the step of back propagation currently on

    output:
    nothing


"""
function chainBackProp!(X,Y,
                       model::Model,
                       cLayer::L=nothing,
                       cnt = -1;
                       tMiniBatch::Integer) where {L<:Union{Layer,Nothing}}
    if cnt < 0
        cnt = model.outLayer.backCount+1
    end

    if cLayer==nothing
        outBackProp!(model, Y, cnt, tMiniBatch=tMiniBatch)
        chainBackProp!(X,Y,model, model.outLayer.prevLayer, model.outLayer.backCount, tMiniBatch=tMiniBatch)

    elseif cLayer isa AddLayer
        backProp!(X, model, cLayer, cnt, tMiniBatch=tMiniBatch)
        for prevLayer in cLayer.prevLayer
            chainBackProp!(X,Y,model, prevLayer, cLayer.backCount, tMiniBatch=tMiniBatch)
        end #for
    else #if cLayer==nothing
        backProp!(X,model,cLayer, cnt, tMiniBatch=tMiniBatch)
        if cLayer.prevLayer != nothing
            chainBackProp!(X,Y,model, cLayer.prevLayer, cLayer.backCount, tMiniBatch=tMiniBatch)
        end #if cLayer.prevLayer == nothing
    end #if cLayer==nothing


end #backProp

export chainBackProp!



function updateParams!(model::Model,
                       cLayer::Layer,
                       cnt::Integer = -1;
                       tMiniBatch::Integer = 1)

    optimizer = model.optimizer
    α = model.α
    β1, β2, ϵAdam = model.β1, model.β2, model.ϵAdam

    if cLayer.updateCount >= cnt
        return nothing
    end #if cLayer.updateCount >= cnt

    cLayer.updateCount += 1
    #initialize the needed variables to hold the corrected values
    #it is being init here cause these are not needed elsewhere
    VCorrected = Dict(:dw=>similar(cLayer.dW), :db=>similar(cLayer.dB))
    SCorrected = Dict(:dw=>similar(cLayer.dW), :db=>similar(cLayer.dB))
    if optimizer==:adam || optimizer==:momentum

        cLayer.V[:dw] .= β1 .* cLayer.V[:dw] .+ (1-β1) .* cLayer.dW
        cLayer.V[:db] .= β1 .* cLayer.V[:db] .+ (1-β1) .* cLayer.dB

        ##correcting
        VCorrected[:dw] .= cLayer.V[:dw] ./ (1-β1^tMiniBatch)
        VCorrected[:db] .= cLayer.V[:db] ./ (1-β1^tMiniBatch)

        if optimizer==:adam
            cLayer.S[:dw] .= β2 .* cLayer.S[:dw] .+ (1-β2) .* (cLayer.dW.^2)
            cLayer.S[:db] .= β2 .* cLayer.S[:db] .+ (1-β2) .* (cLayer.dB.^2)

            ##correcting
            SCorrected[:dw] .= cLayer.S[:dw] ./ (1-β2^tMiniBatch)
            SCorrected[:db] .= cLayer.S[:db] ./ (1-β2^tMiniBatch)

            ##update parameters with adam
            cLayer.W .-= (α .* (VCorrected[:dw] ./ (sqrt.(SCorrected[:dw]) .+ ϵAdam)))
            cLayer.B .-= (α .* (VCorrected[:db] ./ (sqrt.(SCorrected[:db]) .+ ϵAdam)))

        else#if optimizer==:momentum

            cLayer.W .-= (α .* VCorrected[:dw])
            cLayer.B .-= (α .* VCorrected[:db])

        end #if optimizer==:adam
    else
        cLayer.W .-= (α .* cLayer.dW)
        cLayer.B .-= (α .* cLayer.dB)
    end #if optimizer==:adam || optimizer==:momentum

    return nothing
end #updateParams!

export updateParams!



function chainUpdateParams!(model::Model,
                           cLayer::L=nothing,
                           cnt = -1;
                           tMiniBatch::Integer = 1) where {L<:Union{Layer,Nothing}}

    if cnt < 0
        cnt = model.outLayer.updateCount + 1
    end


    if cLayer==nothing
        updateParams!(model, model.outLayer, cnt, tMiniBatch=tMiniBatch)
        chainUpdateParams!(model, model.outLayer.prevLayer, cnt, tMiniBatch=tMiniBatch)

    elseif cLayer isa AddLayer
        #update the AddLayer updateCounter
        if cLayer.updateCount < cnt
            cLayer.updateCount += 1
        end
        for prevLayer in cLayer.prevLayer
            chainUpdateParams!(model, prevLayer, cnt, tMiniBatch=tMiniBatch)
        end #for
    else #if cLayer==nothing
        updateParams!(model, cLayer, cnt, tMiniBatch=tMiniBatch)
        if cLayer.prevLayer != nothing
            chainUpdateParams!(model, cLayer.prevLayer, cnt, tMiniBatch=tMiniBatch)
        end #if cLayer.prevLayer == nothing
    end #if cLayer==nothing

    return nothing
end #function chainUpdateParams!


export chainUpdateParams!

"""
    Repeat the trainging (forward/backward propagation)

    inputs:
    X_train := the training input
    Y_train := the training labels
    model   := the model to train
    epochs  := the number of repetitions of the training phase
    ;
    kwarg:
    batchSize := the size of training when mini batch training
    printCostsIntervals := the interval (every what to print the current cost value)
    useProgBar := (true, false) value to use prograss bar


"""
function train(X_train,
               Y_train,
               model::Model,
               epochs;
               batchSize = 64,
               printCostsInterval = 0,
               useProgBar = false)

    outLayer, lossFun, α = model.outLayer, model.lossFun, model.α
    Costs = []

    m = size(X_train)[2]
    c = size(Y_train)[1]
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

            a = chainForProp(X,
                             model.outLayer)
            minCost = sum(eval(:($lossFun($a, $Y))))/batchSize

            if lossFun==:binaryCrossentropy
                minCost /= c
            end #if lossFun==:binaryCrossentropy

            push!(minCosts, minCost)


            chainBackProp!(X,Y,
                           model,
                           tMiniBatch = j)

            chainUpdateParams!(model; tMiniBatch = j)

        end #for j=1:nB iterate over the mini batches

        if m%batchSize != 0
            downInd = (nB)*batchSize+1
            batchInd = shufInd[downInd:end]
            X = X_train[:, batchInd]
            Y = Y_train[:, batchInd]

            a = chainForProp(X,
                             model.outLayer)

            minCost = sum(eval(:($lossFun($a, $Y))))/size(X)[2]

            if lossFun==:binaryCrossentropy
                minCost /= c
            end #if lossFun==:binaryCrossentropy

            push!(minCosts, minCost)

            chainBackProp!(X,Y,
                           model,
                           tMiniBatch = nB+1)

            chainUpdateParams!(model; tMiniBatch = nB+1)
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
    return Costs
end #train

export train
