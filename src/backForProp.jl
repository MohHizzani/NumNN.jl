
using ProgressMeter
using Random
using LinearAlgebra

include("layerForProp.jl")

"""
    perform the chained forward propagation using recursive calls

    input:
    X := input of the forward propagation
    cLayer := output layer
    cnt := an internal counter used to cache the layers was performed
           not to redo it again

    returns:
    A := the output of the last layer

    for internal use, it set again the values of Z and A in each layer
        to be used later in back propagation and add one to the layer
        forwCount value when pass through it
"""
function chainForProp!(X, cLayer::Layer, cnt::Integer=-1)
    if cnt<0
        cnt = cLayer.forwCount+1
    end

    if cLayer isa Input
        if cLayer.forwCount < cnt
            layerForProp!(cLayer, X)
        end #if cLayer.forwCount < cnt
        return nothing
    elseif isa(cLayer, AddLayer) #if typeof(cLayer)==AddLayer
        if cLayer.forwCount < cnt
            for prevLayer in cLayer.prevLayer
                chainForProp!(X, prevLayer, cnt)
            end
            layerForProp!(cLayer)
        end #if cLayer.forwCount < cnt

        return nothing
    else #if cLayer.prevLayer==nothing
        if cLayer.forwCount < cnt
            chainForProp!(X, cLayer.prevLayer, cnt)
            layerForProp!(cLayer)
        end #if cLayer.forwCount < cnt
        return nothing
    end #if cLayer.prevLayer!=nothing

end #function chainForProp!

export chainForProp!



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
function predict(
    model::Model,
    X::Array,
    Y=nothing,
    )
    chainForProp!(X, model.outLayer)
    Ŷ = model.outLayer.A
    T = eltype(Ŷ)
    outLayer = model.outLayer
    actFun = outLayer.actFun
    # if isbool(Y)
    return predict(eval(:($actFun)), Ŷ, labels=Y)

end #predict

export predict

"""
    return true if the array values are boolean (ones and zeros)
"""
function isbool(y::Array{T}) where {T}
    return iszero(y[y .!= one(T)])
end



### back propagation

include("layerBackProp.jl")



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
                       tMiniBatch::Integer=-1, #can be used to perform both back and update params
                       ) where {L<:Union{Layer,Nothing}}
    if cnt < 0
        cnt = model.outLayer.backCount+1
    end

    if cLayer==nothing
        layerBackProp!(model.outLayer, model, labels=Y)

        if tMiniBatch > 0
            layerUpdateParams!(model, model.outLayer, cnt; tMiniBatch=tMiniBatch)
        end

        chainBackProp!(X,Y,model, model.outLayer.prevLayer, cnt; tMiniBatch=tMiniBatch)

    elseif cLayer isa AddLayer
        layerBackProp!(cLayer, model)

        if tMiniBatch > 0
            layerUpdateParams!(model, model.outLayer, cnt; tMiniBatch=tMiniBatch)
        end

        if !(cLayer.backCount < cnt) #in case layerBackProp did not do the back
                                     #prop becasue the next layers are not all
                                     #done yet
            for prevLayer in cLayer.prevLayer
                chainBackProp!(X,Y,model, prevLayer, cnt; tMiniBatch=tMiniBatch)
            end #for
        end
    else #if cLayer==nothing
        layerBackProp!(cLayer, model)

        if tMiniBatch > 0
            layerUpdateParams!(model, model.outLayer, cnt; tMiniBatch=tMiniBatch)
        end

        if !(cLayer.backCount < cnt)#in case layerBackProp did not do the back
                                     #prop becasue the next layers are not all
                                     #done yet
            if !(cLayer isa Input)
                chainBackProp!(X,Y,model, cLayer.prevLayer, cnt; tMiniBatch=tMiniBatch)
            end #if cLayer.prevLayer == nothing
        end
    end #if cLayer==nothing

    return nothing
end #backProp

export chainBackProp!


###update parameters

include("layerUpdateParams.jl")





function chainUpdateParams!(model::Model,
                           cLayer::L=nothing,
                           cnt = -1;
                           tMiniBatch::Integer = 1) where {L<:Union{Layer,Nothing}}

    if cnt < 0
        cnt = model.outLayer.updateCount + 1
    end


    if cLayer==nothing
        layerUpdateParams!(model, model.outLayer, cnt, tMiniBatch=tMiniBatch)
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
        layerUpdateParams!(model, cLayer, cnt, tMiniBatch=tMiniBatch)
        if !(cLayer isa Input)
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
function train(
               X_train,
               Y_train,
               model::Model,
               epochs;
               batchSize = 64,
               printCostsInterval = 0,
               useProgBar = false,
               embedUpdate = true,
               )

    outLayer, lossFun, α = model.outLayer, model.lossFun, model.α
    Costs = []

    m = size(X_train)[end]
    c = size(Y_train)[end-1]
    nB = m ÷ batchSize
    shufInd = randperm(m)
    N = ndims(X_train)
    axX = axes(X_train)[1:end-1]
    axY = axes(Y_train)[1:end-1]
    if useProgBar
        p = Progress(epochs, 1)
    end

    for i=1:epochs
        minCosts = [] #the costs of all mini-batches
        for j=1:nB
            downInd = (j-1)*batchSize+1
            upInd   = j * batchSize
            batchInd = shufInd[downInd:upInd]
            X = X_train[axX..., batchInd]
            Y = Y_train[axY..., batchInd]

            chainForProp!(X,
                          model.outLayer)

            a = model.outLayer.A
            minCost = sum(eval(:($lossFun($a, $Y))))/batchSize

            if lossFun==:binaryCrossentropy
                minCost /= c
            end #if lossFun==:binaryCrossentropy

            push!(minCosts, minCost)

            if embedUpdate
                chainBackProp!(X,Y,
                               model;
                               tMiniBatch = j)
            else
                chainBackProp!(X,Y,
                               model;
                               tMiniBatch = -1)

                chainUpdateParams!(model; tMinitBatch = j)
            end #if embedUpdate

            # chainUpdateParams!(model; tMiniBatch = j)

        end #for j=1:nB iterate over the mini batches

        if m%batchSize != 0
            downInd = (nB)*batchSize+1
            batchInd = shufInd[downInd:end]
            X = X_train[axX..., batchInd]
            Y = Y_train[axY..., batchInd]

            chainForProp!(X,
                         model.outLayer)


            a = model.outLayer.A
            minCost = sum(eval(:($lossFun($a, $Y))))/size(X)[2]

            if lossFun==:binaryCrossentropy
                minCost /= c
            end #if lossFun==:binaryCrossentropy

            push!(minCosts, minCost)

            if embedUpdate
                chainBackProp!(X,Y,
                               model;
                               tMiniBatch = nB+1)
            else
                chainBackProp!(X,Y,
                               model;
                               tMiniBatch = -1)

                chainUpdateParams!(model; tMinitBatch = nB+1)
            end #if embedUpdate

            # chainUpdateParams!(model; tMiniBatch = nB+1)
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
