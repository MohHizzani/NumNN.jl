
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
function chainForProp!(X, cLayer::Layer, cnt::Integer=-1; kwargs...)
    if cnt<0
        cnt = cLayer.forwCount+1
    end

    if cLayer isa Input
        if cLayer.forwCount < cnt
            layerForProp!(cLayer, X; kwargs...)
        end #if cLayer.forwCount < cnt
        return nothing
    elseif isa(cLayer, AddLayer) #if typeof(cLayer)==AddLayer
        if cLayer.forwCount < cnt
            for prevLayer in cLayer.prevLayer
                chainForProp!(X, prevLayer, cnt; kwargs...)
            end
            layerForProp!(cLayer; kwargs...)
        end #if cLayer.forwCount < cnt

        return nothing
    else #if cLayer.prevLayer==nothing
        if cLayer.forwCount < cnt
            chainForProp!(X, cLayer.prevLayer, cnt; kwargs...)
            layerForProp!(cLayer; kwargs...)
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
function predictBatch(
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
    return probToValue(eval(:($actFun)), Ŷ, labels=Y)

end #predict


function predict(
    model::Model,
    X_In::AbstractArray,
    Y_In=nothing;
    batchSize = 32,
    printAcc = true,
    useProgBar = false,
    GCInt = 5,
    noBool = false,
    )

    outLayer, lossFun, α = model.outLayer, model.lossFun, model.α
    Costs = []


    N = ndims(X_In)
    T = eltype(X_In)
    m = size(X_In)[end]
    # c = size(Y_train)[end-1]
    nB = m ÷ batchSize
    N = ndims(X_In)
    axX = axes(X_In)[1:end-1]
    Y = nothing
    if Y_In != nothing
        axY = axes(Y_In)[1:end-1]
    end
    if useProgBar
        p = Progress((m % batchSize != 0 ? nB+1 : nB), 0.1)
    end

    Ŷ_out = Array{T,N}(undef,repeat([0],N)...)
    accuracy = []
    @simd for j=1:nB
        downInd = (j-1)*batchSize+1
        upInd   = j * batchSize
        X = X_In[axX..., downInd:upInd]
        if Y_In != nothing
            Y = Y_In[axY..., downInd:upInd]
        end
        Ŷ, acc = predictBatch(model, X, Y)
        # Ŷ_out = cat(Ŷ_out, Ŷ, dims=1:N)
        if acc != nothing
            push!(accuracy, acc)
        end
        if useProgBar
            update!(p, j, showvalues=[("Instances $m", j*batchSize)])
        end #if useProgBar
        # X = Y = nothing
        Ŷ = nothing
        # if j%GCInt == 0
        #     Base.GC.gc()
        # end
    end

    if m % batchSize != 0
        downInd = (nB)*batchSize+1
        X = X_In[axX..., downInd:end]
        if Y_In != nothing
            Y = Y_In[axY..., downInd:end]
        end
        Ŷ, acc = predict(model, X, Y)

        Ŷ_out = cat(Ŷ_out, Ŷ, dims=1:N)
        if acc != nothing
            push!(accuracy, acc)
        end
        # X = Y = nothing
        Ŷ = nothing
        # Base.GC.gc()
        update!(p, nB+1, showvalues=[("Instances $m", m)])
    end

    accuracyM = nothing
    if !(isempty(accuracy))
        accuracyM = mean(accuracy)
    end
    if noBool
        Ŷ_out = nothing
    end

    return Ŷ_out, accuracyM

end #function predict(
    # model::Model,
    # X_In::AbstractArray,
    # Y_In=nothing;
    # batchSize = 32,
    # printAcc = true,
    # useProgBar = false,
    # )

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
                       kwargs...,
                       ) where {L<:Union{Layer,Nothing}}
    if cnt < 0
        cnt = model.outLayer.backCount+1
    end

    if cLayer==nothing
        layerBackProp!(model.outLayer, model; labels=Y, kwargs...)

        if tMiniBatch > 0
            layerUpdateParams!(model, model.outLayer, cnt; tMiniBatch=tMiniBatch, kwargs...)
        end

        chainBackProp!(X,Y,model, model.outLayer.prevLayer, cnt; tMiniBatch=tMiniBatch, kwargs...)

    elseif cLayer isa AddLayer
        layerBackProp!(cLayer, model; kwargs...)

        if tMiniBatch > 0
            layerUpdateParams!(model, cLayer, cnt; tMiniBatch=tMiniBatch, kwargs...)
        end

        if cLayer.backCount >= cnt #in case layerBackProp did not do the back
                                     #prop becasue the next layers are not all
                                     #done yet
            for prevLayer in cLayer.prevLayer
                chainBackProp!(X,Y,model, prevLayer, cnt; tMiniBatch=tMiniBatch, kwargs...)
            end #for
        end
    else #if cLayer==nothing
        layerBackProp!(cLayer, model; kwargs...)

        if tMiniBatch > 0
            layerUpdateParams!(model, cLayer, cnt; tMiniBatch=tMiniBatch, kwargs...)
        end

        if cLayer.backCount >= cnt #in case layerBackProp did not do the back
                                     #prop becasue the next layers are not all
                                     #done yet
            if !(cLayer isa Input)
                chainBackProp!(X,Y,model, cLayer.prevLayer, cnt; tMiniBatch=tMiniBatch, kwargs...)
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
               metrics::Array{Symbol,1} = [:accuracy, :cost],
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
    accuracy = []
    avgCost = 0.0
    avgAcc = 0.0
    for i=1:epochs
        minCosts = [] #the costs of all mini-batches
        minAcc = []
        for j=1:nB
            downInd = (j-1)*batchSize+1
            upInd   = j * batchSize
            batchInd = shufInd[downInd:upInd]
            X = X_train[axX..., batchInd]
            Y = Y_train[axY..., batchInd]

            if :accuracy in metrics
                ŷ, acc = predict(model, X, Y; noBool = true, batchSize = batchSize)
                push!(minAcc, acc)
                avgAcc = mean([avgAcc, acc])
            else
                chainForProp!(X,
                              model.outLayer)
            end

            a = model.outLayer.A
            if :cost in metrics
                minCost = sum(eval(:($lossFun($a, $Y))))/batchSize

                if lossFun==:binaryCrossentropy
                    minCost /= c
                end #if lossFun==:binaryCrossentropy

                push!(minCosts, minCost)
                avgCost = mean([avgCost, minCost])
            end

            if embedUpdate
                chainBackProp!(X,Y,
                               model;
                               tMiniBatch = j)
            else
                chainBackProp!(X,Y,
                               model;
                               tMiniBatch = -1)

                chainUpdateParams!(model; tMiniBatch = j)
            end #if embedUpdate
            if useProgBar
                if :accuracy in metrics && :cost in metrics
                    update!(p, i; showvalues=[(:Epoch, i), ("Instances ($m)", j*batchSize), (:Accuracy, avgAcc), (:Cost, avgCost)])
                elseif :accuracy in metrics
                    update!(p, i; showvalues=[(:Epoch, i), ("Instances ($m)", j*batchSize), (:Accuracy, avgAcc)])
                elseif :cost in metrics
                    update!(p, i; showvalues=[(:Epoch, i), ("Instances ($m)", j*batchSize), (:Cost, avgCost)])
                else
                    update!(p, i; showvalues=[(:Epoch, i), ("Instances ($m)", j*batchSize)])
                end
            end
            # chainUpdateParams!(model; tMiniBatch = j)

        end #for j=1:nB iterate over the mini batches

        if m%batchSize != 0
            downInd = (nB)*batchSize+1
            batchInd = shufInd[downInd:end]
            X = X_train[axX..., batchInd]
            Y = Y_train[axY..., batchInd]

            if :accuracy in metrics
                ŷ, acc = predict(model, X, Y; noBool = true, batchSize = size(X)[end])
                push!(minAcc, acc)
                avgAcc = mean([avgAcc, acc])

            else
                chainForProp!(X,
                              model.outLayer)
            end

            if :cost in metrics
                a = model.outLayer.A
                minCost = sum(eval(:($lossFun($a, $Y))))/size(X)[end]

                if lossFun==:binaryCrossentropy
                    minCost /= c
                end #if lossFun==:binaryCrossentropy

                push!(minCosts, minCost)
            end

            if embedUpdate
                chainBackProp!(X,Y,
                               model;
                               tMiniBatch = nB+1)
            else
                chainBackProp!(X,Y,
                               model;
                               tMiniBatch = -1)

                chainUpdateParams!(model; tMiniBatch = nB+1)
            end #if embedUpdate

            # chainUpdateParams!(model; tMiniBatch = nB+1)
        end

        push!(accuracy, mean(minAcc))
        push!(Costs, mean(minCosts))
        if printCostsInterval>0 && i%printCostsInterval==0
            println("N = $i, Cost = $(Costs[end])")
        end

        if useProgBar
            if useProgBar
                if :accuracy in metrics && :cost in metrics
                    update!(p, i; showvalues=[(:Epoch, i), ("Instances ($m)", m), (:Accuracy, avgAcc), (:Cost, avgCost)])
                elseif :accuracy in metrics
                    update!(p, i; showvalues=[(:Epoch, i), ("Instances ($m)", j*batchSize), (:Accuracy, avgAcc)])
                elseif :cost in metrics
                    update!(p, i; showvalues=[(:Epoch, i), ("Instances ($m)", j*batchSize), (:Cost, avgCost)])
                else
                    update!(p, i; showvalues=[(:Epoch, i), ("Instances ($m)", j*batchSize)])
                end
            end
        end
    end

    # model.W, model.B = W, B
    return Costs
end #train

export train
