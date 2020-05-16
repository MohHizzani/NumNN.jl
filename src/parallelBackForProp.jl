include("parallelLayerForProp.jl")


using ProgressMeter

try
    ProgressMeter.ijulia_behavior(:clear)
catch

end
using Random
using LinearAlgebra

###

@doc raw"""
    function chainForProp(
        X::AbstractArray{T,N},
        cLayer::Layer,
        cnt::Integer = -1;
        FCache = Dict{Layer,Dict{Symbol,AbstractArray}}(),
        kwargs...,
    ) where {T,N}

perform the chained forward propagation using recursive calls

# Arguments:

- `X::AbstractArray{T,N}` := input of the input layer

- `cLayer::Layer` := Input Layer

- `cnt::Integer` := an internal counter used to cache the layers was performed not to redo it again

# Returns

- `Cache::Dict{Layer, Dict{Symbol, Array}}` := the output each layer either A, Z or together As Dict of layer to dict of Symbols and Arrays for internal use, it set again the values of Z and A in each layer to be used later in back propagation and add one to the layer forwCount value when pass through it
"""
function chainForProp(
    X::AbstractArray{T,N},
    cLayer::Layer,
    cnt::Integer = -1;
    FCache = Dict{Layer,Dict{Symbol,AbstractArray}}(),
    Done = Dict{Layer, Bool}(),
    kwargs...,
) where {T,N}
    if cnt < 0
        # cnt = cLayer.forwCount + 1
        cnt = 1
    end

    if length(cLayer.nextLayers) == 0
        # if cLayer.forwCount < cnt
        if haskey(Done, cLayer.prevLayer) && Done[cLayer.prevLayer]
            FCache[cLayer] = layerForProp(cLayer; FCache = FCache, Done = Done, kwargs...)
        end #if cLayer.forwCount < cnt
        return FCache
    elseif isa(cLayer, MILayer) #if typeof(cLayer)==AddLayer
        if all(
            i -> (haskey(Done,i) && Done[i]),
            cLayer.prevLayer,
        )
            FCache[cLayer] = layerForProp(cLayer; FCache = FCache, Done = Done, kwargs...)
            for nextLayer in cLayer.nextLayers
                FCache =
                    chainForProp(X, nextLayer, cnt; FCache = FCache, Done = Done, kwargs...)
            end
        end #if all

        return FCache
    else #if cLayer.prevLayer==nothing
        # if cLayer.forwCount < cnt
        if haskey(Done, cLayer.prevLayer) && Done[cLayer.prevLayer]
            if cLayer isa Input
                FCache[cLayer] =
                    layerForProp(cLayer, X; FCache = FCache, Done = Done, kwargs...)
            else
                FCache[cLayer] =
                    layerForProp(cLayer; FCache = FCache, Done = Done, kwargs...)
            end
            for nextLayer in cLayer.nextLayers
                FCache =
                    chainForProp(X, nextLayer, cnt; FCache = FCache, Done = Done, kwargs...)
            end

        end #if cLayer.forwCount < cnt
        return FCache
    end #if cLayer.prevLayer!=nothing

    return FCache

end #function chainForProp!

export chainForProp


@doc raw"""
    predictBatch(model::Model, X::AbstractArray, Y = nothing; kwargs...)

predict Y using the model and the input X and the labels Y

# Inputs

- `model::Model` := the trained model

- `X::AbstractArray` := the input `Array`

- `Y` := the input labels to compare with (optional)

# Output

- a `Tuple` of

    * `Ŷ` := the predicted values
    * `Ŷ_bool` := the predicted labels
    * `"accuracy"` := the accuracy of the predicted labels


"""
function predictBatch(model::Model, X::AbstractArray, Y = nothing; kwargs...)

    kwargs = Dict{Symbol, Any}(kwargs...)
    kwargs[:prediction] = getindex(kwargs, :prediction, default = true)

    FCache = chainForProp(X, model.inLayer; kwargs...)
    Ŷ = FCache[model.outLayer][:A]
    T = eltype(Ŷ)
    outLayer = model.outLayer
    actFun = outLayer.actFun
    lossFun = model.lossFun
    costs = nothing
    if Y != nothing
        costs = cost(eval(:($lossFun)), Ŷ, Y)
    end
    # if isbool(Y)
    return Ŷ, probToValue(eval(:($actFun)), Ŷ; labels = Y)..., costs

end #predict

export predictBatch


###
@doc raw"""
    predict(model::Model, X_In::AbstractArray, Y_In = nothing; kwargs...)

Run the prediction based on the trained `model`

# Arguments

- `model::Model` := the trained `Model` to predict on

- `X_In` := the input `Array`

- `Y_In` := labels (optional) to evaluate the model

## Key-word Arugmets

- `batchSize` := default `32`

- `useProgBar` := (`Bool`) where or not to shoe the prograss bar

# Return

- a `Dict` of:

    * `:YhatValue` := Array of the output of the integer prediction values
    * `:YhatProb` := Array of the output probabilities
    * `:accuracy` := the accuracy of prediction in case `Y_In` is given
"""
function predict(model::Model, X_In::AbstractArray, Y_In = nothing; kwargs...)

    kwargs = Dict{Symbol, Any}(kwargs...)
    batchSize = getindex(kwargs, :batchSize; default = 32)
    printAcc = getindex(kwargs, :printAcc; default = true)
    useProgBar = getindex(kwargs, :useProgBar; default = true)
    GCInt = getindex(kwargs, :GCInt, default = 5)
    noBool = getindex(kwargs, :noBool, default = false)
    kwargs[:prediction] = getindex(kwargs, :prediction, default = true)

    outLayer, lossFun, α = model.outLayer, model.lossFun, model.α
    Costs = []


    nX = ndims(X_In)
    T = eltype(X_In)
    m = size(X_In)[end]
    # c = size(Y_train)[end-1]
    nB = m ÷ batchSize
    axX = axes(X_In)[1:end-1]
    Y = nothing
    if Y_In != nothing
        axY = axes(Y_In)[1:end-1]
        costs =
            Array{AbstractFloat,1}(undef, nB + ((m % batchSize == 0) ? 0 : 1))
    end
    if useProgBar
        p = Progress((m % batchSize != 0 ? nB + 1 : nB), 0.1)
    end

    nY = length(model.outLayer.outputS)
    Ŷ_out =
        Array{AbstractArray{T,nY},1}(undef, nB + ((m % batchSize == 0) ? 0 : 1))
    Ŷ_prob_out =
        Array{AbstractArray{T,nY},1}(undef, nB + ((m % batchSize == 0) ? 0 : 1))
    accuracy =
        Array{AbstractFloat,1}(undef, nB + ((m % batchSize == 0) ? 0 : 1))
    Threads.@threads for j = 1:nB
    # @simd for j = 1:nB
        downInd = (j - 1) * batchSize + 1
        upInd = j * batchSize
        X = view(X_In, axX..., downInd:upInd)
        if Y_In != nothing
            Y = view(Y_In, axY..., downInd:upInd)
        end
        Ŷ_prob_out[j], Ŷ_out[j], accuracy[j], costs[j] =
            predictBatch(model, X, Y; kwargs...)

        if useProgBar
            update!(p, j, showvalues = [("Instances $m", j * batchSize)])
        end #if useProgBar
        # X = Y = nothing
        # Ŷ = nothing
        # if j%GCInt == 0
        #     Base.GC.gc()
        # end
    end

    if m % batchSize != 0
        downInd = (nB) * batchSize + 1
        X = view(X_In, axX..., downInd:m)
        if Y_In != nothing
            Y = view(Y_In, axY..., downInd:m)
        end
        Ŷ_prob_out[nB+1], Ŷ_out[nB+1], accuracy[nB+1], costs[nB+1] =
            predictBatch(model, X, Y; kwargs...)

        # X = Y = nothing
        # Base.GC.gc()
        if useProgBar
            update!(p, nB + 1, showvalues = [("Instances $m", m)])
        end
    end

    # accuracyM = nothing
    # if !(isempty(accuracy))
    accuracyM = mean(filter(x -> x != nothing, accuracy))
    costM = nothing
    if Y_In != nothing
        costM = mean(filter(x -> x != nothing, costs))
    end
    # end
    if noBool
        Ŷ_values = nothing
        Ŷ_prob = nothing
    else
        # Ŷ_values = Array{T,nY}(undef, repeat([0], nY)...)
        Ŷ_values = cat(Ŷ_out... ; dims = nY)
        Ŷ_prob = cat(Ŷ_prob_out...; dims = nY)
    end

    return Dict(
        :YhatValue => Ŷ_values,
        :YhatProb => Ŷ_prob,
        :accuracy => accuracyM,
        :cost => costM,
    )

end #function predict(
# model::Model,
# X_In::AbstractArray,
# Y_In=nothing;
# batchSize = 32,
# printAcc = true,
# useProgBar = false,
# )

export predict


### chainBackProp
include("parallelLayerBackProp.jl")
# include("layerUpdateParams.jl")


@doc raw"""
    function chainBackProp(
        X::AbstractArray{T1,N1},
        Y::AbstractArray{T2,N2},
        model::Model,
        FCache::Dict{Layer,Dict{Symbol,AbstractArray}},
        cLayer::L = nothing,
        BCache::Dict{Layer,Dict{Symbol,AbstractArray}}=Dict{Layer,Dict{Symbol,AbstractArray}}(),
        cnt = -1;
        tMiniBatch::Integer = -1, #can be used to perform both back and update params
        kwargs...,
    ) where {L<:Union{Layer,Nothing},T1,T2,N1,N2}

# Arguments

- `X` := train data

- `Y` := train labels

- `model` := is the model to perform the back propagation on

- `FCache` := the cached values of the forward propagation as `Dict{Layer, Dict{Symbol, AbstractArray}}`

- `cLayer` := is an internal variable to hold the current layer

- `BCache` := to hold the cache of the back propagtion (internal variable)

- `cnt` := is an internal variable to count the step of back propagation currently on to avoid re-do it

## Key-word Arguments

- `tMiniBatch` := to perform both the back prop and update trainable parameters in the same recursive call (if less than 1 update during back propagation is ditched)

- `kwargs` := other key-word arguments to be bassed to `layerBackProp` methods

# Return

- `BCache` := the cached values of the back propagation


"""
function chainBackProp(
    X::AbstractArray{T1,N1},
    Y::AbstractArray{T2,N2},
    model::Model,
    FCache::Dict{Layer,Dict{Symbol,AbstractArray}},
    cLayer::L = nothing,
    BCache::Dict{Layer,Dict{Symbol,AbstractArray}}=Dict{Layer,Dict{Symbol,AbstractArray}}(),
    cnt = -1;
    tMiniBatch::Integer = -1, #can be used to perform both back and update params
    kwargs...,
) where {L<:Union{Layer,Nothing},T1,T2,N1,N2}
    if cnt < 0
        cnt = model.outLayer.backCount + 1
    end

    if cLayer == nothing
        cLayer = model.outLayer
        BCache[cLayer] =
            layerBackProp(cLayer, model, FCache, BCache; labels = Y, kwargs...)

        if tMiniBatch > 0
            layerUpdateParams!(
                model,
                cLayer,
                cnt;
                tMiniBatch = tMiniBatch,
                kwargs...,
            )
        end

        BCache = chainBackProp(
            X,
            Y,
            model,
            FCache,
            cLayer.prevLayer,
            BCache,
            cnt;
            tMiniBatch = tMiniBatch,
            kwargs...,
        )

    elseif cLayer isa MILayer
        BCache[cLayer] = layerBackProp(cLayer, model, FCache, BCache; kwargs...)

        if tMiniBatch > 0
            layerUpdateParams!(
                model,
                cLayer,
                cnt;
                tMiniBatch = tMiniBatch,
                kwargs...,
            )
        end

        if cLayer.backCount >= cnt #in case layerBackProp did not do the back
            #prop becasue the next layers are not all
            #done yet
            for prevLayer in cLayer.prevLayer
                BCache = chainBackProp(
                    X,
                    Y,
                    model,
                    FCache,
                    prevLayer,
                    BCache,
                    cnt;
                    tMiniBatch = tMiniBatch,
                    kwargs...,
                )
            end #for
        end
    else #if cLayer==nothing
        BCache[cLayer] = layerBackProp(cLayer, model, FCache, BCache; kwargs...)

        if tMiniBatch > 0
            layerUpdateParams!(
                model,
                cLayer,
                cnt;
                tMiniBatch = tMiniBatch,
                kwargs...,
            )
        end

        if cLayer.backCount >= cnt #in case layerBackProp did not do the back
            #prop becasue the next layers are not all
            #done yet
            if !(cLayer isa Input)
                BCache = chainBackProp(
                    X,
                    Y,
                    model,
                    FCache,
                    cLayer.prevLayer,
                    BCache,
                    cnt;
                    tMiniBatch = tMiniBatch,
                    kwargs...,
                )
            end #if cLayer.prevLayer == nothing
        end
    end #if cLayer==nothing

    return BCache
end #backProp

export chainBackProp

###update parameters

include("layerUpdateParams.jl")




@doc raw"""
    chainUpdateParams!(model::Model,
                       cLayer::L=nothing,
                       cnt = -1;
                       tMiniBatch::Integer = 1) where {L<:Union{Layer,Nothing}}

Update trainable parameters using recursive call

# Arguments

- `model` := the model holds the training and update process

- `cLayer` := internal variable for recursive call holds the current layer

- `cnt` := an internal variable to hold the count of update in each layer not to re-do it

## Key-word Arguments

- `tMiniBatch` := the number of mini-batch of the total train collection

# Return

- `nothing`

"""
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

    elseif cLayer isa MILayer
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

### train

@doc raw"""
    train(
          X_train,
          Y_train,
          model::Model,
          epochs;
          testData = nothing,
          testLabels = nothing,
          kwargs...,
          )

Repeat the trainging (forward/backward propagation and update parameters)

# Argument

- `X_train` := the training data

- `Y_train` := the training labels

- `model`   := the model to train

- `epochs`  := the number of repetitions of the training phase

# Key-word Arguments

- `testData` := to evaluate the training process over test data too

- `testLabels` := to evaluate the training process over test data too

- `batchSize` := the size of training when mini batch training

` `useProgBar` := (true, false) value to use prograss bar

- `kwargs` := other key-word Arguments to pass for the lower functions in hierarchy

# Return

- A `Dict{Symbol, Vector}` of:

    * `:trainAccuracies` := an `Array` of the accuracies of training data at each epoch
    * `:trainCosts` := an `Array` of the costs of training data at each epoch
    * In case `testDate` and `testLabels` are givens:

        + `:testAccuracies` := an `Array` of the accuracies of test data at each epoch
        + `:testCosts` := an `Array` of the costs of test data at each epoch

"""
function train(
               X_train,
               Y_train,
               model::Model,
               epochs;
               testData = nothing,
               testLabels = nothing,
               kwargs...,
               )


    kwargs = Dict{Symbol, Any}(kwargs...)
    batchSize = getindex(kwargs, :batchSize; default = 32)
    printCostsInterval = getindex(kwargs, :printCostsInterval; default = 0)
    useProgBar = getindex(kwargs, :useProgBar; default = true)
    embedUpdate = getindex(kwargs, :embedUpdate; default = true)
    metrics = getindex(kwargs, :metrics; default = [:accuracy, :cost])
    Accuracy = :accuracy in metrics
    Cost = :cost in metrics
    test = testData != nothing && testLabels != nothing

    inLayer, outLayer, lossFun, α = model.inLayer, model.outLayer, model.lossFun, model.α

    outAct = outLayer.actFun
    m = size(X_train)[end]
    c = size(Y_train)[end-1]
    nB = m ÷ batchSize

    N = ndims(X_train)
    axX = axes(X_train)[1:end-1]
    axY = axes(Y_train)[1:end-1]
    nMiniBatch = (nB + Integer(m % batchSize != 0))
    if useProgBar
        p = Progress(epochs*nMiniBatch, 0.1)
        showValues = Dict{String, Real}("Epoch $(epochs)"=>0,
                                        "Instances $(m)"=>0,
                                        )
    end



    if Accuracy
        showValues["Train Accuracy"] = 0.0
    end
    if Cost
        showValues["Train Cost"] = 0.0
    end

    if test
        testAcc = []
        testCost = []
        showValues["Test Accuracy"] = 0.0
        showValues["Test Cost"] = 0.0
    end


    Accuracies = []
    Costs = []
    for i=1:epochs
        shufInd = randperm(m)
        minCosts = [] #the costs of all mini-batches
        minAcc = []
        for j=1:nB
            downInd = (j-1)*batchSize+1
            upInd   = j * batchSize
            batchInd = shufInd[downInd:upInd]
            X = X_train[axX..., batchInd]
            Y = Y_train[axY..., batchInd]

            FCache = chainForProp(X, inLayer; kwargs...)
            a = FCache[outLayer][:A]
            if Accuracy
                _, acc = probToValue(eval(:($outAct)), a; labels = Y)
                push!(minAcc, acc)
                if useProgBar
                    tmpAcc = mean(minAcc)
                    showValues["Train Accuracy"] = round(tmpAcc ; digits=4)
                end
            end


            if Cost
                minCost = cost(eval(:($lossFun)), a, Y)
                push!(minCosts, minCost)
                if useProgBar
                    tmpCost = mean(minCosts)
                    showValues["Train Cost"] = round(tmpCost; digits = 4)
                end
            end

            if embedUpdate
                BCache = chainBackProp(X,Y,
                                       model,
                                       FCache;
                                       tMiniBatch = j,
                                       kwargs...)
            else
                BCache = chainBackProp(X,Y,
                                      model,
                                      FCache;
                                      kwargs...)

                chainUpdateParams!(model; tMiniBatch = j)
            end #if embedUpdate
            if useProgBar
                showValues["Epoch $(epochs)"] = i
                showValues["Instances $(m)"] = j*batchSize
                update!(p, (i-1)*nMiniBatch+j; showvalues=[("Epoch $(epochs)", pop!(showValues, "Epoch $(epochs)")), ("Instances $(m)", pop!(showValues, "Instances $(m)")), showValues...])
            end
                # if :accuracy in metrics && :cost in metrics
                #     update!(p, ((i-1)*(nB + ((m % batchSize == 0) ? 0 : 1)))+j; showvalues=[("Epoch ($epochs)", i), ("Instances ($m)", j*batchSize), (:Accuracy, round(mean(minAcc); digits=4)), (:Cost, round(mean(minCosts); digits=4))])
                # elseif :accuracy in metrics
                #     update!(p, ((i-1)*(nB + ((m % batchSize == 0) ? 0 : 1)))+j; showvalues=[("Epoch ($epochs)", i), ("Instances ($m)", j*batchSize), (:Accuracy, round(mean(minAcc); digits=4))])
                # elseif :cost in metrics
                #     update!(p, ((i-1)*(nB + ((m % batchSize == 0) ? 0 : 1)))+j; showvalues=[("Epoch ($epochs)", i), ("Instances ($m)", j*batchSize), (:Cost, round(mean(minCosts); digits=4))])
                # else
                #     update!(p, ((i-1)*(nB + ((m % batchSize == 0) ? 0 : 1)))+j; showvalues=[("Epoch ($epochs)", i), ("Instances ($m)", j*batchSize)])
                # end
            # end
            # chainUpdateParams!(model; tMiniBatch = j)

        end #for j=1:nB iterate over the mini batches

        if m%batchSize != 0
            downInd = (nB)*batchSize+1
            batchInd = shufInd[downInd:end]
            X = X_train[axX..., batchInd]
            Y = Y_train[axY..., batchInd]

            FCache = chainForProp(X, inLayer; kwargs...)
            if Accuracy
                _, acc = probToValue(eval(:($outAct)), FCache[outLayer][:A]; labels = Y)
                push!(minAcc, acc)
                if useProgBar
                    tmpAcc = mean(minAcc)
                    showValues["Train Cost"] = round(tmpAcc ; digits=4)
                end
            end

            a = FCache[outLayer][:A]
            if Cost
                minCost = cost(eval(:($lossFun)), a, Y)
                push!(minCosts, minCost)
                if useProgBar
                    tmpCost = mean(minCosts)
                    showValues["Train Cost"] = round(tmpCost; digits = 4)
                end
            end

            if embedUpdate
                BCache = chainBackProp(X,Y,
                                       model,
                                       FCache;
                                       tMiniBatch = nB + 1,
                                       kwargs...)
            else
                BCache = chainBackProp(X,Y,
                                      model,
                                      FCache;
                                      kwargs...)

                chainUpdateParams!(model; tMiniBatch = nB + 1)
            end #if embedUpdate
        end

        if Accuracy
            tmpAcc = mean(minAcc)
            push!(Accuracies, tmpAcc)
            if useProgBar
                showValues["Train Accuracy"] = round(tmpAcc; digits=4)
            end
        end

        if Cost
            tmpCost = mean(minCosts)
            push!(Costs, tmpCost)
            if useProgBar
                showValues["Train Cost"] = round(tmpCost; digits=4)
            end
        end

        if printCostsInterval>0 && i%printCostsInterval==0
            println("N = $i, Cost = $(Costs[end])")
        end

        if test
            testDict = predict(model, testData, testLabels; useProgBar = false, batchSize=batchSize)
            testcost = cost(eval(:($lossFun)), testDict[:YhatProb], testLabels)
            if Accuracy
                push!(testAcc, testDict[:accuracy])
                if useProgBar
                    showValues["Test Accuracy"] = round(testDict[:accuracy]; digits=4)
                end
            end
            if Cost
                push!(testCost, testcost)
                if useProgBar
                    showValues["Test Cost"] = round(testcost; digits=4)
                end
            end
        end

        if useProgBar
            showValues["Epoch $(epochs)"] = i
            showValues["Instances $(m)"] = m
            update!(p, (i-1)*nMiniBatch+(nB+1); showvalues=[("Epoch $(epochs)", pop!(showValues, "Epoch $(epochs)")), ("Instances $(m)", pop!(showValues, "Instances $(m)")), showValues...])
        end

        # if useProgBar
        #     if useProgBar
        #         if :accuracy in metrics && :cost in metrics
        #             update!(p, ((i-1)*(nB + ((m % batchSize == 0) ? 0 : 1)))+(nB + 1); showvalues=[("Epoch ($epochs)", i), ("Instances ($m)", m), (:Accuracy, round(mean(minAcc); digits=4)), (:Cost, round(mean(minCosts); digits=4))])
        #         elseif :accuracy in metrics
        #             update!(p, ((i-1)*(nB + ((m % batchSize == 0) ? 0 : 1)))+(nB + 1); showvalues=[("Epoch ($epochs)", i), ("Instances ($m)", j*batchSize), (:Accuracy, round(mean(minAcc); digits=4))])
        #         elseif :cost in metrics
        #             update!(p, ((i-1)*(nB + ((m % batchSize == 0) ? 0 : 1)))+(nB + 1); showvalues=[("Epoch ($epochs)", i), ("Instances ($m)", j*batchSize), (:Cost, round(mean(minCosts); digits=4))])
        #         else
        #             update!(p, ((i-1)*(nB + ((m % batchSize == 0) ? 0 : 1)))+(nB + 1); showvalues=[("Epoch ($epochs)", i), ("Instances ($m)", j*batchSize)])
        #         end
        #     end
        # end
    end

    # model.W, model.B = W, B
    outDict = Dict{Symbol, Array{T,1} where {T}}()
    if Accuracy
        outDict[:trainAccuracies] = Accuracies
        if test
            outDict[:testAccuracies] = testAcc
        end
    end

    if Cost
        outDict[:trainCosts] = Costs
        if test
            outDict[:testCosts] = testCost
        end
    end
    return outDict
end #train

export train
