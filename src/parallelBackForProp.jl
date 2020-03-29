include("parallelLayerForProp.jl")


using ProgressMeter
using Random
using LinearAlgebra

###

"""
    perform the chained forward propagation using recursive calls

    input:
    X := input of the input layer
    cLayer := Input Layer
    cnt := an internal counter used to cache the layers was performed
           not to redo it again

    returns:
    Cache := the output each layer either A, Z or together As Dict of layer to dict of Symbols and Arrays

    for internal use, it set again the values of Z and A in each layer
        to be used later in back propagation and add one to the layer
        forwCount value when pass through it
"""
function chainForProp(
    X::AbstractArray{T,N},
    cLayer::Layer,
    cnt::Integer = -1;
    FCache = Dict{Layer,Dict{Symbol,AbstractArray}}(),
    kwargs...,
) where {T,N}
    if cnt < 0
        cnt = cLayer.forwCount + 1
    end

    if length(cLayer.nextLayers) == 0
        if cLayer.forwCount < cnt
            FCache[cLayer] = layerForProp(cLayer; FCache = FCache, kwargs...)
        end #if cLayer.forwCount < cnt
        return FCache
    elseif isa(cLayer, AddLayer) #if typeof(cLayer)==AddLayer
        if all(
            i -> (i.forwCount == cLayer.prevLayer[1].forwCount),
            cLayer.prevLayer,
        )
            FCache[cLayer] = layerForProp(cLayer; FCache = FCache, kwargs...)
            for nextLayer in cLayer.nextLayers
                FCache =
                    chainForProp(X, nextLayer, cnt; FCache = FCache, kwargs...)
            end
        end #if all

        return FCache
    else #if cLayer.prevLayer==nothing
        if cLayer.forwCount < cnt
            if cLayer isa Input
                FCache[cLayer] =
                    layerForProp(cLayer, X; FCache = FCache, kwargs...)
            else
                FCache[cLayer] =
                    layerForProp(cLayer; FCache = FCache, kwargs...)
            end
            for nextLayer in cLayer.nextLayers
                FCache =
                    chainForProp(X, nextLayer, cnt; FCache = FCache, kwargs...)
            end

        end #if cLayer.forwCount < cnt
        return FCache
    end #if cLayer.prevLayer!=nothing

    return FCache

end #function chainForProp!

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
function predictBatch(model::Model, X::AbstractArray, Y = nothing; kwargs...)

    kwargs = Dict(kwargs...)
    kwargs[:prediction] = getindex(kwargs, :prediction, default = true)

    FCache = chainForProp(X, model.inLayer; kwargs...)
    Ŷ = FCache[model.outLayer][:A]
    T = eltype(Ŷ)
    outLayer = model.outLayer
    actFun = outLayer.actFun
    # if isbool(Y)
    return Ŷ, probToValue(eval(:($actFun)), Ŷ; labels = Y)...

end #predict

export predictBatch

function predict(model::Model, X_In::AbstractArray, Y_In = nothing; kwargs...)

    kwargs = Dict(kwargs...)
    batchSize = getindex(kwargs, :batchSize; default = 32)
    printAcc = getindex(kwargs, :printAcc; default = true)
    useProgBar = getindex(kwargs, :useProgBar; default = false)
    GCInt = getindex(kwargs, :GCInt, default = 5)
    noBool = getindex(kwargs, :noBool, default = false)
    kwargs[:prediction] = getindex(kwargs, :prediction, default = true)

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
        p = Progress((m % batchSize != 0 ? nB + 1 : nB), 0.1)
    end

    Ŷ_out =
        Array{AbstractArray{T,N},1}(undef, nB + ((m % batchSize == 0) ? 0 : 1))
    Ŷ_prob_out =
        Array{AbstractArray{T,N},1}(undef, nB + ((m % batchSize == 0) ? 0 : 1))
    accuracy =
        Array{AbstractFloat,1}(undef, nB + ((m % batchSize == 0) ? 0 : 1))
    # Threads.@threads
    @simd for j = 1:nB
        downInd = (j - 1) * batchSize + 1
        upInd = j * batchSize
        X = view(X_In, axX..., downInd:upInd)
        if Y_In != nothing
            Y = view(Y_In, axY..., downInd:upInd)
        end
        Ŷ_prob_out[j], Ŷ_out[j], accuracy[j] =
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
        Ŷ_prob_out[nB+1], Ŷ_out[nB+1], accuracy[nB+1] =
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
    # end
    if noBool
        Ŷ_values = nothing
        Ŷ_prob = nothing
    else
        Ŷ_values = Array{T,N}(undef, repeat([0], N)...)
        Ŷ_values = cat(Ŷ_values, Ŷ_out[1]; dims = 1:N)
        Ŷ_prob = Array{T,N}(undef, repeat([0], N)...)
        Ŷ_prob = cat(Ŷ_prob, Ŷ_prob_out[1]; dims = 1:N)
        for i = 2:length(Ŷ_out)
            Ŷ_values = cat(Ŷ_values, Ŷ_out[i], dims = N)
            Ŷ_prob = cat(Ŷ_prob, Ŷ_prob_out[i], dims = N)
        end
    end

    return Dict(
        :YhatValue => Ŷ_values,
        :YhatProb => Ŷ_prob,
        :accuracy => accuracyM,
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



"""

    inputs:
    X := is a (nx, m) matrix
    Y := is a (c,  m) matrix
    model := is the model to perform the back propagation on
    FCache := the cached values of the forward propagation
    cLayer := is an internal variable to hold the current layer
    cnt := is an internal variable to count the step of back propagation currently on

    output:
    nothing


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

    elseif cLayer isa AddLayer
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
