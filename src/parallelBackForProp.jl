include("parallelLayerForProp.jl")


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
function chainForProp(X::AbstractArray{T,N}, cLayer::Layer, cnt::Integer=-1; FCache=Dict{Layer,Dict{Symbol, AbstractArray}}(), kwargs...) where {T,N}
    if cnt<0
        cnt = cLayer.forwCount+1
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
                FCache = chainForProp(X, nextLayer, cnt; FCache = FCache, kwargs...)
            end
        end #if all

        return FCache
    else #if cLayer.prevLayer==nothing
        if cLayer.forwCount < cnt
            if cLayer isa Input
                FCache[cLayer] = layerForProp(cLayer, X; FCache = FCache, kwargs...)
            else
                FCache[cLayer] = layerForProp(cLayer; FCache = FCache, kwargs...)
            end
            for nextLayer in cLayer.nextLayers
                FCache = chainForProp(X, nextLayer, cnt; FCache = FCache, kwargs...)
            end

        end #if cLayer.forwCount < cnt
        return FCache
    end #if cLayer.prevLayer!=nothing

    return FCache

end #function chainForProp!

export chainForProp
