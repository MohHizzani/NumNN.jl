
"""
    convert array of integer classes into one Hot coding
"""
function oneHot(Y; classes = [], numC = 0)
    if numC > 0
        c = numC
        Cs = sort(unique(Y))
    elseif length(classes) > 0
        Cs = sort(classes)
        c = length(Cs)
    else
        Cs = sort(unique(Y))
        c = length(Cs)
    end
    hotY = BitArray{2}(undef, c, 0)
    for y in Y
        hotY = hcat(hotY, Integer.(Cs .== y))
    end
    return hotY
end #oneHot

export oneHot

function resetCount!(outLayer::Layer,
                     cnt::Symbol)
    prevLayer = outLayer.prevLayer
    if outLayer isa Input
        # if outLayer.forwCount != 0
            eval(:($outLayer.$cnt = 0))
        # end #if outLayer.forwCount != 0
    elseif isa(outLayer, AddLayer) #if prevLayer == nothing
        for prevLayer in outLayer.prevLayer
            resetCount!(prevLayer, cnt)
        end #for
        eval(:($outLayer.$cnt = 0))
    else #if prevLayer == nothing
        resetCount!(prevLayer, cnt)
        eval(:($outLayer.$cnt = 0))
    end #if prevLayer == nothing
    return nothing
end #function resetForwCount

export resetCount!


### to extend the getindex fun

Base.getindex(it, key; default) = haskey(it, key) ? it[key] : default


### getTrainParams

function getTrainParams(
    outLayer::AoL,
    Params::Array{Symbol,1}=[:W,:B],
    cLayer::LoN=nothing,
    paramsDict::Dict{Layer,Array{Symbol,1}}=Dict(),
    cnt::Integer = -1
) where {AoL <: Union{Array{Layer,1},Layer}, LoN <: Union{Layer, Nothing}}

    if outLayer isa Layer
        outLayer = [outLayer]
    end

    if cLayer == nothing
        for oLayer in outLayer
            paramsDict = getTrainParams(outLayer, Params, oLayer, paramsDict, cnt)
        end
    end


    if cLayer isa Input
        cLayer.backCount += 1

        for field in Params
            if hasfield(cLayer, field)

end
