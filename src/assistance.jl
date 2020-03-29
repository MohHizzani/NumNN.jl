
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
