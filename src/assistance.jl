
"""
    convert array of integer classes into one Hot coding
"""
function oneHot(Y; classes = [], numC = 0)
    if numC > 0
        c = numC
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

function resetForwCount!(outLayer::Layer)
    prevLayer = outLayer.prevLayer
    if prevLayer == nothing
        if outLayer.forwCount != 0
            outLayer.forwCount = 0
        end #if outLayer.forwCount != 0
    elseif outLayer==AddLayer #if prevLayer == nothing
        resetForwCount!(prevLayer)
        resetForwCount!(outLayer.l2)
        outLayer.forwCount = 0
    else #if prevLayer == nothing
        resetForwCount!(prevLayer)
        outLayer.forwCount = 0
    end #if prevLayer == nothing
    return nothing
end #function resetForwCount

export resetForwCount!
