
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
