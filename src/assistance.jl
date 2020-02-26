using PaddedViews


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
    if prevLayer isa Input
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


###Padding functions


"""
    pad zeros to the Array Ai with amount of p values

    inputs:
        Ai := Array of type T and dimension N
        p  := integer determinde the amount of zeros padding
              i.e.
              if Ai is a 3-dimensional array the padding will be for the first
                  dimension
              if Ai is a 4-dimensional array the padding will be for the first 2
                  dimensions
              if Ai is a 5-dimensional array the padding will be for the first 3
                  dimensions

    output:
        PaddinView array where it contains the padded values and the original
            data without copying it
"""
function padding(Ai::Array{T,4},
                 p_H::Integer,
                 p_W::Integer=-1) where {T}

    if p_W < 0
        p_W = p_H
    end
    n_H, n_W, c, m = size(Ai)
    return PaddedView(0, Ai, (2*p_H+n_H, 2*p_W+n_W, c, m), (1+p_H, 1+p_W, 1, 1))
end #function padding(Ai::Array{T,4}


function padding(Ai::Array{T,3},
                 p::Integer) where {T}

    n_H, c, m = size(Ai)
    return PaddedView(0, Ai, (2*p+n_H, c, m), (1+p, 1, 1))
end #function padding(Ai::Array{T,3}


function padding(Ai::Array{T,5},
                 p_H::Integer,
                 p_W::Integer=-1,
                 p_D::Integer=-1) where {T}

    if p_W < 0
        p_W = p_H
    end
    if p_D < 0
        p_D = p_H
    end
    n_H, n_W, n_D, c, m = size(Ai)
    return PaddedView(0, Ai,
                        (2*p_H+n_H, 2*p_W+n_W, 2*p_D+n_D, c, m),
                        (1+p_H, 1+p_W, 1+p_D, 1, 1))
end #function padding(Ai::Array{T,5}

export padding
