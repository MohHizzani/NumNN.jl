

"""
    returns the outLayer from an array of layers and
    the input of the model X
"""
function chain(X, arr::Array{L,1}) where {L<:Layer}
    prevLayer = nothing
    La = eltype(arr)
    for l in arr
        if isequal(prevLayer, nothing)
            global a = La(l.numNodes, l.actFun, X, keepProb=l.keepProb)
            prevLayer = a
        else
            global a = La(l.numNodes, l.actFun, prevLayer, keepProb=l.keepProb)
            prevLayer = a
        end
    end #for

    return a
end

export chain
