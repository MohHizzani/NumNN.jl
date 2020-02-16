

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



"""
    connect with the previous layer
"""
function (l::FCLayer)(li_1::Layer)
    l.prevLayer = li_1
    if ! in(l,li_1.nextLayers)
        push!(li_1.nextLayers, l)
    end
    return l
end #function (l::FCLayer)(li_1::Layer)

function (l::AddLayer)(li::Array{Layer,1})
    for lj in li
        if !in(lj,l.prevLayer)
            push!(l.prevLayer, lj)
        end
        if !in(l, lj.nextLayer)
            push!(li.nextLayers, l)
        end
    end #for
end #function (l::AddLayer)(li::Array{Layer,1})

"""
    define input as X
"""
function (l::FCLayer)(x::Array)
    l.prevLayer = nothing
    return l
end #function (l::FCLayer)(x::Array)

export l
