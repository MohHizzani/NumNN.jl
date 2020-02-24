

"""
    returns the outLayer from an array of layers and
    the input of the model X
"""
function chain(X, arr::Array{L,1}) where {L<:Layer}
    prevLayer = nothing
    La = eltype(arr)
    a = nothing
    for l=1:length(arr)
        if l==1
            X = arr[l](X)
        else
            X = arr[l](X)
        end
    end #for

    return X
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


function (l::Conv1D)(li_1::Layer)
    l.prevLayer = li_1
    if ! in(l,li_1.nextLayers)
        push!(li_1.nextLayers, l)
    end
    return l
end

function (l::Conv2D)(li_1::Layer)
    l.prevLayer = li_1
    if ! in(l,li_1.nextLayers)
        push!(li_1.nextLayers, l)
    end
    return l
end

function (l::Conv3D)(li_1::Layer)
    l.prevLayer = li_1
    if ! in(l,li_1.nextLayers)
        push!(li_1.nextLayers, l)
    end
    return l
end

function (l::AddLayer)(ls::Array{L,1}) where {L<:Layer}
    for li in ls
        if !in(li,l.prevLayer)
            push!(l.prevLayer, li)
        end
        if !in(l, li.nextLayers)
            push!(li.nextLayers, l)
        end
    end #for
    l.numNodes = ls[1].numNodes
    return l
end #function (l::AddLayer)(li::Array{Layer,1})

"""
    define input as X
"""
function (l::FCLayer)(x::Array)
    l.prevLayer = nothing
    return l
end #function (l::FCLayer)(x::Array)

function (l::Conv1D)(x::Array)
    l.prevLayer = nothing
    return l
end

function (l::Conv2D)(x::Array)
    l.prevLayer = nothing
    return l
end

function (l::Conv3D)(x::Array)
    l.prevLayer = nothing
    return l
end


export l
