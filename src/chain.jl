

"""
    returns the outLayer from an array of layers and
    the input of the model X
"""
function chain(X, arr::Array{L,1}) where {L<:Layer}
    if ! (arr[1] isa Input)
        X = Input(X)
    end

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

function (l::MaxPool1D)(li_1::Layer)
    l.prevLayer = li_1
    l.channels = li_1.channels
    if ! in(l,li_1.nextLayers)
        push!(li_1.nextLayers, l)
    end
    return l
end

function (l::MaxPool2D)(li_1::Layer)
    l.prevLayer = li_1
    l.channels = li_1.channels
    if ! in(l,li_1.nextLayers)
        push!(li_1.nextLayers, l)
    end
    return l
end

function (l::MaxPool3D)(li_1::Layer)
    l.prevLayer = li_1
    l.channels = li_1.channels
    if ! in(l,li_1.nextLayers)
        push!(li_1.nextLayers, l)
    end
    return l
end

function (l::AveragePool1D)(li_1::Layer)
    l.prevLayer = li_1
    l.channels = li_1.channels
    if ! in(l,li_1.nextLayers)
        push!(li_1.nextLayers, l)
    end
    return l
end

function (l::AveragePool2D)(li_1::Layer)
    l.prevLayer = li_1
    l.channels = li_1.channels
    if ! in(l,li_1.nextLayers)
        push!(li_1.nextLayers, l)
    end
    return l
end

function (l::AveragePool3D)(li_1::Layer)
    l.prevLayer = li_1
    l.channels = li_1.channels
    if ! in(l,li_1.nextLayers)
        push!(li_1.nextLayers, l)
    end
    return l
end

function (l::Activation)(li_1::Layer)
    l.prevLayer = li_1
    try
        l.numNodes = li_1.numNodes
        if !(li_1.numNodes>0)
            l.channels = li_1.channels
        end
    catch e
        l.channels = li_1.channels
    end

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

function (l::BatchNorm)(li_1::Layer)
    l.prevLayer = li_1
    try
        l.numNodes = li_1.numNodes
        if !(li_1.numNodes>0)
            l.channels = li_1.channels
        end
    catch e
        l.channels = li_1.channels
    end

    if ! in(l,li_1.nextLayers)
        push!(li_1.nextLayers, l)
    end
    return l

end #function (l::BatchNorm)


function (l::Input)(X::AbstractArray{T,N}) where {T,N}
    l.A = X
    if N==2
        channels = size(X)[1]
    elseif N==3
        channels = size(X)[2]
    elseif N==4
        channels = size(X)[3]
    elseif N==5
        channels = size(X)[4]
    end
    l.channels = l.numNodes = channels
    return l
end #function (l::Input)(X::AbstractArray{T,N})

export l
