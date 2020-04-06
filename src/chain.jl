

"""
    function chain(X, arr::Array{L,1}) where {L<:Layer}

Returns the input `Layer` and the output `Layer` from an `Array` of layers and the input of the model as and `Array` `X`

"""
function chain(X, arr::Array{L,1}) where {L<:Layer}
    if ! (arr[1] isa Input)
        X_Input = Input(X)
    end

    for l=1:length(arr)
        if l==1
            if ! isa(arr[l],Input)
                X = arr[l](X_Input)
            else
                X_Input = X = arr[l](X)
            end
        else
            X = arr[l](X)
        end
    end #for

    return X_Input, X
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
    l.inputS = li_1.outputS
    l.outputS = (l.channels, li_1.outputS[2])
    return l
end #function (l::FCLayer)(li_1::Layer)


function (l::Activation)(li_1::Layer)
    l.prevLayer = li_1

    l.channels = li_1.channels

    l.inputS = l.outputS = li_1.outputS

    if ! in(l,li_1.nextLayers)
        push!(li_1.nextLayers, l)
    end
    return l
end

function (l::BatchNorm)(li_1::Layer)
    l.prevLayer = li_1

    l.channels = li_1.channels

    N = length(li_1.outputS)

    if l.dim >= N
        throw(DimensionMismatch("Normalization Dimension must be less than $N"))
    end

    l.inputS = l.outputS = li_1.outputS



    if ! in(l,li_1.nextLayers)
        push!(li_1.nextLayers, l)
    end
    return l

end #function (l::BatchNorm)

function (l::AddLayer)(ls::Array{L,1}) where {L<:Layer}
    for li in ls
        if !in(li,l.prevLayer)
            push!(l.prevLayer, li)
        end
        if !in(l, li.nextLayers)
            push!(li.nextLayers, l)
        end
    end #for

    l.inputS = l.outputS = ls[1].outputS
    l.channels = ls[1].channels

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


function (l::Input)(X::AbstractArray{T,N}) where {T,N}
    l.A = X
    channels = size(X)[end-1]
    l.channels = channels
    return l
end #function (l::Input)(X::AbstractArray{T,N})

export l
