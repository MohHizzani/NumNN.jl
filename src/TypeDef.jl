
abstract type Layer end

export Layer


### FCLayer


mutable struct FCLayer <: Layer
    numNodes::Integer
    actFun::Symbol

    """
        drop-out keeping node probability
    """
    keepProb::AbstractFloat
    W::Array{T,N} where {T,N}
    B::Array{T,N} where {T,N}
    dW::Array{T,N} where {T,N}
    dB::Array{T,N} where {T,N}
    ### adding Z & A place holder for recursive calling
    ### and a counter for how many it was called
    Z::Array{T,N} where {T,N}
    dZ::Array{T,N} where {T,N}
    A::Array{T,N} where {T,N}
    forwCount::Integer
    V::Dict{Symbol,Array{T,N} where {T,N}}
    S::Dict{Symbol,Array{T,N} where {T,N}}
    backCount::Integer
    updateCount::Integer
    """
        pointer to previous layer
    """
    prevLayer::L where {L<:Union{Layer,Nothing}}
    nextLayers::Array{Layer,1}
    function FCLayer(numNodes, actFun, layerInput = nothing; keepProb = 1.0)
        # W, B
        if isa(layerInput, Layer)
            T = eltype(layerInput.W)
            nl = numNodes
            nl_1 = size(layerInput.W)[1]
            prevLayer = layerInput
        elseif isa(layerInput, Array)
            T = eltype(layerInput)
            nl = numNodes
            nl_1 = size(layerInput)[1]
            prevLayer = nothing
        else
            T = Any
            nl = 0
            nl_1 = 0
            prevLayer = nothing
        end
        new(
            numNodes,
            actFun,
            keepProb,
            Matrix{T}(undef, nl, nl_1),
            Matrix{T}(undef, nl, 1),
            Matrix{T}(undef, 0, 0),
            Matrix{T}(undef, 0, 0),
            Matrix{T}(undef, 0, 0),
            Matrix{T}(undef, 0, 0),
            Matrix{T}(undef, 0, 0),
            0,
            Dict(:dw => Matrix{T}(undef, 0, 0), :db => Matrix{T}(undef, 0, 0)),
            Dict(:dw => Matrix{T}(undef, 0, 0), :db => Matrix{T}(undef, 0, 0)),
            0,
            0,
            prevLayer,
            Array{Layer,1}(undef,0)
        )#
    end #FCLayer
end #struct FCLayer

export FCLayer


### AddLayer


mutable struct AddLayer <: Layer
    nextLayers::Array{Layer,1}
    prevLayer::Array{Layer,1}
    numNodes::Integer
    channels::Integer
    forwCount::Integer
    backCount::Integer
    updateCount::Integer
    A::Array{T,N} where {T,N}
    dZ::Array{T,N} where {T,N}
    function AddLayer(; numNodes = 0)
        # numNodes = l1.numNodes
        # T = eltype(l1.W)
        new(Array{Layer,1}(undef,0),
            Array{Layer,1}(undef,0),
            numNodes, 0, 0, 0,
            Matrix{Nothing}(undef, 0, 0),
            Matrix{Nothing}(undef, 0, 0))
    end #function AddLayer
end

export AddLayer



### Activation

mutable struct Activation <: Layer
    actFun::Symbol
    numNodes::Integer
    channels::Integer

    nextLayers::Array{Layer,1}
    prevLayer::L where {L<:Union{Layer,Nothing}}

    forwCount::Integer
    backCount::Integer
    updateCount::Integer

    A::Array{T,N} where {T,N}
    dZ::Array{T,N} where {T,N}

    function Activation(actFun=:relu)
        new(
            actFun,
            0, #numNodes
            0, #channels
            Array{Layer,1}(undef,0),
            nothing,
            0,
            0,
            0,
            Matrix{Nothing}(undef, 0, 0),
            Matrix{Nothing}(undef, 0, 0),
        )
    end #function Activation
end #mutable struct Activation


export Activation



### Input

mutable struct Input <: Layer
    A::Array{T,N} where {T,N}
    dZ::Array{T,N} where {T,N}
    nextLayers::Array{Layer,1}
    prevLayer::L where {L<:Union{Layer,Nothing}}
    channels::Integer
    numNodes::Integer
    forwCount::Integer
    backCount::Integer
    updateCount::Integer
    function Input(X::Array{T,N}) where {T,N}
        if N==2
            channels = size(X)[1]
        elseif N==3
            channels = size(X)[2]
        elseif N==4
            channels = size(X)[3]
        elseif N==5
            channels = size(X)[4]
        end
        dZ = similar(X)
        dZ .= 0
        new(X,
            dZ,
            Array{Layer,1}(undef,0),
            nothing,
            channels,
            channels,
            0,
            0,
            0,
            )
    end #function Layer
end #mutable struct Input


export Input


### BatchNorm

mutable struct BatchNorm <: Layer
    numNodes::Integer
    channels::Integer

    nextLayers::Array{Layer,1}
    prevLayer::L where {L<:Union{Layer,Nothing}}

    forwCount::Integer
    backCount::Integer
    updateCount::Integer

    W::N where {N <: Number}
    dW::N where {N <: Number}

    B::N where {N <: Number}
    dB::N where {N <: Number}

    V::Dict{Symbol, N where {N <: Number}}
    S::Dict{Symbol, N where {N <: Number}}

    Z::Array{T, M} where {T,M}
    dZ::Array{T, M} where {T,M}
    A::Array{T, M} where {T,M}

    dim::Integer

    ϵ::AbstractFloat


    function BatchNorm(;dim=1, ϵ=1e-10)
        new(
            0, #numNodes
            0, #channels
            Array{Layer,1}(undef,0), #nextLayers
            nothing, #prevLayer
            0, #forwCount
            0, #backCount
            0, #updateCount
            randn(), #W
            0, #dW
            0, #B
            0, #dB
            Dict(:dw=>0,
                 :db=>0), #V
            Dict(:dw=>0,
                 :db=>0), #S
            Array{Any,1}(undef,0), #Z
            Array{Any,1}(undef,0), #dZ
            Array{Any,1}(undef,0), #A
            dim,
            ϵ,
        )
    end #function BatchNorm


end #mutable struct BatchNorm

export BatchNorm

### Model

mutable struct Model
    # layers::Array{Layer,1}
    outLayer::Layer
    lossFun::Symbol

    """
        regulization type
            0 : means no regulization
            1 : L1 regulization
            2 : L2 regulization
    """
    regulization::Integer

    """
        regulization constant
    """
    λ::AbstractFloat

    """
        learning rate
    """
    α::AbstractFloat

    optimizer::Symbol
    ϵAdam::AbstractFloat
    β1::AbstractFloat
    β2::AbstractFloat

    function Model(
        X,
        Y,
        outLayer::Layer,
        α;
        optimizer = :gds,
        β1 = 0.9,
        β2 = 0.999,
        ϵAdam = 1e-8,
        regulization = 0,
        λ = 1.0,
        lossFun = :categoricalCrossentropy,
    )

        deepInitWB!(X, outLayer)
        resetCount!(outLayer, :forwCount)
        deepInitVS!(outLayer, optimizer)
        resetCount!(outLayer, :forwCount)
        @assert regulization in [0, 1, 2]
        return new(
            outLayer,
            lossFun,
            regulization,
            λ,
            α,
            optimizer,
            ϵAdam,
            β1,
            β2,
        )
    end #inner-constructor

end #Model

export Model
