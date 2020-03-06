
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
    W::Array{T,2} where {T}
    B::Array{T,2} where {T}
    dW::Array{T,2} where {T}
    dB::Array{T,2} where {T}
    ### adding Z & A place holder for recursive calling
    ### and a counter for how many it was called
    Z::Array{T,2} where {T}
    dA::Array{T,2} where {T}
    A::Array{T,2} where {T}

    V::Dict{Symbol,Array{T,2} where {T,N}}
    S::Dict{Symbol,Array{T,2} where {T,N}}

    forwCount::Integer
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
            Matrix{T}(undef, 0, 0),
            Matrix{T}(undef, 0, 0),
            Matrix{T}(undef, 0, 0),
            Matrix{T}(undef, 0, 0),
            Matrix{T}(undef, 0, 0),
            Matrix{T}(undef, 0, 0),
            Matrix{T}(undef, 0, 0),
            Dict(:dw => Matrix{T}(undef, 0, 0), :db => Matrix{T}(undef, 0, 0)),
            Dict(:dw => Matrix{T}(undef, 0, 0), :db => Matrix{T}(undef, 0, 0)),
            0,
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
    numNodes::Integer
    channels::Integer

    A::Array{T,N} where {T,N}
    dA::Array{T,N} where {T,N}

    forwCount::Integer
    backCount::Integer
    updateCount::Integer

    nextLayers::Array{Layer,1}
    prevLayer::Array{Layer,1}
    function AddLayer(; numNodes = 0)
        # numNodes = l1.numNodes
        # T = eltype(l1.W)
        new(
            numNodes,
            numNodes,
            Matrix{Nothing}(undef, 0, 0),
            Matrix{Nothing}(undef, 0, 0),
            0,
            0,
            0,
            Array{Layer,1}(undef,0), #nextLayers
            Array{Layer,1}(undef,0), #prevLayer
            )
    end #function AddLayer
end

export AddLayer



### Activation

mutable struct Activation <: Layer
    actFun::Symbol
    numNodes::Integer
    channels::Integer


    forwCount::Integer
    backCount::Integer
    updateCount::Integer

    A::Array{T,N} where {T,N}
    dA::Array{T,N} where {T,N}

    nextLayers::Array{Layer,1}
    prevLayer::L where {L<:Union{Layer,Nothing}}

    function Activation(actFun=:relu)
        new(
            actFun,
            0, #numNodes
            0, #channels
            0,
            0,
            0,
            Matrix{Nothing}(undef, 0, 0),
            Matrix{Nothing}(undef, 0, 0),
            Array{Layer,1}(undef,0),
            nothing,
            )
    end #function Activation
end #mutable struct Activation


export Activation



### Input

mutable struct Input <: Layer
    channels::Integer
    numNodes::Integer

    A::Array{T,N} where {T,N}
    dA::Array{T,N} where {T,N}

    forwCount::Integer
    backCount::Integer
    updateCount::Integer

    nextLayers::Array{Layer,1}
    prevLayer::L where {L<:Union{Layer,Nothing}}

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
        dA = similar(X)
        dA .= 0
        new(
            channels,
            channels,
            X,
            dA,
            0,
            0,
            0,
            Array{Layer,1}(undef,0),
            nothing,
            )
    end #function Layer
end #mutable struct Input


export Input


### BatchNorm

mutable struct BatchNorm <: Layer
    numNodes::Integer
    channels::Integer

    dim::Integer

    ϵ::AbstractFloat

    μ::Array{T,N} where{T,N}
    var::Array{T,N} where {T,N}

    W::Array{T, N} where {T,N}
    dW::Array{T, N} where {T,N}

    B::Array{T, N} where {T,N}
    dB::Array{T, N} where {T,N}

    V::Dict{Symbol,Array}
    S::Dict{Symbol,Array}

    Z::Array{T, M} where {T,M}
    Ai_μ::Array{T, M} where {T,M}
    Ai_μ_s::Array{T, M} where {T,M}
    dA::Array{T, M} where {T,M}
    A::Array{T, M} where {T,M}

    forwCount::Integer
    backCount::Integer
    updateCount::Integer

    nextLayers::Array{Layer,1}
    prevLayer::L where {L<:Union{Layer,Nothing}}


    function BatchNorm(;dim=1, ϵ=1e-10)
        new(
            0, #numNodes
            0, #channels
            dim,
            ϵ,
            Array{Any,1}(undef,0), #μ
            Array{Any,1}(undef,0), #var
            Array{Any,1}(undef,0), #W
            Array{Any,1}(undef,0), #dW
            Array{Any,1}(undef,0), #B
            Array{Any,1}(undef,0), #dB
            Dict(:dw=>Array{Any,1}(undef,0),
                 :db=>Array{Any,1}(undef,0)), #V
            Dict(:dw=>Array{Any,1}(undef,0),
                 :db=>Array{Any,1}(undef,0)), #S
            Array{Any,1}(undef,0), #Z
            Array{Any,1}(undef,0), #Ai_μ
            Array{Any,1}(undef,0), #Ai_μ_s
            Array{Any,1}(undef,0), #dA
            Array{Any,1}(undef,0), #A
            0, #forwCount
            0, #backCount
            0, #updateCount
            Array{Layer,1}(undef,0), #nextLayers
            nothing, #prevLayer
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
