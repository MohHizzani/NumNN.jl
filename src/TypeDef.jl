
abstract type Layer end

export Layer

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


mutable struct AddLayer <: Layer
    nextLayers::Array{Layer,1}
    prevLayer::Array{Layer,1}
    numNodes::Integer
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
