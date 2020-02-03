
abstract type Layer end

export Layer

mutable struct FCLayer <: Layer
    numNodes::Integer
    actFun::Symbol

    """
        drop-out keeping node probability
    """
    keepProb::AbstractFloat
    W::Array{T,2} where {T}
    B::Array{T,2} where {T}
    # Vdw::Array{T,2} where {T}
    # Vdb::Array{T,2} where {T}
    # Sdw::Array{T,2} where {T}
    # Sdb::Array{T,2} where {T}

    """
        pointer to previous layer
    """
    prevLayer::L where {L<:Union{Layer, Nothing}}
    function FCLayer(numNodes,actFun, layerInput = nothing; keepProb=1.0)
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
        new(numNodes, actFun, keepProb,
            Matrix{T}(undef, nl,nl_1),
            Matrix{T}(undef, nl,1),
            # Matrix{T}(undef, numNodes,nl_1),
            # Matrix{T}(undef, numNodes,1),
            # Matrix{T}(undef, numNodes,nl_1),
            # Matrix{T}(undef, numNodes,1),
            prevLayer)# != nothing ? Ptr{Layer}(pointer_from_objref(prevLayer)) : nothing)
    end #FCLayer
end #struct FCLayer

export FCLayer


struct AddLayer <: Layer
    prevLayer::Layer
    l2::Layer
    numNodes::Integer
    function AddLayer(l1, l2; numNodes = 0)
        numNodes = l1.numNodes
        new(l1, l1, numNodes)
    end #function AddLayer
end

export AddLayer

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
    W::Array{Array{T,N},1} where {T,N}
    B::Array{Array{T,N},1} where {T,N}


    """
        V is the velocity
        S is the RMSProp
    """
    V::Dict{Symbol,Array{Array{T,N},1}}  where {T,N}
    S::Dict{Symbol,Array{Array{T,N},1}}  where {T,N}
    optimizer::Symbol
    ϵAdam::AbstractFloat
    β1::AbstractFloat
    β2::AbstractFloat
    function Model(X, Y,
                   outLayer::Layer,
                   α;
                   optimizer = :gds,
                   β1 = 0.9,
                   β2 = 0.999,
                   ϵAdam = 1e-8,
                   regulization = 0,
                   λ = 1.0,
                   lossFun = :categoricalCrossentropy)

        W, B = deepInitWB(X, Y, outLayer)
        V, S = deepInitVS(W,B, optimizer)
        @assert regulization in [0, 1, 2]
        return new(outLayer,
                   lossFun,
                   regulization, λ,
                   α,
                   W, B,
                   V, S,
                   optimizer, ϵAdam, β1, β2)
    end #inner-constructor
end #Model

export Model
