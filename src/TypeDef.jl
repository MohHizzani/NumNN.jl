
abstract type Layer end

export Layer

struct FCLayer <: Layer
    numNodes::Integer
    actFun::Symbol

    """
        drop-out keeping node probability
    """
    keepProb::AbstractFloat
    # W::AbstractArray{<:AbstractFloat,2}
    # B::AbstractArray{<:AbstractFloat,2}
    function FCLayer(numNodes,actFun;keepProb=1.0)
        # W, B
        new(numNodes, actFun, keepProb)
    end #Layer
end #struct Layer

export FCLayer

mutable struct Model
    layers::AbstractArray{Layer,1}
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
    W::Array{Array{<:Number,2},1}
    B::Array{Array{<:Number,2},1}


    """
        V is the velocity
        S is the RMSProp
    """
    V::Dict{Symbol,Array{Array{<:Number,2},1}}
    S::Dict{Symbol,Array{Array{<:Number,2},1}}
    optimizer::Symbol
    ϵAdam::AbstractFloat
    β1::AbstractFloat
    β2::AbstractFloat
    function Model(X, Y, layers, α;
                   optimizer = :gds,
                   β1 = 0.9,
                   β2 = 0.999,
                   ϵAdam = 1e-8,
                   regulization = 0,
                   λ = 1.0,
                   lossFun = :categoricalCrossentropy)

        W, B = deepInitWB(X, Y, layers)
        V, S = deepInitVS(W,B, optimizer)
        @assert regulization in [0, 1, 2]
        return new(layers, lossFun, regulization, λ, α,
                   W, B,
                   V, S, optimizer, ϵAdam, β1, β2)
    end #inner-constructor
end #Model

export Model
