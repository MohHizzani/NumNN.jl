
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
    W::AbstractArray{AbstractArray{Number,2},1}
    B::AbstractArray{AbstractArray{Number,2},1}


    """
        V is the velocity
        S is the RMSProp
    """
    V::AbstractArray{AbstractArray{Number,2},1}
    S::AbstractArray{AbstractArray{Number,2},1}
    optimizer::Symbol
    function Model(X, Y, layers, α; optimizer = :nonadam, regulization = 0, λ = 1.0, lossFun = :categoricalCrossentropy)
        W, B = deepInitWB(X, Y, layers)
        if optimizer == "adam"
            V, S = deepInitVS(W,B)
        else
            V = Array{Array{Number,2},1}(undef,0)
            S = Array{Array{Number,2},1}(undef,0)
        end
        @assert regulization in [0, 1, 2]
        return new(layers, lossFun, regulization, λ, α, W, B, V, S, optimizer)
    end #inner-constructor
end #Model

export Model
