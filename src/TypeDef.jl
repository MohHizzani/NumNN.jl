
abstract type Layer end

struct FCLayer <: Layer
    numNodes::Integer
    actFun::Symbol

    """
        drop-out keeping node probability
    """
    keepProb::AbstractFloat
    # W::AbstractArray{<:AbstractFloat,2}
    # B::AbstractArray{<:AbstractFloat,2}
    function Layer(numNodes,actFun;keepProb=1.0)
        # W, B
        new(numNodes, actFun, keepProb)
    end #Layer
end #struct Layer

mutable struct Model
    layers::AbstractArray{Layer,1}
    lossFun::Symbol

    """
        regulization type
            0 : mean no regulization
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
    W::AbstractArray{AbstractArray{AbstractFloat,2},1}
    B::AbstractArray{AbstractArray{AbstractFloat,2},1}
    function Model(X, Y, layers, α; regulization = 0, λ = 1.0, lossFun = :crossentropy)
        W, B = deepInitWB(X, Y, layers)
        @assert regulization in [0, 1, 2]
        return new(layers, lossFun, regulization, λ, α, W, B)
    end #inner-constructor
end #Model
