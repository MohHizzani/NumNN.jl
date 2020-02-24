

abstract type ConvLayer <: Layer end

export ConvLayer


mutable struct Conv2D <: ConvLayer
    channels::Integer

    """
        filter size
    """
    f::Tuple{Integer, Integer}

    """
        stides size
    """
    s::Tuple{Integer, Integer}

    padding::Symbol

    W::Array{Array{F, 3},1}  where {F}
    dW::Array{Array{F,3},1}  where {F}



    B::Array{Array{F,3},1} where {F}
    dB::Array{Array{F,3},1} where {F}

    actFun::Symbol
    keepProb::AbstractFloat

    Z::Array{T,4} where {T}
    dZ::Array{T,4} where {T}
    A::Array{T,4} where {T}

    V::Dict{Symbol, Array{Array{T,3},1} where {T}}
    S::Dict{Symbol, Array{Array{T,3},1} where {T}}

    forwCount::Integer
    backCount::Integer
    updateCount::Integer

    prevLayer::L where {L<:Union{Layer,Nothing}}
    nextLayers::Array{Layer,1}

    function Conv2D(c,
                    f::Tuple{Integer, Integer};
                    prevLayer=nothing,
                    strides::Tuple{Integer, Integer}=(1, 1),
                    padding::Symbol=:same,
                    activation::Symbol=:noAct,
                    keepProb=1.0)

        if prevLayer == nothing
            T = Any
        elseif prevLayer isa Array
            T = eltype(prevLayer)
        else
            T = eltype(prevLayer.W)
        end

        new(c,
            f,
            strides,
            padding,
            [Array{T,3}(undef,0,0,0) for i=1:c], #W
            [Array{T,3}(undef,0,0,0) for i=1:c], #dW
            [Array{T,3}(undef,0,0,0) for i=1:c], #B
            [Array{T,3}(undef,0,0,0) for i=1:c], #dB
            activation,
            keepProb,
            Array{T,4}(undef, 0,0,0,0), #Z
            Array{T,4}(undef, 0,0,0,0), #dZ
            Array{T,4}(undef, 0,0,0,0), #A
            Dict(:dw=>[Array{T,3}(undef,0,0,0) for i=1:c],
                 :db=>[Array{T,3}(undef,0,0,0) for i=1:c]), #V
            Dict(:dw=>[Array{T,3}(undef,0,0,0) for i=1:c],
                 :db=>[Array{T,3}(undef,0,0,0) for i=1:c]), #S
            0, #forwCount
            0, #backCount
            0, #updateCount
            prevLayer,
            Array{Layer,1}(undef, 0)  #nextLayer
            )


    end #function Conv2D

end #mutable struct Conv2D

export Conv2D


mutable struct Conv1D <: ConvLayer
    channels::Integer

    """
        filter size
    """
    f::Tuple{Integer}

    """
        stides size
    """
    s::Tuple{Integer}

    padding::Symbol

    W::Array{Array{F, 2},1}  where {F}
    dW::Array{Array{F,2},1}  where {F}



    B::Array{Array{F,2},1} where {F}
    dB::Array{Array{F,2},1} where {F}

    actFun::Symbol
    keepProb::AbstractFloat

    Z::Array{T,3} where {T}
    dZ::Array{T,3} where {T}
    A::Array{T,3} where {T}

    V::Dict{Symbol, Array{Array{T,2},1} where {T}}
    S::Dict{Symbol, Array{Array{T,2},1} where {T}}

    forwCount::Integer
    backCount::Integer
    updateCount::Integer

    prevLayer::L where {L<:Union{Layer,Nothing}}
    nextLayers::Array{Layer,1}

    function Conv1D(c,
                    f::Tuple{Integer};
                    prevLayer=nothing,
                    strides::Tuple{Integer}=(1),
                    padding::Symbol=:same,
                    activation::Symbol=:noAct,
                    keepProb=1.0)

        if prevLayer == nothing
            T = Any
        elseif prevLayer isa Array
            T = eltype(prevLayer)
        else
            T = eltype(prevLayer.W)
        end

        new(c,
            f,
            strides,
            padding,
            [Array{T,2}(undef,0,0) for i=1:c], #W
            [Array{T,2}(undef,0,0) for i=1:c], #dW
            [Array{T,2}(undef,0,0) for i=1:c], #B
            [Array{T,2}(undef,0,0) for i=1:c], #dB
            activation,
            keepProb,
            Array{T,3}(undef, 0,0,0), #Z
            Array{T,3}(undef, 0,0,0), #dZ
            Array{T,3}(undef, 0,0,0), #A
            Dict(:dw=>[Array{T,2}(undef,0,0) for i=1:c],
                 :db=>[Array{T,2}(undef,0,0) for i=1:c]), #V
            Dict(:dw=>[Array{T,2}(undef,0,0) for i=1:c],
                 :db=>[Array{T,2}(undef,0,0) for i=1:c]), #S
            0, #forwCount
            0, #backCount
            0, #updateCount
            prevLayer,
            Array{Layer,1}(undef, 0)  #nextLayer
            )


    end #function Conv1D

end #mutable struct Conv1D

export Conv1D
