
abstract type PaddableLayer <: Layer end

abstract type ConvLayer <: PaddableLayer end

export ConvLayer

### Convolution layers

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



    B::Array{F,3} where {F}
    dB::Array{F,3} where {F}

    actFun::Symbol
    keepProb::AbstractFloat

    Z::Array{T,4} where {T}
    dZ::Array{T,4} where {T}
    A::Array{T,4} where {T}

    V::Dict{Symbol, Array}
    S::Dict{Symbol, Array}

    forwCount::Integer
    backCount::Integer
    updateCount::Integer

    prevLayer::L where {L<:Union{Layer,Nothing}}
    nextLayers::Array{Layer,1}

    function Conv2D(c,
                    f::Tuple{Integer, Integer};
                    prevLayer=nothing,
                    strides::Tuple{Integer, Integer}=(1, 1),
                    padding::Symbol=:valid,
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
            Array{T,3}(undef,0,0,0), #B
            Array{T,3}(undef,0,0,0), #dB
            activation,
            keepProb,
            Array{T,4}(undef, 0,0,0,0), #Z
            Array{T,4}(undef, 0,0,0,0), #dZ
            Array{T,4}(undef, 0,0,0,0), #A
            Dict(:dw=>[Array{T,3}(undef,0,0,0) for i=1:c],
                 :db=>Array{T,3}(undef,0,0,0)), #V
            Dict(:dw=>[Array{T,3}(undef,0,0,0) for i=1:c],
                 :db=>Array{T,3}(undef,0,0,0)), #S
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
    f::Integer

    """
        stides size
    """
    s::Integer

    padding::Symbol

    W::Array{Array{F, 2},1}  where {F}
    dW::Array{Array{F,2},1}  where {F}



    B::Array{F,2} where {F}
    dB::Array{F,2} where {F}

    actFun::Symbol
    keepProb::AbstractFloat

    Z::Array{T,3} where {T}
    dZ::Array{T,3} where {T}
    A::Array{T,3} where {T}

    V::Dict{Symbol, Array}
    S::Dict{Symbol, Array}

    forwCount::Integer
    backCount::Integer
    updateCount::Integer

    prevLayer::L where {L<:Union{Layer,Nothing}}
    nextLayers::Array{Layer,1}

    function Conv1D(c,
                    f::Integer;
                    prevLayer=nothing,
                    strides::Integer=1,
                    padding::Symbol=:valid,
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
            Array{T,2}(undef,0,0), #B
            Array{T,2}(undef,0,0), #dB
            activation,
            keepProb,
            Array{T,3}(undef, 0,0,0), #Z
            Array{T,3}(undef, 0,0,0), #dZ
            Array{T,3}(undef, 0,0,0), #A
            Dict(:dw=>[Array{T,2}(undef,0,0) for i=1:c],
                 :db=>Array{T,2}(undef,0,0)), #V
            Dict(:dw=>[Array{T,2}(undef,0,0) for i=1:c],
                 :db=>Array{T,2}(undef,0,0)), #S
            0, #forwCount
            0, #backCount
            0, #updateCount
            prevLayer,
            Array{Layer,1}(undef, 0)  #nextLayer
            )


    end #function Conv1D

end #mutable struct Conv1D

export Conv1D



mutable struct Conv3D <: ConvLayer
    channels::Integer

    """
        filter size
    """
    f::Tuple{Integer, Integer, Integer}

    """
        stides size
    """
    s::Tuple{Integer, Integer, Integer}

    padding::Symbol

    W::Array{Array{F, 4},1}  where {F}
    dW::Array{Array{F,4},1}  where {F}



    B::Array{F,4} where {F}
    dB::Array{F,4} where {F}

    actFun::Symbol
    keepProb::AbstractFloat

    Z::Array{T,5} where {T}
    dZ::Array{T,5} where {T}
    A::Array{T,5} where {T}

    V::Dict{Symbol, Array}
    S::Dict{Symbol, Array}

    forwCount::Integer
    backCount::Integer
    updateCount::Integer

    prevLayer::L where {L<:Union{Layer,Nothing}}
    nextLayers::Array{Layer,1}

    function Conv3D(c,
                    f::Tuple{Integer, Integer, Integer};
                    prevLayer=nothing,
                    strides::Tuple{Integer, Integer, Integer}=(1, 1, 1),
                    padding::Symbol=:valid,
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
            [Array{T,4}(undef,0,0,0,0) for i=1:c], #W
            [Array{T,4}(undef,0,0,0,0) for i=1:c], #dW
            Array{T,4}(undef,0,0,0,0), #B
            Array{T,4}(undef,0,0,0,0), #dB
            activation,
            keepProb,
            Array{T,5}(undef, 0,0,0,0,0), #Z
            Array{T,5}(undef, 0,0,0,0,0), #dZ
            Array{T,5}(undef, 0,0,0,0,0), #A
            Dict(:dw=>[Array{T,4}(undef,0,0,0,0) for i=1:c],
                 :db=>Array{T,4}(undef,0,0,0,0)), #V
            Dict(:dw=>[Array{T,4}(undef,0,0,0,0) for i=1:c],
                 :db=>Array{T,4}(undef,0,0,0,0)), #S
            0, #forwCount
            0, #backCount
            0, #updateCount
            prevLayer,
            Array{Layer,1}(undef, 0)  #nextLayer
            )


    end #function Conv3D

end #mutable struct Conv3D

export Conv3D


###  Pooling layers

abstract type PoolLayer <: PaddableLayer end

export PoolLayer

abstract type MaxPoolLayer <: PoolLayer end

export MaxPoolLayer

mutable struct MaxPool2D <: MaxPoolLayer
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

    dZ::Array{T,4} where {T}
    A::Array{T,4} where {T}

    forwCount::Integer
    backCount::Integer
    updateCount::Integer

    prevLayer::L where {L<:Union{Layer,Nothing}}
    nextLayers::Array{Layer,1}

    function MaxPool2D(f::Tuple{Integer,Integer}=(2,2);
                       prevLayer=nothing,
                       strides::S=nothing,
                       padding::Symbol=:valid,
                       ) where {S <: Union{Tuple{Integer,Integer}, Nothing}}

        if prevLayer == nothing
           T = Any
        elseif prevLayer isa Array
           T = eltype(prevLayer)
        else
           T = eltype(prevLayer.W)
        end

        if strides == nothing
            strides = f
        end

        return new(
                   0, #channels
                   f,
                   strides,
                   padding,
                   Array{T,4}(undef,0), #dZ
                   Array{T,4}(undef,0), #A
                   0, #forwCount
                   0, #backCount
                   0, #updateCount
                   prevLayer,
                   Array{Layer,1}(undef,0),
                   )
    end #function MaxPool2D

end #mutable struct MaxPool2D

export MaxPool2D

mutable struct MaxPool1D <: MaxPoolLayer
    channels::Integer

    """
        filter size
    """
    f::Integer

    """
        stides size
    """
    s::Integer

    padding::Symbol

    dZ::Array{T,3} where {T}
    A::Array{T,3} where {T}

    forwCount::Integer
    backCount::Integer
    updateCount::Integer

    prevLayer::L where {L<:Union{Layer,Nothing}}
    nextLayers::Array{Layer,1}

    function MaxPool1D(f::Integer=2;
                       prevLayer=nothing,
                       strides::S=nothing,
                       padding::Symbol=:valid,
                       ) where {S <: Union{Integer, Nothing}}

        if prevLayer == nothing
           T = Any
        elseif prevLayer isa Array
           T = eltype(prevLayer)
        else
           T = eltype(prevLayer.W)
        end

        if strides == nothing
            strides = f
        end

        return new(
                   0, #channels
                   f,
                   strides,
                   padding,
                   Array{T,3}(undef,0), #dZ
                   Array{T,3}(undef,0), #A
                   0, #forwCount
                   0, #backCount
                   0, #updateCount
                   prevLayer,
                   Array{Layer,1}(undef,0),
                   )
    end #function MaxPool1D

end #mutable struct MaxPool1D

export MaxPool1D

mutable struct MaxPool3D <: MaxPoolLayer
    channels::Integer

    """
        filter size
    """
    f::Tuple{Integer, Integer, Integer}

    """
        stides size
    """
    s::Tuple{Integer, Integer, Integer}

    padding::Symbol

    dZ::Array{T,5} where {T}
    A::Array{T,5} where {T}

    forwCount::Integer
    backCount::Integer
    updateCount::Integer

    prevLayer::L where {L<:Union{Layer,Nothing}}
    nextLayers::Array{Layer,1}

    function MaxPool3D(f::Tuple{Integer,Integer,Integer}=(2,2,2);
                       prevLayer=nothing,
                       strides::S=nothing,
                       padding::Symbol=:valid,
                       ) where {S <: Union{Tuple{Integer,Integer,Integer}, Nothing}}

        if prevLayer == nothing
           T = Any
        elseif prevLayer isa Array
           T = eltype(prevLayer)
        else
           T = eltype(prevLayer.W)
        end

        if strides == nothing
            strides = f
        end

        return new(
                   0, #channels
                   f,
                   strides,
                   padding,
                   Array{T,5}(undef,0), #dZ
                   Array{T,5}(undef,0), #A
                   0, #forwCount
                   0, #backCount
                   0, #updateCount
                   prevLayer,
                   Array{Layer,1}(undef,0),
                   )
    end #function MaxPool3D

end #mutable struct MaxPool3D

export MaxPool3D


#### AveragePoolLayer

abstract type AveragePoolLayer <: PoolLayer end

export AveragePoolLayer

mutable struct AveragePool2D <: AveragePoolLayer
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

    dZ::Array{T,4} where {T}
    A::Array{T,4} where {T}

    forwCount::Integer
    backCount::Integer
    updateCount::Integer

    prevLayer::L where {L<:Union{Layer,Nothing}}
    nextLayers::Array{Layer,1}

    function AveragePool2D(f::Tuple{Integer,Integer}=(2,2);
                       prevLayer=nothing,
                       strides::S=nothing,
                       padding::Symbol=:valid,
                       ) where {S <: Union{Tuple{Integer,Integer}, Nothing}}

        if prevLayer == nothing
           T = Any
        elseif prevLayer isa Array
           T = eltype(prevLayer)
        else
           T = eltype(prevLayer.W)
        end

        if strides == nothing
            strides = f
        end

        return new(
                   0, #channels
                   f,
                   strides,
                   padding,
                   Array{T,4}(undef,0), #dZ
                   Array{T,4}(undef,0), #A
                   0, #forwCount
                   0, #backCount
                   0, #updateCount
                   prevLayer,
                   Array{Layer,1}(undef,0),
                   )
    end #function AveragePool2D

end #mutable struct AveragePool2D

export AveragePool2D

mutable struct AveragePool1D <: AveragePoolLayer
    channels::Integer

    """
        filter size
    """
    f::Integer

    """
        stides size
    """
    s::Integer

    padding::Symbol

    dZ::Array{T,3} where {T}
    A::Array{T,3} where {T}

    forwCount::Integer
    backCount::Integer
    updateCount::Integer

    prevLayer::L where {L<:Union{Layer,Nothing}}
    nextLayers::Array{Layer,1}

    function AveragePool1D(f::Integer=2;
                       prevLayer=nothing,
                       strides::S=nothing,
                       padding::Symbol=:valid,
                       ) where {S <: Union{Integer, Nothing}}

        if prevLayer == nothing
           T = Any
        elseif prevLayer isa Array
           T = eltype(prevLayer)
        else
           T = eltype(prevLayer.W)
        end

        if strides == nothing
            strides = f
        end

        return new(
                   0, #channels
                   f,
                   strides,
                   padding,
                   Array{T,3}(undef,0), #dZ
                   Array{T,3}(undef,0), #A
                   0, #forwCount
                   0, #backCount
                   0, #updateCount
                   prevLayer,
                   Array{Layer,1}(undef,0),
                   )
    end #function AveragePool1D

end #mutable struct AveragePool1D

export AveragePool1D

mutable struct AveragePool3D <: AveragePoolLayer
    channels::Integer

    """
        filter size
    """
    f::Tuple{Integer, Integer, Integer}

    """
        stides size
    """
    s::Tuple{Integer, Integer, Integer}

    padding::Symbol

    dZ::Array{T,5} where {T}
    A::Array{T,5} where {T}

    forwCount::Integer
    backCount::Integer
    updateCount::Integer

    prevLayer::L where {L<:Union{Layer,Nothing}}
    nextLayers::Array{Layer,1}

    function AveragePool3D(f::Tuple{Integer,Integer,Integer}=(2,2,2);
                       prevLayer=nothing,
                       strides::S=nothing,
                       padding::Symbol=:valid,
                       ) where {S <: Union{Tuple{Integer,Integer,Integer}, Nothing}}

        if prevLayer == nothing
           T = Any
        elseif prevLayer isa Array
           T = eltype(prevLayer)
        else
           T = eltype(prevLayer.W)
        end

        if strides == nothing
            strides = f
        end

        return new(
                   0, #channels
                   f,
                   strides,
                   padding,
                   Array{T,5}(undef,0), #dZ
                   Array{T,5}(undef,0), #A
                   0, #forwCount
                   0, #backCount
                   0, #updateCount
                   prevLayer,
                   Array{Layer,1}(undef,0),
                   )
    end #function AveragePool3D

end #mutable struct AveragePool3D

export AveragePool3D
