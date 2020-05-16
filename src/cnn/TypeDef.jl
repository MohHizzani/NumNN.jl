
@doc raw"""
# Summary

    abstract type PaddableLayer <: Layer

Abstract Type to hold all Paddable Layers (i.e.  `ConvLayer` & `PoolLayer`)

# Subtypes

    ConvLayer
    PoolLayer

# Supertype Hierarchy

    PaddableLayer <: Layer <: Any
"""
abstract type PaddableLayer <: Layer end

@doc raw"""
# Summary

    abstract type ConvLayer <: PaddableLayer

Abstract Type to hold all ConvLayer

# Subtypes

    Conv1D
    Conv2D
    Conv3D

# Supertype Hierarchy

    ConvLayer <: PaddableLayer <: Layer <: Any
"""
abstract type ConvLayer <: PaddableLayer end

export ConvLayer

### Convolution layers

@doc raw"""
# Summary

    mutable struct Conv2D <: ConvLayer

# Fields

    channels    :: Integer
    f           :: Tuple{Integer,Integer}
    s           :: Tuple{Integer,Integer}
    inputS      :: Tuple
    outputS     :: Tuple
    padding     :: Symbol
    W           :: Array{F,4} where F
    dW          :: Array{F,4} where F
    K           :: Array{F,2} where F
    dK          :: Array{F,2} where F
    B           :: Array{F,4} where F
    dB          :: Array{F,4} where F
    actFun      :: Symbol
    keepProb    :: AbstractFloat
    V           :: Dict{Symbol,Array{F,4} where F}
    S           :: Dict{Symbol,Array{F,4} where F}
    V̂dk         :: Array{F,2} where F
    Ŝdk         :: Array{F,2} where F
    forwCount   :: Integer
    backCount   :: Integer
    updateCount :: Integer
    prevLayer   :: Union{Nothing, Layer}
    nextLayers  :: Array{Layer,1}

# Supertype Hierarchy

    Conv2D <: ConvLayer <: PaddableLayer <: Layer <: Any
"""
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

    inputS::Tuple
    outputS::Tuple


    """
        Padding mode `:valid` or `:same`
    """
    padding::Symbol

    W::Array{F, 4} where {F}
    dW::Array{F,4} where {F}

    """
        Unrolled filters
    """
    K::Array{F, 2} where {F}
    dK::Array{F, 2} where {F}

    B::Array{F, 4} where {F}
    dB::Array{F, 4} where {F}

    actFun::Symbol
    keepProb::AbstractFloat

    # Z::Array{T,4} where {T}
    #
    # dA::Array{T,4} where {T}
    # A::Array{T,4} where {T}

    V::Dict{Symbol, Array{F, 4}  where {F}}
    S::Dict{Symbol, Array{F, 4}  where {F}}

    V̂dk::Array{F, 2} where {F}
    Ŝdk::Array{F, 2} where {F}

    forwCount::Integer
    backCount::Integer
    updateCount::Integer

    prevLayer::L where {L<:Union{Layer,Nothing}}
    nextLayers::Array{Layer,1}

    function Conv2D(
        c,
        f::Tuple{Integer, Integer};
        prevLayer=nothing,
        strides::Tuple{Integer, Integer}=(1, 1),
        padding::Symbol=:valid,
        activation::Symbol=:noAct,
        keepProb=1.0
    )

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
            (0,), #inputS
            (0,), #outputS
            padding,
            Array{T,4}(undef,0,0,0,0), #W
            Array{T,4}(undef,0,0,0,0), #dW
            Matrix{T}(undef,0,0), #K
            Matrix{T}(undef,0,0), #dK
            Array{T,4}(undef,0,0,0,0), #B
            Array{T,4}(undef,0,0,0,0), #dB
            activation,
            keepProb,
            # Array{T,4}(undef, 0,0,0,0), #Z
            # Array{T,4}(undef, 0,0,0,0), #dA
            # Array{T,4}(undef, 0,0,0,0), #A
            Dict(:dw=>Array{T,4}(undef,0,0,0,0),
                 :db=>Array{T,4}(undef,0,0,0,0)), #V
            Dict(:dw=>Array{T,4}(undef,0,0,0,0),
                 :db=>Array{T,4}(undef,0,0,0,0)), #S
            Matrix{T}(undef,0,0), #V̂dk
            Matrix{T}(undef,0,0), #Ŝdk
            0, #forwCount
            0, #backCount
            0, #updateCount
            prevLayer,
            Array{Layer,1}(undef, 0)  #nextLayer
            )


    end #function Conv2D

end #mutable struct Conv2D

Base.show(io::IO, l::Conv2D) = print(io, "Conv2D($(l.channels), $(l.f), $(l.s), $(l.padding), $(l.actFun), $(l.inputS), $(l.outputS), ...)")

export Conv2D

@doc raw"""
# Summary

    mutable struct Conv1D <: ConvLayer

# Fields

    channels    :: Integer
    f           :: Integer
    s           :: Integer
    inputS      :: Tuple
    outputS     :: Tuple
    padding     :: Symbol
    W           :: Array{F,3} where F
    dW          :: Array{F,3} where F
    K           :: Array{F,2} where F
    dK          :: Array{F,2} where F
    B           :: Array{F,3} where F
    dB          :: Array{F,3} where F
    actFun      :: Symbol
    keepProb    :: AbstractFloat
    V           :: Dict{Symbol,Array{F,3} where F}
    S           :: Dict{Symbol,Array{F,3} where F}
    V̂dk         :: Array{F,2} where F
    Ŝdk         :: Array{F,2} where F
    forwCount   :: Integer
    backCount   :: Integer
    updateCount :: Integer
    prevLayer   :: Union{Nothing, Layer}
    nextLayers  :: Array{Layer,1}

# Supertype Hierarchy

    Conv1D <: ConvLayer <: PaddableLayer <: Layer <: Any
"""
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

    inputS::Tuple
    outputS::Tuple

    padding::Symbol

    W::Array{F, 3}  where {F}
    dW::Array{F, 3}  where {F}

    """
        Unrolled filters
    """
    K::Array{F, 2} where {F}
    dK::Array{F, 2} where {F}


    B::Array{F, 3}  where {F}
    dB::Array{F, 3}  where {F}

    actFun::Symbol
    keepProb::AbstractFloat

    # Z::Array{T,3} where {T}
    # dA::Array{T,3} where {T}
    # A::Array{T,3} where {T}

    V::Dict{Symbol, Array{F, 3}  where {F}}
    S::Dict{Symbol, Array{F, 3}  where {F}}

    V̂dk::Array{F, 2} where {F}
    Ŝdk::Array{F, 2} where {F}

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
            (0,), #inputS
            (0,), #outputS
            padding,
            Array{T,3}(undef,0,0,0), #W
            Array{T,3}(undef,0,0,0), #dW
            Matrix{T}(undef,0,0), #K
            Matrix{T}(undef,0,0), #dK
            Array{T,3}(undef,0,0,0), #B
            Array{T,3}(undef,0,0,0), #dB
            activation,
            keepProb,
            # Array{T,3}(undef, 0,0,0), #Z
            # Array{T,3}(undef, 0,0,0), #dA
            # Array{T,3}(undef, 0,0,0), #A
            Dict(:dw=>Array{T,3}(undef,0,0,0),
                 :db=>Array{T,3}(undef,0,0,0)), #V
            Dict(:dw=>Array{T,3}(undef,0,0,0),
                 :db=>Array{T,3}(undef,0,0,0)), #S
            Matrix{T}(undef,0,0), #V̂dk
            Matrix{T}(undef,0,0), #Ŝdk
            0, #forwCount
            0, #backCount
            0, #updateCount
            prevLayer,
            Array{Layer,1}(undef, 0)  #nextLayer
            )


    end #function Conv1D

end #mutable struct Conv1D

Base.show(io::IO, l::Conv1D) = print(io, "Conv1D($(l.channels), $(l.f), $(l.s), $(l.padding), $(l.actFun), $(l.inputS), $(l.outputS), ...)")

export Conv1D


@doc raw"""
# Summary

    mutable struct Conv3D <: ConvLayer

# Fields

    channels    :: Integer
    f           :: Tuple{Integer,Integer,Integer}
    s           :: Tuple{Integer,Integer,Integer}
    inputS      :: Tuple
    outputS     :: Tuple
    padding     :: Symbol
    W           :: Array{F,5} where F
    dW          :: Array{F,5} where F
    K           :: Array{F,2} where F
    dK          :: Array{F,2} where F
    B           :: Array{F,5} where F
    dB          :: Array{F,5} where F
    actFun      :: Symbol
    keepProb    :: AbstractFloat
    V           :: Dict{Symbol,Array{F,5} where F}
    S           :: Dict{Symbol,Array{F,5} where F}
    V̂dk        :: Array{F,2} where F
    Ŝdk         :: Array{F,2} where F
    forwCount   :: Integer
    backCount   :: Integer
    updateCount :: Integer
    prevLayer   :: Union{Nothing, Layer}
    nextLayers  :: Array{Layer,1}

# Supertype Hierarchy

    Conv3D <: ConvLayer <: PaddableLayer <: Layer <: Any
"""
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

    inputS::Tuple
    outputS::Tuple

    padding::Symbol

    W::Array{F, 5}  where {F}
    dW::Array{F, 5}  where {F}

    """
        Unrolled filters
    """
    K::Array{F, 2} where {F}
    dK::Array{F, 2} where {F}


    B::Array{F, 5}  where {F}
    dB::Array{F, 5}  where {F}

    actFun::Symbol
    keepProb::AbstractFloat

    # Z::Array{T,5} where {T}
    # dA::Array{T,5} where {T}
    # A::Array{T,5} where {T}

    V::Dict{Symbol, Array{F, 5}  where {F}}
    S::Dict{Symbol, Array{F, 5}  where {F}}

    V̂dk::Array{F, 2} where {F}
    Ŝdk::Array{F, 2} where {F}

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
            (0,), #inputS
            (0,), #outputS
            padding,
            Array{T,5}(undef,0,0,0,0,0), #W
            Array{T,5}(undef,0,0,0,0,0), #dW
            Matrix{T}(undef,0,0), #K
            Matrix{T}(undef,0,0), #dK
            Array{T,5}(undef,0,0,0,0,0), #B
            Array{T,5}(undef,0,0,0,0,0), #dB
            activation,
            keepProb,
            # Array{T,5}(undef, 0,0,0,0,0), #Z
            # Array{T,5}(undef, 0,0,0,0,0), #dA
            # Array{T,5}(undef, 0,0,0,0,0), #A
            Dict(:dw=>Array{T,5}(undef,0,0,0,0,0),
                 :db=>Array{T,5}(undef,0,0,0,0,0)), #V
            Dict(:dw=>Array{T,5}(undef,0,0,0,0,0),
                 :db=>Array{T,5}(undef,0,0,0,0,0)), #S
            Matrix{T}(undef,0,0), #V̂dk
            Matrix{T}(undef,0,0), #Ŝdk
            0, #forwCount
            0, #backCount
            0, #updateCount
            prevLayer,
            Array{Layer,1}(undef, 0)  #nextLayer
            )


    end #function Conv3D

end #mutable struct Conv3D

Base.show(io::IO, l::Conv3D) = print(io, "Conv3D($(l.channels), $(l.f), $(l.s), $(l.padding), $(l.actFun), $(l.inputS), $(l.outputS), ...)")

export Conv3D


###  Pooling layers

@doc raw"""
# Summary

    abstract type PoolLayer <: PaddableLayer

Abstract Type to hold all the `PoolLayer`s

# Subtypes

    AveragePoolLayer
    MaxPoolLayer

# Supertype Hierarchy

    PoolLayer <: PaddableLayer <: Layer <: Any
"""
abstract type PoolLayer <: PaddableLayer end

export PoolLayer

@doc raw"""
# Summary

    abstract type MaxPoolLayer <: PoolLayer

Abstract Type to hold all the `MaxPoolLayer`s

# Subtypes

    MaxPool1D
    MaxPool2D
    MaxPool3D

# Supertype Hierarchy

    MaxPoolLayer <: PoolLayer <: PaddableLayer <: Layer <: Any
"""
abstract type MaxPoolLayer <: PoolLayer end

export MaxPoolLayer

@doc """
    MaxPool2D(
        f::Tuple{Integer,Integer}=(2,2);
        prevLayer=nothing,
        strides::Tuple{Integer,Integer}=f,
        padding::Symbol=:valid,
    )

# Summary

    mutable struct MaxPool2D <: MaxPoolLayer

# Fields

    channels    :: Integer
    f           :: Tuple{Integer,Integer}
    s           :: Tuple{Integer,Integer}
    inputS      :: Tuple
    outputS     :: Tuple
    padding     :: Symbol
    forwCount   :: Integer
    backCount   :: Integer
    updateCount :: Integer
    prevLayer   :: Union{Nothing, Layer}
    nextLayers  :: Array{Layer,1}

# Supertype Hierarchy

    MaxPool2D <: MaxPoolLayer <: PoolLayer <: PaddableLayer <: Layer <: Any
"""
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

    inputS::Tuple
    outputS::Tuple

    padding::Symbol

    # dA::Array{T,4} where {T}
    # A::Array{T,4} where {T}

    forwCount::Integer
    backCount::Integer
    updateCount::Integer

    prevLayer::L where {L<:Union{Layer,Nothing}}
    nextLayers::Array{Layer,1}

    function MaxPool2D(
        f::Tuple{Integer,Integer}=(2,2);
        prevLayer=nothing,
        strides::Tuple{Integer,Integer}=f,
        padding::Symbol=:valid,
    )

        if prevLayer == nothing
           T = Any
        elseif prevLayer isa Array
           T = eltype(prevLayer)
        else
           T = eltype(prevLayer.W)
        end

        return new(
                   0, #channels
                   f,
                   strides,
                   (0,), #inputS
                   (0,), #outputS
                   padding,
                   # Array{T,4}(undef,0,0,0,0), #dA
                   # Array{T,4}(undef,0,0,0,0), #A
                   0, #forwCount
                   0, #backCount
                   0, #updateCount
                   prevLayer,
                   Array{Layer,1}(undef,0),
                   )
    end #function MaxPool2D

end #mutable struct MaxPool2D

Base.show(io::IO, l::MaxPool2D) = print(io, "MaxPool2D($(l.channels), $(l.f), $(l.s), $(l.padding), $(l.inputS), $(l.outputS), ...)")

export MaxPool2D

@doc """
    MaxPool1D(
        f::Integer=2;
        prevLayer=nothing,
        strides::Integer=f,
        padding::Symbol=:valid,
    )

# Summary

    mutable struct MaxPool1D <: MaxPoolLayer

# Fields

    channels    :: Integer
    f           :: Integer
    s           :: Integer
    inputS      :: Tuple
    outputS     :: Tuple
    padding     :: Symbol
    forwCount   :: Integer
    backCount   :: Integer
    updateCount :: Integer
    prevLayer   :: Union{Nothing, Layer}
    nextLayers  :: Array{Layer,1}

# Supertype Hierarchy

    MaxPool1D <: MaxPoolLayer <: PoolLayer <: PaddableLayer <: Layer <: Any
"""
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

    inputS::Tuple
    outputS::Tuple

    padding::Symbol

    # dA::Array{T,3} where {T}
    # A::Array{T,3} where {T}

    forwCount::Integer
    backCount::Integer
    updateCount::Integer

    prevLayer::L where {L<:Union{Layer,Nothing}}
    nextLayers::Array{Layer,1}

    function MaxPool1D(
        f::Integer=2;
        prevLayer=nothing,
        strides::Integer=f,
        padding::Symbol=:valid,
    )

        if prevLayer == nothing
           T = Any
        elseif prevLayer isa Array
           T = eltype(prevLayer)
        else
           T = eltype(prevLayer.W)
        end

        return new(
                   0, #channels
                   f,
                   strides,
                   (0,), #inputS
                   (0,), #outputS
                   padding,
                   # Array{T,3}(undef,0,0,0), #dA
                   # Array{T,3}(undef,0,0,0), #A
                   0, #forwCount
                   0, #backCount
                   0, #updateCount
                   prevLayer,
                   Array{Layer,1}(undef,0),
                   )
    end #function MaxPool1D

end #mutable struct MaxPool1D

Base.show(io::IO, l::MaxPool1D) = print(io, "MaxPool1D($(l.channels), $(l.f), $(l.s), $(l.padding), $(l.inputS), $(l.outputS), ...)")

export MaxPool1D

@doc """
    MaxPool3D(
        f::Tuple{Integer,Integer,Integer}=(2,2,2);
        prevLayer=nothing,
        strides::Tuple{Integer,Integer,Integer}=f,
        padding::Symbol=:valid,
    )

# Summary

    mutable struct MaxPool3D <: MaxPoolLayer

# Fields

    channels    :: Integer
    f           :: Tuple{Integer,Integer,Integer}
    s           :: Tuple{Integer,Integer,Integer}
    inputS      :: Tuple
    outputS     :: Tuple
    padding     :: Symbol
    forwCount   :: Integer
    backCount   :: Integer
    updateCount :: Integer
    prevLayer   :: Union{Nothing, Layer}
    nextLayers  :: Array{Layer,1}

# Supertype Hierarchy

    MaxPool3D <: MaxPoolLayer <: PoolLayer <: PaddableLayer <: Layer <: Any
"""
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

    inputS::Tuple
    outputS::Tuple

    padding::Symbol

    # dA::Array{T,5} where {T}
    # A::Array{T,5} where {T}

    forwCount::Integer
    backCount::Integer
    updateCount::Integer

    prevLayer::L where {L<:Union{Layer,Nothing}}
    nextLayers::Array{Layer,1}

    function MaxPool3D(
        f::Tuple{Integer,Integer,Integer}=(2,2,2);
        prevLayer=nothing,
        strides::Tuple{Integer,Integer,Integer}=f,
        padding::Symbol=:valid,
    )

        if prevLayer == nothing
           T = Any
        elseif prevLayer isa Array
           T = eltype(prevLayer)
        else
           T = eltype(prevLayer.W)
        end

        return new(
                   0, #channels
                   f,
                   strides,
                   (0,), #inputS
                   (0,), #outputS
                   padding,
                   # Array{T,5}(undef,0,0,0,0,0), #dA
                   # Array{T,5}(undef,0,0,0,0,0), #A
                   0, #forwCount
                   0, #backCount
                   0, #updateCount
                   prevLayer,
                   Array{Layer,1}(undef,0),
                   )
    end #function MaxPool3D

end #mutable struct MaxPool3D

Base.show(io::IO, l::MaxPool3D) = print(io, "MaxPool3D($(l.channels), $(l.f), $(l.s), $(l.padding), $(l.inputS), $(l.outputS), ...)")

export MaxPool3D


#### AveragePoolLayer

@doc raw"""
# Summary

    abstract type AveragePoolLayer <: PoolLayer

# Subtypes

    AveragePool1D
    AveragePool2D
    AveragePool3D

# Supertype Hierarchy

    AveragePoolLayer <: PoolLayer <: PaddableLayer <: Layer <: Any
"""
abstract type AveragePoolLayer <: PoolLayer end

export AveragePoolLayer

@doc raw"""
    AveragePool2D(
        f::Tuple{Integer,Integer}=(2,2);
        prevLayer=nothing,
        strides::Tuple{Integer,Integer}=f,
        padding::Symbol=:valid,
    )


# Summary

    mutable struct AveragePool2D <: AveragePoolLayer

# Fields

    channels    :: Integer
    f           :: Tuple{Integer,Integer}
    s           :: Tuple{Integer,Integer}
    inputS      :: Tuple
    outputS     :: Tuple
    padding     :: Symbol
    forwCount   :: Integer
    backCount   :: Integer
    updateCount :: Integer
    prevLayer   :: Union{Nothing, Layer}
    nextLayers  :: Array{Layer,1}

# Supertype Hierarchy

    AveragePool2D <: AveragePoolLayer <: PoolLayer <: PaddableLayer <: Layer <: Any
"""
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

    inputS::Tuple
    outputS::Tuple

    padding::Symbol

    # dA::Array{T,4} where {T}
    # A::Array{T,4} where {T}

    forwCount::Integer
    backCount::Integer
    updateCount::Integer

    prevLayer::L where {L<:Union{Layer,Nothing}}
    nextLayers::Array{Layer,1}

    function AveragePool2D(
        f::Tuple{Integer,Integer}=(2,2);
        prevLayer=nothing,
        strides::Tuple{Integer,Integer}=f,
        padding::Symbol=:valid,
    )

        if prevLayer == nothing
           T = Any
        elseif prevLayer isa Array
           T = eltype(prevLayer)
        else
           T = eltype(prevLayer.W)
        end

        return new(
                   0, #channels
                   f,
                   strides,
                   (0,), #inputS
                   (0,), #outputS
                   padding,
                   # Array{T,4}(undef,0,0,0,0), #dA
                   # Array{T,4}(undef,0,0,0,0), #A
                   0, #forwCount
                   0, #backCount
                   0, #updateCount
                   prevLayer,
                   Array{Layer,1}(undef,0),
                   )
    end #function AveragePool2D

end #mutable struct AveragePool2D

Base.show(io::IO, l::AveragePool2D) = print(io, "AveragePool2D($(l.channels), $(l.f), $(l.s), $(l.padding), $(l.inputS), $(l.outputS), ...)")

export AveragePool2D

@doc raw"""
    AveragePool1D(
        f::Integer=2;
        prevLayer=nothing,
        strides::Integer=f,
        padding::Symbol=:valid,
    )

# Summary

    mutable struct AveragePool1D <: AveragePoolLayer

# Fields

    channels    :: Integer
    f           :: Integer
    s           :: Integer
    inputS      :: Tuple
    outputS     :: Tuple
    padding     :: Symbol
    forwCount   :: Integer
    backCount   :: Integer
    updateCount :: Integer
    prevLayer   :: Union{Nothing, Layer}
    nextLayers  :: Array{Layer,1}

# Supertype Hierarchy

    AveragePool1D <: AveragePoolLayer <: PoolLayer <: PaddableLayer <: Layer <: Any
"""
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

    inputS::Tuple
    outputS::Tuple

    padding::Symbol

    # dA::Array{T,3} where {T}
    # A::Array{T,3} where {T}

    forwCount::Integer
    backCount::Integer
    updateCount::Integer

    prevLayer::L where {L<:Union{Layer,Nothing}}
    nextLayers::Array{Layer,1}

    function AveragePool1D(
        f::Integer=2;
        prevLayer=nothing,
        strides::Integer=f,
        padding::Symbol=:valid,
    )

        if prevLayer == nothing
           T = Any
        elseif prevLayer isa Array
           T = eltype(prevLayer)
        else
           T = eltype(prevLayer.W)
        end

        return new(
                   0, #channels
                   f,
                   strides,
                   (0,), #inputS
                   (0,), #outputS
                   padding,
                   # Array{T,3}(undef,0,0,0), #dA
                   # Array{T,3}(undef,0,0,0), #A
                   0, #forwCount
                   0, #backCount
                   0, #updateCount
                   prevLayer,
                   Array{Layer,1}(undef,0),
                   )
    end #function AveragePool1D

end #mutable struct AveragePool1D

Base.show(io::IO, l::AveragePool1D) = print(io, "AveragePool1D($(l.channels), $(l.f), $(l.s), $(l.padding), $(l.inputS), $(l.outputS), ...)")

export AveragePool1D

@doc raw"""
    AveragePool3D(
        f::Tuple{Integer,Integer,Integer}=(2,2,2);
        prevLayer=nothing,
        strides::Tuple{Integer,Integer,Integer}=f,
        padding::Symbol=:valid,
    )


# Summary

    mutable struct AveragePool3D <: AveragePoolLayer

# Fields

    channels    :: Integer
    f           :: Tuple{Integer,Integer,Integer}
    s           :: Tuple{Integer,Integer,Integer}
    inputS      :: Tuple
    outputS     :: Tuple
    padding     :: Symbol
    forwCount   :: Integer
    backCount   :: Integer
    updateCount :: Integer
    prevLayer   :: Union{Nothing, Layer}
    nextLayers  :: Array{Layer,1}

# Supertype Hierarchy

    AveragePool3D <: AveragePoolLayer <: PoolLayer <: PaddableLayer <: Layer <: Any
"""
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

    inputS::Tuple
    outputS::Tuple

    padding::Symbol

    # dA::Array{T,5} where {T}
    # A::Array{T,5} where {T}

    forwCount::Integer
    backCount::Integer
    updateCount::Integer

    prevLayer::L where {L<:Union{Layer,Nothing}}
    nextLayers::Array{Layer,1}

    function AveragePool3D(
        f::Tuple{Integer,Integer,Integer}=(2,2,2);
        prevLayer=nothing,
        strides::Tuple{Integer,Integer,Integer}=f,
        padding::Symbol=:valid,
    )

        if prevLayer == nothing
           T = Any
        elseif prevLayer isa Array
           T = eltype(prevLayer)
        else
           T = eltype(prevLayer.W)
        end

        return new(
                   0, #channels
                   f,
                   strides,
                   (0,), #inputS
                   (0,), #outputS
                   padding,
                   # Array{T,5}(undef,0,0,0,0,0), #dA
                   # Array{T,5}(undef,0,0,0,0,0), #A
                   0, #forwCount
                   0, #backCount
                   0, #updateCount
                   prevLayer,
                   Array{Layer,1}(undef,0),
                   )
    end #function AveragePool3D

end #mutable struct AveragePool3D

Base.show(io::IO, l::AveragePool3D) = print(io, "AveragePool3D($(l.channels), $(l.f), $(l.s), $(l.padding), $(l.inputS), $(l.outputS), ...)")

export AveragePool3D
