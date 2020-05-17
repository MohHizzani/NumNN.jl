
"""
    abstract type to include all layers
"""
abstract type Layer end

export Layer


### FCLayer

@doc raw"""
    FCLayer(channels=0, actFun=:noAct, [layerInput = nothing; keepProb = 1.0])

Fully-connected layer (equivalent to Dense in TensorFlow etc.)

# Arguments

- `channels` := (`Integer`) is the number of nodes in the layer

- `actFun` := (`Symbol`) is the activation function of this layer

- `layerInput` := (`Layer` or `Array`) the input of this array (optional don't need to assign it)

- `keepProb` := (`AbstractFloat`) the keep probability (1 - prob of the dropout rate)

---------

# Summary

    mutable struct FCLayer <: Layer

-------

# Fields

- `channels::Integer` := is the number of nodes in the layer

- `actFun::Symbol` := the activation function of this layer

- `inputS::Tuple{Integer, Integer}` := input size of the layer, of the shape (channels of the previous layer, size of mini-batch)

- `outputS::Tuple{Integer, Integer}` := output size of the layer, of the shape (channels of this layer, size of mini-batch)

- `keepProb::AbstractFloat` := the keep probability (rate) of the drop-out operation `<1.0`

- `W::Array{T,2} where {T}` := the scaling parameters of this layer `W * X`, of the shape (channels of this layer, channels of the previous layer)

- `B::Array{T,2} where {T}` := the bias of this layer `W * X .+ B`, of the shape (channels of this layer, 1)

- `dW::Array{T,2} where {T}` := the derivative of the loss function to the W parameters $\frac{dJ}{dW}$

- `dB::Array{T,2} where {T}` := the derivative of the loss function to the B parameters $\frac{dJ}{dB}$

- `forwCount::Integer` := forward propagation counter

- `backCount::Integer` := backward propagation counter

- `updateCount::Integer` := update parameters counter

- `prevLayer::L where {L<:Union{Layer,Nothing}}` := the previous layer which is
the input of this layer

- `nextLayers::Array{Layer,1}` := An array of the next `layer`(s)

---------

# Supertype Hierarchy

    FCLayer <: Layer <: Any

---------

# Examples

```julia
X_Input = Input(X_train)
X = FCLayer(20, :relu)(X_Input)

```

In the previous example the variable `X_Input` is a pointer to the `Input` layer, and `X` is an pointer to the `FCLayer(20, :relu)` layer.
**Note** that the layer instance can be used as a connecting function.

"""
mutable struct FCLayer <: Layer
    channels::Integer
    actFun::Symbol

    inputS::Tuple{Integer}
    outputS::Tuple{Integer}

    """
        drop-out keeping node probability
    """
    keepProb::AbstractFloat
    W::Array{T,2} where {T}
    B::Array{T,2} where {T}
    dW::Array{T,2} where {T}
    dB::Array{T,2} where {T}
    # ### adding Z & A place holder for recursive calling
    # ### and a counter for how many it was called
    # Z::Array{T,2} where {T}
    # dA::Array{T,2} where {T}
    # A::Array{T,2} where {T}

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
    function FCLayer(channels=0, actFun=:noAct, layerInput = nothing; keepProb = 1.0)
        # W, B
        if isa(layerInput, Layer)
            T = eltype(layerInput.W)
            nl = channels
            nl_1 = size(layerInput.W)[1]
            prevLayer = layerInput
        elseif isa(layerInput, Array)
            T = eltype(layerInput)
            nl = channels
            nl_1 = size(layerInput)[1]
            prevLayer = nothing
        else
            T = Any
            nl = 0
            nl_1 = 0
            prevLayer = nothing
        end
        new(
            channels,
            actFun,
            (0,), #inputS
            (0,), #outputS
            keepProb,
            Matrix{T}(undef, 0, 0),
            Matrix{T}(undef, 0, 0),
            Matrix{T}(undef, 0, 0),
            Matrix{T}(undef, 0, 0),
            # Matrix{T}(undef, 0, 0),
            # Matrix{T}(undef, 0, 0),
            # Matrix{T}(undef, 0, 0),
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

Base.show(io::IO, l::FCLayer) = print(io, "FCLayer($(l.channels), $(l.actFun), $(l.inputS), $(l.outputS), ...)")

export FCLayer

### Mulit-Input Layer MILayer

export MILayer

abstract type MILayer <: Layer end

### AddLayer

@doc raw"""
    AddLayer(; [channels = 0])

Layer performs and addition of multiple previous layers

# Arguments

- `channels` := (`Integer`) number of channels/nodes of this array which equals to the same of the previous layer(s)

---------

# Summary

    mutable struct AddLayer <: MILayer

# Fields

- `channels::Integer` := is the number of nodes or `channels` in the layer

- `inputS::Tuple` := input size of the layer

- `outputS::Tuple` := output size of the layer

- `forwCount::Integer` := forward propagation counter

- `backCount::Integer` := backward propagation counter

- `updateCount::Integer` := update parameters counter

- `nextLayers::Array{Layer,1}` := An array of the next `layer`(s)

- `prevLayer::Array{Layer,1}` := An array of the previous `layer`(s) to be added

---------

# Supertype Hierarchy

    AddLayer <: MILayer <: Layer <: Any

-------

# Examples

```julia
XIn1 = Input(X_train)
X1 = FCLayer(10, :relu)(XIn1)
XIn2 = Input(X_train)
X2 = FCLayer(10, :tanh)(XIn2)

Xa = AddLayer()([X1,X2])
```

"""
mutable struct AddLayer <: MILayer
    channels::Integer

    inputS::Tuple
    outputS::Tuple


    # A::Array{T,N} where {T,N}
    # dA::Array{T,N} where {T,N}

    forwCount::Integer
    backCount::Integer
    updateCount::Integer

    nextLayers::Array{Layer,1}
    prevLayer::Array{Layer,1}
    function AddLayer(; channels = 0)
        # channels = l1.channels
        # T = eltype(l1.W)
        new(
            channels,
            (0,), #inputS
            (0,), #outputS
            # Matrix{Nothing}(undef, 0, 0),
            # Matrix{Nothing}(undef, 0, 0),
            0,
            0,
            0,
            Array{Layer,1}(undef,0), #nextLayers
            Array{Layer,1}(undef,0), #prevLayer
            )
    end #function AddLayer
end

Base.show(io::IO, l::AddLayer) = print(io, "AddLayer($(l.channels), $(l.inputS), $(l.outputS), ...)")

export AddLayer


### ConcatLayer

export ConcatLayer


@doc raw"""
    ConcatLayer(; channels = 0)

Perform concatenation of group of previous `Layer`s

# Summary

    mutable struct ConcatLayer <: MILayer

# Fields

    channels    :: Integer
    inputS      :: Tuple
    outputS     :: Tuple
    forwCount   :: Integer
    backCount   :: Integer
    updateCount :: Integer
    nextLayers  :: Array{Layer,1}
    prevLayer   :: Array{Layer,1}

# Supertype Hierarchy

    ConcatLayer <: MILayer <: Layer <: Any
"""
mutable struct ConcatLayer <: MILayer
    channels::Integer

    inputS::Tuple
    outputS::Tuple


    # A::Array{T,N} where {T,N}
    # dA::Array{T,N} where {T,N}

    forwCount::Integer
    backCount::Integer
    updateCount::Integer

    nextLayers::Array{Layer,1}
    prevLayer::Array{Layer,1}

    LSlice::Dict{Layer,UnitRange{Integer}}
    function ConcatLayer(; channels = 0)
        # channels = l1.channels
        # T = eltype(l1.W)
        new(
            channels,
            (0,), #inputS
            (0,), #outputS
            # Matrix{Nothing}(undef, 0, 0),
            # Matrix{Nothing}(undef, 0, 0),
            0,
            0,
            0,
            Array{Layer,1}(undef,0), #nextLayers
            Array{Layer,1}(undef,0), #prevLayer
            Dict{Layer,UnitRange{Integer}}(),
            )
    end #function ConcatLayer
end #mutable struct ConcatLayer

Base.show(io::IO, l::ConcatLayer) = print(io, "ConcatLayer($(l.channels), $(l.inputS), $(l.outputS), ...)")

### Activation

@doc raw"""
    Activation(actFun)

# Arguments

- `actFun::Symbol` := the activation function of this layer

---------

# Summary

mutable struct Activation <: Layer

--------

# Fields

- `actFun::Symbol` := the activation function of this layer

- `channels::Integer` := is the number of nodes or `channels` in the layer

- `inputS::Tuple` := input size of the layer

- `outputS::Tuple` := output size of the layer

- `forwCount::Integer` := forward propagation counter

- `backCount::Integer` := backward propagation counter

- `updateCount::Integer` := update parameters counter

- `nextLayers::Array{Layer,1}` := An array of the next `layer`(s)

- `prevLayer::Array{Layer,1}` := An array of the previous `layer`(s) to be added

---------

# Supertype Hierarchy

    Activation <: Layer <: An

---------

# Examples

```julia
X_Input = Input(X_train)
X = FCLayer(10, :noAct)(X_Input)
X = Activation(:relu)(X)
```

"""
mutable struct Activation <: Layer
    actFun::Symbol
    channels::Integer

    inputS::Tuple
    outputS::Tuple

    forwCount::Integer
    backCount::Integer
    updateCount::Integer

    nextLayers::Array{Layer,1}
    prevLayer::L where {L<:Union{Layer,Nothing}}

    function Activation(actFun=:relu)
        new(
            actFun,
            0, #channels
            (0,), #inputS
            (0,), #outputS
            0,
            0,
            0,
            Array{Layer,1}(undef,0),
            nothing,
            )
    end #function Activation
end #mutable struct Activation

Base.show(io::IO, l::Activation) = print(io, "Activation($(l.channels), $(l.actFun), $(l.inputS), $(l.outputS), ...)")

export Activation



### Input

@doc raw"""
    Input(X_shape::Tuple)

`Input` `Layer` that is used as a pointer to the input array(s).

# Arguments

- `X_shape::Tuple` := shape of the input Array

-------

# Summary

    mutable struct Input <: Layer

# Fields

- `channels::Integer` := is the number of nodes or `channels` in the layer

- `inputS::Tuple` := input size of the layer

- `outputS::Tuple` := output size of the layer

- `forwCount::Integer` := forward propagation counter

- `backCount::Integer` := backward propagation counter

- `updateCount::Integer` := update parameters counter

- `nextLayers::Array{Layer,1}` := An array of the next `layer`(s)

- `prevLayer::Array{Layer,1}` := An array of the previous `layer`(s) to be added


------

# Supertype Hierarchy

    Input <: Layer <: Any

--------

# Examples

```julia
X_Input = Input(size(X_train))
X = FCLayer(10, :relu)(X_Input)
```

It is possible to use the Array instead of its size `NumNN` will take care of the rest

```julia
X_Input = Input(X_train)
X = FCLayer(10, :relu)(X_Input)
```

"""
mutable struct Input <: Layer
    channels::Integer

    inputS::Tuple
    outputS::Tuple

    forwCount::Integer
    backCount::Integer
    updateCount::Integer

    nextLayers::Array{Layer,1}
    prevLayer::L where {L<:Union{Layer,Nothing}}

    function Input(X_shape::Tuple)
        N = length(X_shape)
        channels = X_shape[end-1]
        # if N==2
        #     channels = X_shape[1]
        # elseif N==3
        #     channels = X_shape[2]
        # elseif N==4
        #     channels = X_shape[3]
        # elseif N==5
        #     channels = X_shape[4]
        # end

        new(
            channels,
            X_shape[1:end-1], #inputS
            X_shape[1:end-1], #outputS
            0,
            0,
            0,
            Array{Layer,1}(undef,0),
            nothing,
            )
    end #function Layer
end #mutable struct Input

function Input(X::Array{T,N}) where {T,N}
    X_shape = size(X)
    Input(X_shape)
end #function Input(X::Array{T,N}) where {T,N}

Base.show(io::IO, l::Input) = print(io, "Input($(l.channels), $(l.inputS), $(l.outputS), ...)")

export Input


### BatchNorm

@doc raw"""
    BatchNorm(;dim=1, ϵ=1e-10)

Batch Normalization `Layer` that is used ot normalize across the dimensions specified by the argument `dim`.

# Arguments

- `dim::Integer` := is the dimension to normalize across

- `ϵ::AbstractFloat` := is a backup constant that is used to prevent from division on zero when ``σ^2`` is zero

---------

# Summary

    mutable struct BatchNorm <: Layer

-------

# Fields

- `channels::Integer` := is the number of nodes in the layer

- `inputS::Tuple{Integer, Integer}` := input size of the layer, of the shape (channels of the previous layer, size of mini-batch)

- `outputS::Tuple{Integer, Integer}` := output size of the layer, of the shape (channels of this layer, size of mini-batch)

- `dim::Integer` := the dimension to normalize across

- `ϵ::AbstractFloat` := backup constant to protect from dividing on zero when ``σ^2 = 0``

- `W::Array{T,2} where {T}` := the scaling parameters of this layer `W * X`, same shape of the mean `μ`

- `B::Array{T,2} where {T}` := the bias of this layer `W * X .+ B`, same shape of the variance ``σ^2``

- `dW::Array{T,2} where {T}` := the derivative of the loss function to the W parameters $\frac{dJ}{dW}$

- `dB::Array{T,2} where {T}` := the derivative of the loss function to the B parameters $\frac{dJ}{dB}$

- `forwCount::Integer` := forward propagation counter

- `backCount::Integer` := backward propagation counter

- `updateCount::Integer` := update parameters counter

- `prevLayer::L where {L<:Union{Layer,Nothing}}` := the previous layer which is
the input of this layer

- `nextLayers::Array{Layer,1}` := An array of the next `layer`(s)

---------

# Supertype Hierarchy

    BatchNorm <: Layer <: Any

---------

# Examples

```julia
X_train = rand(14,14,3,32) #input of shape `14×14` with channels of `3` and mini-batch size `32`

X_Input = Input(X_train)
X = Conv2D(10, (3,3))(X_Input)
X = BatchNorm(dim=3) #to normalize across the channels dimension
X = Activation(:relu)
```

```julia
X_train = rand(128,5,32) #input of shape `128` with channels of `5` and mini-batch size `32`

X_Input = Input(X_train)
X = Conv1D(10, 5)(X_Input)
X = BatchNorm(dim=2) #to normalize across the channels dimension
X = Activation(:relu)
``


```julia
X_train = rand(64*64,32) #input of shape `64*64` and mini-batch size `32`

X_Input = Input(X_train)
X = FCLayer(10, :noAct)(X_Input)
X = BatchNorm(dim=1) #to normalize across the features dimension
X = Activation(:relu)
````

"""
mutable struct BatchNorm <: Layer
    channels::Integer

    inputS::Tuple
    outputS::Tuple

    dim::Integer

    ϵ::AbstractFloat

    W::Array{T, N} where {T,N}
    dW::Array{T, N} where {T,N}

    B::Array{T, N} where {T,N}
    dB::Array{T, N} where {T,N}

    V::Dict{Symbol,Array}
    S::Dict{Symbol,Array}

    forwCount::Integer
    backCount::Integer
    updateCount::Integer

    nextLayers::Array{Layer,1}
    prevLayer::L where {L<:Union{Layer,Nothing}}


    function BatchNorm(;dim=1, ϵ=1e-7)
        new(
            0, #channels
            (0,), #inputS
            (0,), #outputS
            dim,
            ϵ,
            Array{Any,1}(undef,0), #W
            Array{Any,1}(undef,0), #dW
            Array{Any,1}(undef,0), #B
            Array{Any,1}(undef,0), #dB
            Dict(:dw=>Array{Any,1}(undef,0),
                 :db=>Array{Any,1}(undef,0)), #V
            Dict(:dw=>Array{Any,1}(undef,0),
                 :db=>Array{Any,1}(undef,0)), #S
            0, #forwCount
            0, #backCount
            0, #updateCount
            Array{Layer,1}(undef,0), #nextLayers
            nothing, #prevLayer
            )
    end #function BatchNorm


end #mutable struct BatchNorm

Base.show(io::IO, l::BatchNorm) = print(io, "BatchNorm($(l.channels), dim = $(l.dim), $(l.inputS), $(l.outputS), ...)")

export BatchNorm

### Flatten

export Flatten


@doc raw"""
    Flatten()

Flatten the input into 2D `Array`

# Summary

    mutable struct Flatten <: Layer

# Fields

    channels    :: Integer
    inputS      :: Tuple
    outputS     :: Tuple
    forwCount   :: Integer
    backCount   :: Integer
    updateCount :: Integer
    nextLayers  :: Array{Layer,1}
    prevLayer   :: Union{Nothing, Layer}

# Supertype Hierarchy

    Flatten <: Layer <: Any
"""
mutable struct Flatten <: Layer
    channels::Integer

    inputS::Tuple
    outputS::Tuple

    forwCount::Integer
    backCount::Integer
    updateCount::Integer

    nextLayers::Array{Layer,1}
    prevLayer::L where {L<:Union{Layer,Nothing}}

    function Flatten()

        return new(
            0, #channels
            (0,), #inputS
            (0,), #outputS
            0, #forwCount
            0, #backCount
            0, #updateCount
            Array{Layer,1}(undef,0), #nextLayers
            nothing,
        )
    end

end

Base.show(io::IO, l::Flatten) = print(io, "Flatten($(l.channels), $(l.inputS), $(l.outputS), ...)")

### Model

@doc raw"""

    function Model(
        X,
        Y,
        inLayer::Layer,
        outLayer::Layer,
        α;
        optimizer = :gds,
        β1 = 0.9,
        β2 = 0.999,
        ϵAdam = 1e-8,
        regulization = 0,
        λ = 1.0,
        lossFun = :categoricalCrossentropy,
        paramsDtype::DataType = Float64,
    )

# Summary

    mutable struct Model <: Any

# Fields

    inLayer      :: Layer
    outLayer     :: Layer
    lossFun      :: Symbol
    paramsDtype  :: DataType
    regulization :: Integer
    λ            :: AbstractFloat
    α            :: AbstractFloat
    optimizer    :: Symbol
    ϵAdam        :: AbstractFloat
    β1           :: AbstractFloat
    β2           :: AbstractFloat
"""
mutable struct Model
    # layers::Array{Layer,1}
    inLayer::Layer
    outLayer::Layer
    lossFun::Symbol

    paramsDtype::DataType

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
        inLayer::Layer,
        outLayer::Layer,
        α;
        optimizer = :gds,
        β1 = 0.9,
        β2 = 0.999,
        ϵAdam = 1e-8,
        regulization = 0,
        λ = 1.0,
        lossFun = :categoricalCrossentropy,
        paramsDtype::DataType = eltype(X),
    )

        deepInitWB!(outLayer; dtype = paramsDtype)
        resetCount!(outLayer, :forwCount)
        deepInitVS!(outLayer, optimizer)
        resetCount!(outLayer, :forwCount)
        @assert regulization in [0, 1, 2]
        return new(
            inLayer,
            outLayer,
            lossFun,
            paramsDtype,
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
