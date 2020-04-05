
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

# Parameters

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

# Examples

```julia
X_Input = Input(X_train)
X = FCLayer(20, :relu)(X_Input)
X = FCLayer(10, :softmax)(X)
```

In the previous example the variable `X_Input` is a pointer to the `Input` layer, and `X` is an pointer to the `FCLayer(20, :relu)` layer.
**Note** that the layer instance can be used as a connecting function.

"""
mutable struct FCLayer <: Layer
    channels::Integer
    actFun::Symbol

    inputS::Tuple{Integer, Integer}
    outputS::Tuple{Integer, Integer}

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

    # V::Dict{Symbol,Array{T,2} where {T,N}}
    # S::Dict{Symbol,Array{T,2} where {T,N}}

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
            Matrix{T}(undef, 0, 0),
            # Matrix{T}(undef, 0, 0),
            # Matrix{T}(undef, 0, 0),
            # Dict(:dw => Matrix{T}(undef, 0, 0), :db => Matrix{T}(undef, 0, 0)),
            # Dict(:dw => Matrix{T}(undef, 0, 0), :db => Matrix{T}(undef, 0, 0)),
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

@doc raw"""
    AddLayer(; [channels = 0])

Layer performs and addition of multiple previous layers

# Arguments

- `channels` := (`Integer`) number of channels/nodes of this array which equals to the same of the previous layer(s)

---------

# Parameters

- `channels::Integer` := is the number of nodes or `channels` in the layer

- `inputS::Tuple` := input size of the layer

- `outputS::Tuple` := output size of the layer

- `forwCount::Integer` := forward propagation counter

- `backCount::Integer` := backward propagation counter

- `updateCount::Integer` := update parameters counter

- `nextLayers::Array{Layer,1}` := An array of the next `layer`(s)

- `prevLayer::Array{Layer,1}` := An array of the previous `layer`(s) to be added

---------

# Examples

```julia
XIn1 = Input(X_train)
X1 = FCLayer(10, :relu)(XIn1)
XIn2 = Input(X_train)
X2 = FCLayer(10, :tanh)(XIn2)

Xa = AddLayer()([X1,X2])
```

"""
mutable struct AddLayer <: Layer
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
@doc raw"""
    Activation(actFun)

# Arguments

- `actFun::Symbol` := the activation function of this layer

---------

# Parameters

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


export Activation



### Input

@doc raw"""
    Input(X_shape::Tuple)

`Input` `Layer` that is used as a pointer to the input array(s).

# Arguments

- `X_shape::Tuple` := shape of the input Array

-------

# Parameters

- `channels::Integer` := is the number of nodes or `channels` in the layer

- `inputS::Tuple` := input size of the layer

- `outputS::Tuple` := output size of the layer

- `forwCount::Integer` := forward propagation counter

- `backCount::Integer` := backward propagation counter

- `updateCount::Integer` := update parameters counter

- `nextLayers::Array{Layer,1}` := An array of the next `layer`(s)

- `prevLayer::Array{Layer,1}` := An array of the previous `layer`(s) to be added


------

# Examples

```julia
X_Input = Input(size(X_train))
X = FCLayer(10, :relu)(X_Input)
```

It is possible to use the Array instead of its size NumNN will take care of the rest

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
        if N==2
            channels = X_shape[1]
        elseif N==3
            channels = X_shape[2]
        elseif N==4
            channels = X_shape[3]
        elseif N==5
            channels = X_shape[4]
        end

        new(
            channels,
            X_shape, #inputS
            X_shape, #outputS
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

export Input


### BatchNorm

@doc raw"""
    BatchNorm(;dim=1, ϵ=1e-10)

Batch Normalization `Layer` that is used ot normalize across the dimensions specified by the argument `dim`.

# Arguments

- `dim::Integer` := is the dimension to normalize across

- `ϵ::AbstractFloat` := is a backup constant that is used to prevent from division on zero when ``σ^2`` is zero

---------

# Parameters

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

    # V::Dict{Symbol,Array}
    # S::Dict{Symbol,Array}

    forwCount::Integer
    backCount::Integer
    updateCount::Integer

    nextLayers::Array{Layer,1}
    prevLayer::L where {L<:Union{Layer,Nothing}}


    function BatchNorm(;dim=1, ϵ=1e-10)
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
            # Dict(:dw=>Array{Any,1}(undef,0),
            #      :db=>Array{Any,1}(undef,0)), #V
            # Dict(:dw=>Array{Any,1}(undef,0),
            #      :db=>Array{Any,1}(undef,0)), #S
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

include("optimizers.jl")


mutable struct Model
    # layers::Array{Layer,1}
    inLayer::Array{Layer,1}
    outLayer::Layer
    lossFun::Symbol

    trainParams::Dict{Layer,Array{Symbol, 1}}

    paramsDtype::DataType

    optimizer::Optimizer

    function Model(
        inLayer::Array{Layer,1},
        outLayer::Array{Layer,1};
        kwargs...,
    )

        optimizer = getindex(kwargs, :optimizer; default=SGD())
        lossFun = getindex(kwargs, :lossFun; default=:categoricalCrossentropy)
        paramsDtype = getindex(kwargs, :paramsDtype; default=Float64)

        for oLayer in outLayer
            deepInitWB!(oLayer; dtype = paramsDtype)
        end
        resetCount!(outLayer, :backCount)
        # deepInitVS!(outLayer, optimizer)
        # resetCount!(outLayer, :forwCount)
        @assert regulization in [0, 1, 2]
        return new(
            inLayer,
            outLayer,
            lossFun,
            paramsDtype,
            optimizer,
        )
    end #inner-constructor

end #Model

function Model(inLayer::Layer,outLayer::AoL; kwargs...) where {AoL <: Union{Array{Layer,1}, Layer}}
    if outLayer isa Array
        return Model([inLayer], outLayer; kwargs...)
    else
        return Model([inLayer], [outLayer]; kwargs...)
    end
end

function Model(inLayer::Array{Layer,1}, outLayer::Layer; kwargs...)
    return Model(inLayer, [outLayer]; kwargs...)
end


export Model
