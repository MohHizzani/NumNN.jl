
export Optimizer

abstract type Optimizer end

### gds

export SGD

@doc raw"""
    SGD([α=0.01])

Stochastic Gradient Descent optimizer. Used to hold the constants of the optimization operation and the update operation of the learnable parameters.

# Arguments

- `α::AbstractFloat` := the learning rate of the optimizer

----------

# Parameters

- `α::AbstractFloat` := the learning rate of the optimizer

# Examples

```julia

opt = SGD(0.001)

model = Model(X_train, Y_train, inLayer=X_Input, outLayer=X_out; optimizer=opt)
```

"""
mutable struct SGD <: Optimizer
    "Learning rate"
    α::AbstractFloat
    SGD(α=0.01) = new(α)
end


### Momentum

export Momentum

@doc raw"""
    Momentum(; α::AbstractFloat=0.001, β1::AbstractFloat=0.9)

Momentum optimizer. Used to hold the constants of the optimization operation and the update operation of the learnable parameters.

# Arguments

- `α::AbstractFloat` := the learning rate of the optimizer

- `β1::AbstractFloat` := the averaging constant of the `Momentum` optimizer

----------

# Parameters

- `α::AbstractFloat` := the learning rate of the optimizer

- `β1::AbstractFloat` := the averaging constant of the `Momentum` optimizer

# Examples

```julia

opt = Momentum(α=0.005) #use Momentum with learning rate of 0.005

model = Model(X_train, Y_train, inLayer=X_Input, outLayer=X_out; optimizer=opt)
```

"""
mutable struct Momentum <: Optimizer
    "Learning rate"
    α::AbstractFloat

    "Coeffecient of the exponentially weighted averaging"
    β1::AbstractFloat

    Momentum(α::AbstractFloat=0.001, β1::AbstractFloat=0.9) = new(α,β)
end


### RMSprop

export RMSprop

@doc raw"""
    RMSprop(; α::AbstractFloat=0.001, β2::AbstractFloat=0.999, ϵ::AbstractFloat=1e-8)

RMS propagation optimizer. Used to hold the constants of the optimization operation and the update operation of the learnable parameters.

# Arguments

- `α::AbstractFloat` := the learning rate of the optimizer

- `β2::AbstractFloat` := the averaging constant of the `RMSprop` optimizer

- `ϵ::AbstractFloat` := To protect from division by zero

----------

# Parameters

- `α::AbstractFloat` := the learning rate of the optimizer

- `β2::AbstractFloat` := the averaging constant of the `RMSprop` optimizer

- `ϵ::AbstractFloat` := To protect from division by zero

# Examples

```julia

opt = RMSprop() #use the default values which are the optimum

model = Model(X_train, Y_train, inLayer=X_Input, outLayer=X_out; optimizer=opt)
```

"""
mutable struct RMSprop <: Optimizer
    "Learning rate"
    α::AbstractFloat

    "Coeffecient of the RMS propagation"
    β2::AbstractFloat

    "To protect from division by zero"
    ϵ::AbstractFloat

    RMSprop(; α::AbstractFloat=0.001, β2::AbstractFloat=0.999, ϵ::AbstractFloat=1e-8) = new(α, β, ϵ)
end


### Adam

export Adam


@doc raw"""
    Adam(; α::AbstractFloat=0.001, β1::AbstractFloat=0.9, β2::AbstractFloat=0.999, ϵ::AbstractFloat=1e-8)

Adam optimizer as a compination of Momentum and RMSprop optimizers. Used to hold the constants of the optimization operation and the update operation of the learnable parameters.

# Arguments

- `α::AbstractFloat` := the learning rate of the optimizer

- `β1::AbstractFloat` := the averaging constant of the `Momentum` optimizer

- `β2::AbstractFloat` := the averaging constant of the `RMSprop` optimizer

- `ϵ::AbstractFloat` := To protect from division by zero

----------

# Parameters

- `α::AbstractFloat` := the learning rate of the optimizer

- `β1::AbstractFloat` := the averaging constant of the `Momentum` optimizer

- `β2::AbstractFloat` := the averaging constant of the `RMSprop` optimizer

- `ϵ::AbstractFloat` := To protect from division by zero

# Examples

```julia

opt = Adam() #use the default values which are the optimum

model = Model(X_train, Y_train, inLayer=X_Input, outLayer=X_out; optimizer=opt)
```

"""
mutable struct Adam <: Optimizer end
    "Learning rate"
    α::AbstractFloat

    "Coeffecient of the exponentially weighted averaging"
    β1::AbstractFloat

    "Coeffecient of the RMS propagation"
    β2::AbstractFloat

    "To protect from division by zero"
    ϵ::AbstractFloat

    Adam(; α::AbstractFloat=0.001, β1::AbstractFloat=0.9, β2::AbstractFloat=0.999, ϵ::AbstractFloat=1e-8) = new(α, β1, β2, ϵ)

end
