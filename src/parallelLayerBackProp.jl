
@doc raw"""
    layerBackProp(
        cLayer::FCLayer,
        model::Model,
        actFun::SoS,
        Ao::AbstractArray,
        labels::AbstractArray,
    ) where {SoS<:Union{Type{softmax},Type{σ}}}

For output `FCLayer` layers with softmax and sigmoid activation functions

"""
function layerBackProp(
    cLayer::FCLayer,
    model::Model,
    actFun::SoS,
    Ao::AbstractArray,
    labels::AbstractArray,
) where {SoS<:Union{Type{softmax},Type{σ}}}


    lossFun = model.lossFun
    dlossFun = Symbol("d", lossFun)
    Y = labels
    dZ = eval(:($dlossFun($Ao, $Y)))

    # dAi = cLayer.W'dZ
    return dZ #, dAi
end #softmax or σ layerBackProp


@doc raw"""
    layerBackProp(
        cLayer::Activation,
        model::Model,
        actFun::SoS,
        Ao::AbstractArray,
        labels::AbstractArray,
    ) where {SoS<:Union{Type{softmax},Type{σ}}}

For output `Activation` layers with softmax and sigmoid activation functions
"""
function layerBackProp(
    cLayer::Activation,
    model::Model,
    actFun::SoS,
    Ao::AbstractArray,
    labels::AbstractArray,
) where {SoS<:Union{Type{softmax},Type{σ}}}


    lossFun = model.lossFun
    dlossFun = Symbol("d", lossFun)
    Y = labels
    dZ = eval(:($dlossFun($Ao, $Y)))

    return dZ
end #softmax or σ layerBackProp

@doc raw"""
    layerBackProp(
        cLayer::FCLayer,
        model::Model,
        FCache::Dict{Layer, Dict{Symbol, AbstractArray}},
        BCache::Dict{Layer, Dict{Symbol, AbstractArray}},
        dAo::AbstractArray{T1,2} = Array{Any,2}(undef,0,0);
        labels::AbstractArray{T2,2} = Array{Any,2}(undef,0,0),
        kwargs...,
    ) where {T1,T2}

Perform the back propagation of `FCLayer` type on the activations and trainable parameters `W` and `B`

# Argument

- `cLayer` := the layer to perform the backprop on

- `model` := the `Model`

- `FCache` := the cache values of the forprop

- `BCache` := the cache values of the backprop from the front `Layer`(s)

- `dAo` := (for test purpose) the derivative of the front `Layer`

- `labels` := in case this is the output layer

# Return

- A `Dict{Symbol, AbstractArray}(:dA => dAi)`
"""
function layerBackProp(
    cLayer::FCLayer,
    model::Model,
    FCache::Dict{Layer, Dict{Symbol, AbstractArray}},
    BCache::Dict{Layer, Dict{Symbol, AbstractArray}},
    dAo::AbstractArray{T1,2} = Array{Any,2}(undef,0,0);
    labels::AbstractArray{T2,2} = Array{Any,2}(undef,0,0),
    kwargs...,
) where {T1,T2}

    prevLayer = cLayer.prevLayer
    lossFun = model.lossFun

    Ao = FCache[cLayer][:A]

    regulization, λ = model.regulization, model.λ

    actFun = cLayer.actFun

    dZ = []
    if cLayer.actFun == model.outLayer.actFun

        dZ = layerBackProp(cLayer, model, eval(:($actFun)), Ao, labels)

    elseif length(dAo) != 0
        keepProb = cLayer.keepProb
        if keepProb < 1.0 #to save time of multiplication in case keepProb was one
            D = rand(cLayer.channels, 1) .< keepProb
            dAo .*= D
            dAo ./= keepProb
        end

        dActFun = Symbol("d", cLayer.actFun)

        Z = FCache[cLayer][:Z]

        dZ = dAo .* eval(:($dActFun($Z)))

    elseif all(
        i -> (i.backCount == cLayer.nextLayers[1].backCount),
        cLayer.nextLayers,
    )
        dAo = []
        for nextLayer in cLayer.nextLayers
            try
                dAo .+= BCache[nextLayer][:dA]
            catch e #in case first time DimensionMismatch
                dAo = BCache[nextLayer][:dA]
            end
        end #for

        keepProb = cLayer.keepProb
        if keepProb < 1.0 #to save time of multiplication in case keepProb was one
            D = rand(cLayer.channels, 1) .< keepProb
            dAo .*= D
            dAo ./= keepProb
        end

        dActFun = Symbol("d", cLayer.actFun)

        Z = FCache[cLayer][:Z]

        dZ = dAo .* eval(:($dActFun($Z)))

    else #in case not every next layer done backprop
        return Dict(:dA => Array{Any,1}(undef,0))

    end #if all(i->(i.backCount==cLayer.nextLayers[1].backCount), cLayer.nextLayers)

    m = size(dZ)[end]

    cLayer.dW = dZ * FCache[cLayer.prevLayer][:A]' ./ m

    if regulization > 0
        if regulization == 1
            cLayer.dW .+= (λ / 2m)
        else
            cLayer.dW .+= (λ / m) .* cLayer.W
        end
    end

    cLayer.dB = 1 / m .* sum(dZ, dims = 2)

    dAi = cLayer.W'dZ

    cLayer.backCount += 1

    return Dict(:dA => dAi)

end #function layerBackProp(cLayer::FCLayer)


@doc raw"""
    layerBackProp(
        cLayer::Activation,
        model::Model,
        FCache::Dict{Layer, Dict{Symbol, AbstractArray}},
        BCache::Dict{Layer, Dict{Symbol, AbstractArray}},
        dAo::AbstractArray{T1,N} = Array{Any,1}(undef,0);
        labels::AbstractArray{T2,N} = Array{Any,1}(undef,0),
        kwargs...,
    ) where {T1,T2,N}

Perform the back propagation of `Activation` type

# Argument

- `cLayer` := the layer to perform the backprop on

- `model` := the `Model`

- `FCache` := the cache values of the forprop

- `BCache` := the cache values of the backprop from the front `Layer`(s)

- `dAo` := (for test purpose) the derivative of the front `Layer`

- `labels` := in case this is the output layer

# Return

- A `Dict{Symbol, AbstractArray}(:dA => dAi)`
"""
function layerBackProp(
    cLayer::Activation,
    model::Model,
    FCache::Dict{Layer, Dict{Symbol, AbstractArray}},
    BCache::Dict{Layer, Dict{Symbol, AbstractArray}},
    dAo::AbstractArray{T1,N} = Array{Any,1}(undef,0);
    labels::AbstractArray{T2,N} = Array{Any,1}(undef,0),
    kwargs...,
) where {T1,T2,N}

    prevLayer = cLayer.prevLayer
    lossFun = model.lossFun

    Ao = FCache[cLayer][:A]

    regulization, λ = model.regulization, model.λ

    actFun = cLayer.actFun

    dZ = []
    if cLayer.actFun == model.outLayer.actFun

        dZ = layerBackProp(cLayer, model, eval(:($actFun)), Ao, labels)

    elseif length(dAo) != 0
        keepProb = cLayer.keepProb
        if keepProb < 1.0 #to save time of multiplication in case keepProb was one
            D = rand(cLayer.channels, 1) .< keepProb
            dAo .*= D
            dAo ./= keepProb
        end

        dActFun = Symbol("d", cLayer.actFun)

        Z = FCache[cLayer][:Z]

        dZ = dAo .* eval(:($dActFun($Z)))

    elseif all(
        i -> (i.backCount == cLayer.nextLayers[1].backCount),
        cLayer.nextLayers,
    )
        dAo = []
        for nextLayer in cLayer.nextLayers
            try
                dAo .+= BCache[nextLayer][:dA]
            catch e #in case first time DimensionMismatch
                dAo = BCache[nextLayer][:dA]
            end
        end #for

        dActFun = Symbol("d", cLayer.actFun)

        ## get the input as the output of the previous layer
        Z = FCache[cLayer.prevLayer][:A]

        dZ = dAo .* eval(:($dActFun($Z)))

    else #in case not every next layer done backprop
        return Dict(:dA => Array{Any,1}(undef,0))

    end #if all(i->(i.backCount==cLayer.nextLayers[1].backCount), cLayer.nextLayers)

    # cLayer.dW = dZ * FCache[cLayer.prevLayer][:A]' ./ m
    #
    # if regulization > 0
    #     if regulization == 1
    #         cLayer.dW .+= (λ / 2m)
    #     else
    #         cLayer.dW .+= (λ / m) .* cLayer.W
    #     end
    # end
    #
    # cLayer.dB = 1 / m .* sum(dZ, dims = 2)

    dAi = dZ

    cLayer.backCount += 1

    return Dict(:dA => dAi)

end #function layerBackProp(cLayer::Activation


@doc raw"""
    layerBackProp(
        cLayer::AddLayer,
        model::Model,
        FCache::Dict{Layer, Dict{Symbol, AbstractArray}},
        BCache::Dict{Layer, Dict{Symbol, AbstractArray}},
        dAo::AbstractArray{T1,N} = Array{Any,1}(undef,0);
        labels::AbstractArray{T2,N} = Array{Any,1}(undef,0),
        kwargs...,
    ) where {T1,T2,N}

Perform the back propagation of `AddLayer` type

# Argument

- `cLayer` := the layer to perform the backprop on

- `model` := the `Model`

- `FCache` := the cache values of the forprop

- `BCache` := the cache values of the backprop from the front `Layer`(s)

- `dAo` := (for test purpose) the derivative of the front `Layer`

- `labels` := in case this is the output layer

# Return

- A `Dict{Symbol, AbstractArray}(:dA => dAi)`
"""
function layerBackProp(
    cLayer::AddLayer,
    model::Model,
    FCache::Dict{Layer, Dict{Symbol, AbstractArray}},
    BCache::Dict{Layer, Dict{Symbol, AbstractArray}},
    dAo::AbstractArray{T1,N} = Array{Any,1}(undef,0);
    labels::AbstractArray{T2,N} = Array{Any,1}(undef,0),
    kwargs...,
) where {T1,T2,N}

    if length(dAo) != 0
        return Dict(:dA => dAo)

    elseif all(
        i -> (i.backCount == cLayer.nextLayers[1].backCount),
        cLayer.nextLayers,
    )
        dAo = []
        for nextLayer in cLayer.nextLayers
            try
                dAo .+= BCache[nextLayer][:dA]
            catch e
                dAo = BCache[nextLayer][:dA] #to initialize dA
            end #try/catch
        end #for


    else #in case not every next layer done backprop
        return Dict(:dA => Array{Any,1}(undef,0))

    end #if all(i->(i.backCount==cLayer.nextLayers[1].backCount), cLayer.nextLayers)

    cLayer.backCount += 1

    return Dict(:dA => dAo)
end #function layerBackProp(cLayer::AddLayer


@doc raw"""
    layerBackProp(
        cLayer::Input,
        model::Model,
        FCache::Dict{Layer, Dict{Symbol, AbstractArray}},
        BCache::Dict{Layer, Dict{Symbol, AbstractArray}},
        dAo::AbstractArray{T1,N} = Array{Any,1}(undef,0);
        labels::AbstractArray{T2,N} = Array{Any,1}(undef,0),
        kwargs...,
    ) where {T1,T2,N}

Perform the back propagation of `Input` type

# Argument

- `cLayer` := the layer to perform the backprop on

- `model` := the `Model`

- `FCache` := the cache values of the forprop

- `BCache` := the cache values of the backprop from the front `Layer`(s)

- `dAo` := (for test purpose) the derivative of the front `Layer`

- `labels` := in case this is the output layer

# Return

- A `Dict{Symbol, AbstractArray}(:dA => dAi)`
"""
function layerBackProp(
    cLayer::Input,
    model::Model,
    FCache::Dict{Layer, Dict{Symbol, AbstractArray}},
    BCache::Dict{Layer, Dict{Symbol, AbstractArray}},
    dAo::AbstractArray{T1,N} = Array{Any,1}(undef,0);
    labels::AbstractArray{T2,N} = Array{Any,1}(undef,0),
    kwargs...,
) where {T1,T2,N}

    if length(dAo) != 0
        return Dict(:dA => dAo)
    elseif all(
        i -> (i.backCount == cLayer.nextLayers[1].backCount),
        cLayer.nextLayers,
    )
        dAo = []
        for nextLayer in cLayer.nextLayers
            try
                dAo .+= BCache[nextLayer][:dA]
            catch e
                dAo = BCache[nextLayer][:dA] #need to initialize dA
            end #try/catch
        end #for

    else #in case not every next layer done backprop
        return Dict(:dA => Array{Any,1}(undef,0))

    end #if all(i->(i.backCount==cLayer.nextLayers[1].backCount), cLayer.nextLayers)

    cLayer.backCount += 1

    return Dict(:dA => dAo)
end #function layerBackProp(cLayer::Input

@doc raw"""
    layerBackProp(
        cLayer::BatchNorm,
        model::Model,
        FCache::Dict{Layer, Dict{Symbol, AbstractArray}},
        BCache::Dict{Layer, Dict{Symbol, AbstractArray}},
        dAo::AbstractArray = Array{Any,1}(undef,0);
        labels::AbstractArray = Array{Any,1}(undef,0),
        kwargs...,
    )

Perform the back propagation of `BatchNorm` type on the activations and trainable parameters `W` and `B`

# Argument

- `cLayer` := the layer to perform the backprop on

- `model` := the `Model`

- `FCache` := the cache values of the forprop

- `BCache` := the cache values of the backprop from the front `Layer`(s)

- `dAo` := (for test purpose) the derivative of the front `Layer`

- `labels` := in case this is the output layer

# Return

- A `Dict{Symbol, AbstractArray}(:dA => dAi)`
"""
function layerBackProp(
    cLayer::BatchNorm,
    model::Model,
    FCache::Dict{Layer, Dict{Symbol, AbstractArray}},
    BCache::Dict{Layer, Dict{Symbol, AbstractArray}},
    dAo::AbstractArray = Array{Any,1}(undef,0);
    labels::AbstractArray = Array{Any,1}(undef,0),
    kwargs...,
)


    prevLayer = cLayer.prevLayer
    lossFun = model.lossFun

    Ao = FCache[cLayer][:A]
    Ai = FCache[cLayer.prevLayer][:A]

    m = size(Ao)[end]

    normDim = cLayer.dim+1

    regulization, λ = model.regulization, model.λ

    N = length(cLayer.outputS)

    dZ = []
    if length(dAo) != 0
        dAo = permutedims(dAo, [N, (1:N-1)...])
        dZ = cLayer.W .* dAo
    elseif all(
        i -> (i.backCount == cLayer.nextLayers[1].backCount),
        cLayer.nextLayers,
    )
        dAo = []
        for nextLayer in cLayer.nextLayers
            try
                dAo .+= BCache[nextLayer][:dA]
            catch e
                dAo = BCache[nextLayer][:dA] #need to initialize dA
            end #try/catch
        end #for
        dAo = permutedims(dAo, [N, (1:N-1)...])
        dZ = cLayer.W .* dAo

    else #in case not every next layer done backprop
        return Dict(:dA => Array{Any,1}(undef,0))

    end #if all(i->(i.backCount==cLayer.nextLayers[1].backCount), cLayer.nextLayers)



    #Z is already flipped
    cLayer.dW = sum(dAo .* FCache[cLayer][:Z], dims = 1:normDim)

    if regulization > 0
        if regulization == 1
            cLayer.dW .+= (λ / 2m)
        else
            cLayer.dW .+= (λ / m) .* cLayer.W
        end
    end

    cLayer.dB = sum(dAo, dims = 1:normDim)

    Num = prod(size(dAo)[1:normDim])

    varep = FCache[cLayer][:var] .+ cLayer.ϵ
    Ai_μ = FCache[cLayer][:Ai_μ]
    Ai_μ_s = FCache[cLayer][:Ai_μ_s]

    svarep = sqrt.(varep)

    dZ1 = dZ ./ svarep

    dZ2 = sum(dZ .* Ai_μ; dims = 1:normDim)
    dZ2 .*= (-1 ./ (svarep .^ 2))
    dZ2 .*= (0.5 ./ svarep)
    dZ2 = dZ2 .* (1.0 / Num) .* ones(promote_type(eltype(Ai_μ), eltype(dZ2)), size(Ai_μ))
    dZ2 .*= (2 .* Ai_μ)

    dẐ = dZ1 .+ dZ2

    dZ3 = dẐ

    dZ4 = .- sum(dẐ; dims = 1:normDim)

    dZ4 = dZ4 .* (1.0 / Num) .* ones(promote_type(eltype(Ai), eltype(dZ4)), size(dZ3))

    dAip = dZ3 .+ dZ4

    dAi = permutedims(dAip, [(2:N)..., 1])

    # dẐ =
    #     dZ ./ sqrt.(varep) .*
    #     (.-(Ai_μ_s) .* ((N - 1) / N^2) ./ (varep) .^ 2 .+ 1) .*
    #     (1 - 1 / N)
    #
    # dAi = dẐ



    cLayer.backCount += 1

    return Dict(:dA => dAi)

end #function layerBackProp(cLayer::BatchNorm)


export layerBackProp
