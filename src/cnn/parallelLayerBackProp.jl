

### convlayers

@doc raw"""
    function layerBackProp(
        cLayer::ConvLayer,
        model::Model,
        actFun::SoS,
        Ao::AbstractArray,
        labels::AbstractArray,
    ) where {SoS<:Union{Type{softmax},Type{σ}}}

Derive the loss function to the input of the activation function when activation is either `softmax` or `σ`


# Return

- `dZ::AbstractArray` := the derivative of the loss function to the input of the activation function
"""
function layerBackProp(
    cLayer::ConvLayer,
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
    function layerBackProp(
        cLayer::ConvLayer,
        model::Model,
        FCache::Dict{Layer, Dict{Symbol, AbstractArray}},
        BCache::Dict{Layer, Dict{Symbol, AbstractArray}},
        Ai::AbstractArray = Array{Any,1}(undef,0),
        Ao::AbstractArray = Array{Any,1}(undef,0),
        dAo::AbstractArray = Array{Any,1}(undef,0);
        labels::AbstractArray = Array{Any,1}(undef,0),
        kwargs...
    )

Performs the layer back propagation for a `ConvLayer`

# Arguments

- `cLayer::ConvLayer`

- `model::Model`

- `FCache` := the cache of the forward propagation step

- `BCache` := the cache of so far done back propagation

- for test purpose
    Ai
    Ao
    dAo

- `labels` := when `cLayer` is an output `Layer`


# Return

- `Dict(:dA => dAi)`
"""
function layerBackProp(
    cLayer::CL,
    model::Model,
    FCache::Dict{Layer, Dict{Symbol, AbstractArray}},
    BCache::Dict{Layer, Dict{Symbol, AbstractArray}},
    Ai::AbstractArray = Array{Any,1}(undef,0),
    Ao::AbstractArray = Array{Any,1}(undef,0),
    dAo::AbstractArray = Array{Any,1}(undef,0);
    labels::AbstractArray = Array{Any,1}(undef,0),
    kwargs...
) where {CL <: ConvLayer}

    kwargs = Dict{Symbol, Any}(kwargs...)
    NNlib = getindex(kwargs, :NNlib; default=true)
    img2col = getindex(kwargs, :img2col; default=true)

    if length(Ai) == 0
        Ai = FCache[cLayer.prevLayer][:A]
    end
    if length(Ao) == 0
        Ao = FCache[cLayer][:A]
    end
    m = size(Ao)[end]

    regulization, λ = model.regulization, model.λ

    actFun = cLayer.actFun

    dZ = []
    if cLayer.actFun == model.outLayer.actFun
        dZ = layerBackProp(cLayer, model, eval(:($actFun)), Ao, labels)
    elseif length(dAo) != 0
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
            catch e
                dAo = BCache[nextLayer][:dA] #need to initialize dA
            end #try/catch
        end #for


        dActFun = Symbol("d", cLayer.actFun)

        Z = FCache[cLayer][:Z]

        dZ = dAo .* eval(:($dActFun($Z)))

    else #in case not every next layer done backprop
        return Dict(:dA => Array{Any,1}(undef,0))

    end #if all(i->(i.backCount==cLayer.nextLayers[1].backCount), cLayer.nextLayers)

    dAi = zeros(promote_type(eltype(Ai),eltype(dZ)), size(Ai))

    if NNlib
        dNNConv!(cLayer, Ai, dAi, dZ)
    elseif img2col
        dimg2colConvolve!(cLayer, Ai, dAi, dZ)
    else
        dconvolve!(cLayer, Ai, dAi, dZ)
    end #if NNlib

    cLayer.backCount += 1

    return Dict(:dA => dAi)
end #function layerBackProp(cLayer::Input


### Pooling Layers

#import only the needed parts not to have conflict
import NNlib.∇maxpool, NNlib.∇meanpool, NNlib.∇maxpool!, NNlib.∇meanpool!, NNlib.PoolDims

@doc raw"""
layerBackProp(
    cLayer::PoolLayer,
    model::Model,
    FCache::Dict{Layer, Dict{Symbol, AbstractArray}},
    BCache::Dict{Layer, Dict{Symbol, AbstractArray}},
    Ai::AbstractArray = Array{Any,1}(undef,0),
    Ao::AbstractArray = Array{Any,1}(undef,0),
    dAo::AbstractArray = Array{Any,1}(undef,0);
    labels::AbstractArray = Array{Any,1}(undef,0),
    kwargs...
)

Performs the layer back propagation for a `PoolLayer`

# Arguments

- `cLayer::ConvLayer`

- `model::Model`

- `FCache` := the cache of the forward propagation step

- `BCache` := the cache of so far done back propagation

- for test purpose
    Ai
    Ao
    dAo

- `labels` := when `cLayer` is an output `Layer`


# Return

- `Dict(:dA => dAi)`
"""
function layerBackProp(
    cLayer::PL,
    model::Model,
    FCache::Dict{Layer, Dict{Symbol, AbstractArray}},
    BCache::Dict{Layer, Dict{Symbol, AbstractArray}},
    Ai::AbstractArray = Array{Any,1}(undef,0),
    Ao::AbstractArray = Array{Any,1}(undef,0),
    dAo::AbstractArray = Array{Any,1}(undef,0);
    labels::AbstractArray = Array{Any,1}(undef,0),
    kwargs...
) where {PL <: PoolLayer}

    kwargs = Dict{Symbol, Any}(kwargs...)

    fastPool = getindex(kwargs, :fastPool; default=true)
    NNlib = getindex(kwargs, :NNlib; default=true)
    if length(Ai) == 0
        Ai = FCache[cLayer.prevLayer][:A]
    end
    if length(Ao) == 0
        Ao = FCache[cLayer][:A]
    end

    padS = paddingSize(cLayer, Ai)



    if length(dAo) == 0
        if all(
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
        else
            return Dict(:dA => Array{Any,1}(undef,0))
        end #if all(i->(i.backCount==cLayer.nextLayers[1].backCount), cLayer.nextLayers)
    end #if dA==nothing

    dAi = zeros(promote_type(eltype(Ai), eltype(dAo)), size(Ai))


    if NNlib
        pooldims = PoolDims(Ai, cLayer.f, stride = cLayer.s, padding = padS)
        if cLayer isa MaxPoolLayer
            ∇maxpool!(dAi, dAo, Ao, Ai, pooldims)
        elseif cLayer isa AveragePoolLayer
            ∇meanpool!(dAi, dAo, Ao, Ai, pooldims)
        end #if cLayer isa MaxPoolLayer

    elseif cLayer.s == cLayer.f && fastPool
        dfastPooling!(cLayer, Ai, dAi, Ao, dAo)
    else
        dpooling!(cLayer, Ai, dAi, Ao, dAo)

    end #if NNlib

    cLayer.backCount += 1
    return Dict(:dA => dAi)

end #unction layerBackProp(cLayer::PL,
