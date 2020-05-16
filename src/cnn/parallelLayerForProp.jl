include("parallelNNConv.jl")
include("parallelConvolve.jl")
include("parallelImg2colConvolve.jl")

###convolution layers forprop

@doc raw"""
    function layerForProp(
        cLayer::ConvLayer,
        Ai::AbstractArray = Array{Any,1}(undef,0);
        FCache::Dict{Layer,Dict{Symbol, AbstractArray}},
        kwargs...
    )

Perform the layer forward propagation for a `ConvLayer`

# Arguments

- `cLayer::ConvLayer`

- `Ai` := optional activation of the previous layer

- `FCache` := a `Dict` holds the outputs of `layerForProp` of the previous `Layer`(s)

# Returns

- `Dict(:Z => Z, :A => Ao)`
"""
function layerForProp(
    cLayer::CL,
    Ai::AbstractArray = Array{Any,1}(undef,0);
    FCache::Dict{Layer,Dict{Symbol, AbstractArray}},
    Done::Dict{Layer,Bool},
    kwargs...
) where {CL <: ConvLayer}

    kwargs = Dict{Symbol, Any}(kwargs...)
    NNlib = getindex(kwargs, :NNlib; default=true)
    img2col = getindex(kwargs, :img2col; default=false)
    if length(Ai) == 0
        Ai = FCache[cLayer.prevLayer][:A]
    end
    if cLayer.inputS != size(Ai)[1:end-1]
        cLayer.inputS = size(Ai)[1:end-1]
    end
    paddedS = paddedSize(cLayer, Ai)[1:end-2]
    s = cLayer.s
    f = cLayer.f
    outputS = outDims(cLayer, Ai)[1:end-1]
    ci, m = size(Ai)[end-1:end]
    co = cLayer.channels

    if NNlib
        D = NNConv(cLayer, Ai)
    elseif img2col
        ## in case input different size than previous time
        if (prod((outputS..., co)), prod((paddedS..., ci))) != size(cLayer.K)
            cLayer.K = unroll(cLayer, (paddedS..., ci, m))
        end
        D = img2colConvolve(cLayer, Ai)
    else
        D = convolve(cLayer, Ai)
    end

    if cLayer.outputS != size(D[:A])[1:end-1]
        cLayer.outputS = size(D[:A])[1:end-1]
    end
    # Ai = nothing
    # cLayer.forwCount += 1
    Done[cLayer] = true
    # Base.GC.gc()
    return D

end #function layerForProp(cLayer::Conv1D)


### Pooling Layers

include("parallelFastPooling.jl")
include("parallelPooling.jl")

#import only the needed parts not to have conflict
import NNlib.maxpool, NNlib.meanpool, NNlib.maxpool!, NNlib.meanpool!, NNlib.PoolDims

@doc raw"""
    layerForProp(
        cLayer::PoolLayer},
        Ai::AbstractArray = Array{Any,1}(undef,0);
        FCache::Dict{Layer,Dict{Symbol, AbstractArray}},
        kwargs...
    )


Perform the layer forward propagation for a `PoolLayer`

# Arguments

- `cLayer::PoolLayer`

- `Ai` := optional activation of the previous layer

- `FCache` := a `Dict` holds the outputs of `layerForProp` of the previous `Layer`(s)

# Returns

- `Dict(:A => Ao)`
"""
function layerForProp(
    cLayer::PL,
    Ai::AbstractArray = Array{Any,1}(undef,0);
    FCache::Dict{Layer,Dict{Symbol, AbstractArray}},
    Done::Dict{Layer,Bool},
    kwargs...
) where {PL <: PoolLayer}

    kwargs = Dict{Symbol, Any}(kwargs...)
    fastPool = getindex(kwargs, :fastPool; default=true)
    NNlib = getindex(kwargs, :NNlib; default=true)
    if length(Ai) == 0
        Ai = FCache[cLayer.prevLayer][:A]
    end

    if cLayer.inputS != size(Ai)[1:end-1]
        cLayer.inputS = size(Ai)[1:end-1]
    end
    paddedS = paddedSize(cLayer, Ai)[1:end-2]
    s = cLayer.s
    f = cLayer.f
    outputS = outDims(cLayer, Ai)[1:end-1]
    ci, m = size(Ai)[end-1:end]
    co = cLayer.channels
    # cLayer.inputS = size(Ai)
    padS = paddingSize(cLayer, Ai)

    Ao = zeros(eltype(Ai), outputS..., co, m)

    if NNlib
        pooldims = PoolDims(Ai, f, stride = s, padding = padS)
        if cLayer isa MaxPoolLayer
            maxpool!(Ao, Ai, pooldims)
        elseif cLayer isa AveragePoolLayer
            meanpool!(Ao, Ai, pooldims)
        end #if cLayer isa MaxPoolLayer
    elseif s == f && fastPool #to use the built-in reshape and maximum and mean
        fastPooling!(cLayer, Ai, Ao)
    else
        pooling!(cLayer, Ai, Ao)
    end #if NNlibConv

    if cLayer.outputS != size(Ao)[1:end-1]
        cLayer.outputS = size(Ao)[1:end-1]
    end
    # Ai = nothing
    # cLayer.forwCount += 1
    Done[cLayer] = true
    # Base.GC.gc()
    return Dict(:A => Ao)
end #unction layerForProp(cLayer::OneD) where {OneD <: Union{MaxPool1D, AveragePool1D}}
