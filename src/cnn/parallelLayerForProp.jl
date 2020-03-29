include("parallelNNConv.jl")
include("parallelConvolve.jl")
include("parallelImg2colConvolve.jl")

###convolution layers forprop

function layerForProp(
    cLayer::CL,
    Ai::AbstractArray = Array{Any,1}(undef,0);
    FCache::Dict{Layer,Dict{Symbol, AbstractArray}},
    kwargs...
) where {CL <: ConvLayer}

    kwargs = Dict(kwargs...)
    NNlib = getindex(kwargs, :NNlib; default=true)
    img2col = getindex(kwargs, :img2col; default=false)
    if length(Ai) == 0
        Ai = FCache[cLayer.prevLayer][:A]
    end
    cLayer.inputS = size(Ai)
    paddedS = paddedSize(cLayer, Ai)[1:end-2]
    s = cLayer.s
    f = cLayer.f
    outputS = outDims(cLayer, Ai)[1:end-2]
    ci, m = cLayer.inputS[end-1:end]
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
        Z = zeros(eltype(Ai), outputS..., co, m)
        D = convolve(cLayer, Ai, Z)
    end

    cLayer.outputS = size(D[:A])

    # Ai = nothing
    cLayer.forwCount += 1
    # Base.GC.gc()
    return D

end #function layerForProp(cLayer::Conv1D)


### Pooling Layers

include("parallelFastPooling.jl")
include("parallelPooling.jl")

#import only the needed parts not to have conflict
import NNlib.maxpool, NNlib.meanpool, NNlib.maxpool!, NNlib.meanpool!, NNlib.PoolDims

function layerForProp(
    cLayer::PL,
    Ai::AbstractArray = Array{Any,1}(undef,0);
    FCache::Dict{Layer,Dict{Symbol, AbstractArray}},
    kwargs...
) where {PL <: PoolLayer}

    kwargs = Dict(kwargs...)
    fastPool = getindex(kwargs, :fastPool; default=true)
    NNlib = getindex(kwargs, :NNlib; default=true)
    if length(Ai) == 0
        Ai = FCache[cLayer.prevLayer][:A]
    end

    cLayer.inputS = size(Ai)
    paddedS = paddedSize(cLayer, Ai)[1:end-2]
    s = cLayer.s
    f = cLayer.f
    outputS = outDims(cLayer, Ai)[1:end-2]
    ci, m = cLayer.inputS[end-1:end]
    co = cLayer.channels
    cLayer.inputS = size(Ai)
    padS = paddingSize(cLayer, Ai)

    Ao = zeros(eltype(Ai), outputS..., co, m)

    if s == f && fastPool #to use the built-in reshape and maximum and mean
        fastPooling!(cLayer, Ai, Ao)
    elseif NNlib
        pooldims = PoolDims(Ai, f, stride = s, padding = padS)
        if cLayer isa MaxPoolLayer
            maxpool!(Ao, Ai, pooldims)
        elseif cLayer isa AveragePoolLayer
            meanpool!(Ao, Ai, pooldims)
        end #if cLayer isa MaxPoolLayer
    else
        pooling!(cLayer, Ai, Ao)
    end #if NNlibConv

    cLayer.outputS = size(Ao)
    # Ai = nothing
    cLayer.forwCount += 1
    # Base.GC.gc()
    return Dict(:A => Ao)
end #unction layerForProp(cLayer::OneD) where {OneD <: Union{MaxPool1D, AveragePool1D}}
