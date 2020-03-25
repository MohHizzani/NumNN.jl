include("parallelNNConv.jl")
include("parallelConvolve.jl")
include("parallelImg2colConvolve.jl")

###convolution layers forprop

function layerForProp(
    cLayer::Conv1D,
    Ai::AbstractArray = Array{Any,1}(undef,0);
    FCache::Dict{Layer,Dict{Symbol, AbstractArray}},
    kwargs...
)

    NNlib = haskey(kwargs, :NNlib) ? kwargs[:NNlib] : true
    img2col = haskey(kwargs, :img2col) ? kwargs[:img2col] : false
    if length(Ai) == 0
        Ai = FCache[cLayer.prevLayer][:A]
    end
    cLayer.inputS = size(Ai)
    # Ai = padding(cLayer, Ai)
    n_Hi, ci, m = paddedSize(cLayer, Ai)
    s_H = cLayer.s
    f_H = cLayer.f
    c = cLayer.channels
    n_H = (n_Hi - f_H) ÷ s_H + 1

    if NNlib
        D = NNConv(cLayer, Ai)
    elseif img2col
        ## in case input different size than previous time
        if (n_H * c, n_Hi * ci) != size(cLayer.K)
            cLayer.K = unroll(cLayer, (n_Hi, ci, m))
        end
        D = img2colConvolve(cLayer, Ai)
    else
        Z = zeros(eltype(Ai), n_H, cLayer.channels, m)
        D = convolve(cLayer, Ai, Z)
    end

    cLayer.outputS = size(D[:A])

    Ai = nothing
    cLayer.forwCount += 1
    # Base.GC.gc()
    return D

end #function layerForProp(cLayer::Conv1D)

function layerForProp(
    cLayer::Conv2D,
    Ai::AbstractArray = Array{Any,1}(undef,0);
    FCache::Dict{Layer,Dict{Symbol, AbstractArray}},
    kwargs...
)
    NNlib = haskey(kwargs, :NNlib) ? kwargs[:NNlib] : true
    img2col = haskey(kwargs, :img2col) ? kwargs[:img2col] : false
    if length(Ai) == 0
        Ai = FCache[cLayer.prevLayer][:A]
    end
    cLayer.inputS = size(Ai)
    # Ai = padding(cLayer, Ai)
    n_Hi, n_Wi, ci, m = paddedSize(cLayer, Ai)
    c = cLayer.channels
    s_H, s_W = cLayer.s
    f_H, f_W = cLayer.f
    c = cLayer.channels
    n_H = (n_Hi - f_H) ÷ s_H + 1
    n_W = (n_Wi - f_W) ÷ s_W + 1



    if NNlib
        D = NNConv(cLayer, Ai)
    elseif img2col
        ## in case input different size than previous time
        if (n_W * n_H * c, n_Hi * n_Wi * ci) != size(cLayer.K)
            cLayer.K = unroll(cLayer, (n_Hi, n_Wi, ci, m))
        end #if (n_W*n_H* c, n_Hi*n_Wi*ci) != size(cLayer.K)
        D = img2colConvolve(cLayer, Ai)
    else
        Z = zeros(eltype(Ai), n_H, n_W, cLayer.channels, m)
        D = convolve(cLayer, Ai, Z)
    end #if img2colConvolve

    cLayer.outputS = size(D[:A])

    Ai = nothing
    cLayer.forwCount += 1

    # Base.GC.gc()
    return D
end #function layerForProp(cLayer::Conv2D)

function layerForProp(
    cLayer::Conv3D,
    Ai::AbstractArray = Array{Any,1}(undef,0);
    FCache::Dict{Layer,Dict{Symbol, AbstractArray}},
    kwargs...
)
    NNlib = haskey(kwargs, :NNlib) ? kwargs[:NNlib] : true
    img2col = haskey(kwargs, :img2col) ? kwargs[:img2col] : false
    if length(Ai) == 0
        Ai = FCache[cLayer.prevLayer][:A]
    end
    cLayer.inputS = size(Ai)
    # Ai = padding(cLayer, Ai)
    n_Hi, n_Wi, n_Di, ci, m = paddedSize(cLayer, Ai)
    s_H, s_W, s_D = cLayer.s
    f_H, f_W, f_D = cLayer.f
    c = cLayer.channels

    n_H = (n_Hi - f_H) ÷ s_H + 1
    n_W = (n_Wi - f_W) ÷ s_W + 1
    n_D = (n_Di - f_D) ÷ s_D + 1


    if NNlib
        D = NNConv(cLayer, Ai)
    elseif img2col
        ## in case input different size than previous time
        if (n_W * n_H * n_D * c, n_Hi * n_Wi * n_Di * ci) != size(cLayer.K)
            cLayer.K = unroll(cLayer, (n_Hi, n_Wi, n_Di, ci, m))
        end #if (n_W*n_H* c, n_Hi*n_Wi*ci) != size(cLayer.K)
        D = img2colConvolve(cLayer, Ai)
    else
        Z = zeros(eltype(Ai), n_H, n_W, n_D, cLayer.channels, m)
        D = convolve(cLayer, Ai, Z)
    end #if img2colConvolve

    cLayer.outputS = size(D[:A])

    cLayer.forwCount += 1

    Ai = nothing

    # Base.GC.gc()
    return D
end #function layerForProp(cLayer::Conv3D)



### Pooling Layers

include("parallelFastPooling.jl")
include("parallelPooling.jl")

#import only the needed parts not to have conflict
import NNlib.maxpool, NNlib.meanpool, NNlib.maxpool!, NNlib.meanpool!, NNlib.PoolDims

function layerForProp(
    cLayer::OneD,
    Ai::AbstractArray = Array{Any,1}(undef,0);
    FCache::Dict{Layer,Dict{Symbol, AbstractArray}},
    kwargs...
) where {OneD<:Union{MaxPool1D,AveragePool1D}}

    NNlib = haskey(kwargs, :NNlib) ? kwargs[:NNlib] : true
    if length(Ai) == 0
        Ai = FCache[cLayer.prevLayer][:A]
    end
    cLayer.inputS = size(Ai)
    # Ai = padding(cLayer, Ai)
    padS = paddingSize(cLayer, Ai)
    n_Hi, ci, m = paddedSize(cLayer, Ai)
    s_H = cLayer.s
    f_H = cLayer.f
    c = cLayer.channels

    n_H = (n_Hi - f_H) ÷ s_H + 1

    Ao = zeros(eltype(Ai), n_H, c, m)

    if NNlib
        pooldims = PoolDims(Ai, cLayer.f, stride = cLayer.s, padding = padS)
        if cLayer isa MaxPoolLayer
            maxpool!(Ao, Ai, pooldims)
        elseif cLayer isa AveragePoolLayer
            meanpool!(Ao, Ai, pooldims)
        end #if cLayer isa MaxPoolLayer
    elseif f_H == s_H #to use the built-in reshape and maximum and mean
        fastPooling!(cLayer, Ai, Ao)
    else
        pooling!(cLayer, Ai, Ao)
    end #if NNlibConv

    cLayer.outputS = size(Ao)
    Ai = nothing
    cLayer.forwCount += 1
    # Base.GC.gc()
    return Dict(:A => Ao)
end #unction layerForProp(cLayer::OneD) where {OneD <: Union{MaxPool1D, AveragePool1D}}

function layerForProp(
    cLayer::TwoD,
    Ai::AbstractArray = Array{Any,1}(undef,0);
    FCache::Dict{Layer,Dict{Symbol, AbstractArray}},
    kwargs...,
) where {TwoD<:Union{MaxPool2D,AveragePool2D}}

    NNlib = haskey(kwargs, :NNlib) ? kwargs[:NNlib] : true
    if length(Ai) == 0
        Ai = FCache[cLayer.prevLayer][:A]
    end
    cLayer.inputS = size(Ai)
    # Ai = padding(cLayer, Ai)
    padS = paddingSize(cLayer, Ai)
    n_Hi, n_Wi, ci, m = paddedSize(cLayer, Ai)
    s_H, s_W = S = cLayer.s
    f_H, f_W = F = cLayer.f
    c = cLayer.channels
    n_H = (n_Hi - f_H) ÷ s_H + 1
    n_W = (n_Wi - f_W) ÷ s_W + 1

    Ao =
        zeros(eltype(Ai), n_H, n_W, c, m)

    if NNlib
        pooldims = PoolDims(Ai, cLayer.f, stride = cLayer.s, padding = padS)
        if cLayer isa MaxPoolLayer
            maxpool!(Ao, Ai, pooldims)
        elseif cLayer isa AveragePoolLayer
            meanpool!(Ao, Ai, pooldims)
        end #if cLayer isa MaxPoolLayer
    elseif F == S #to use the built-in reshape, maximum and mean
        fastPooling!(cLayer, Ai, Ao)
    else
        pooling!(cLayer, Ai, Ao)
    end #if NNlibConv

    cLayer.outputS = size(Ao)

    Ai = nothing
    cLayer.forwCount += 1
    # Base.GC.gc()
    return Dict(:A => Ao)

end #function layerForProp(cLayer::TwoD) where {TwoD <: Union{MaxPool2D, AveragePool2D}}

function layerForProp(
    cLayer::ThreeD,
    Ai::AbstractArray = Array{Any,1}(undef,0);
    FCache::Dict{Layer,Dict{Symbol, AbstractArray}},
    kwargs...,
) where {ThreeD<:Union{MaxPool3D,AveragePool3D}}

    NNlib = haskey(kwargs, :NNlib) ? kwargs[:NNlib] : true
    if length(Ai) == 0
        Ai = FCache[cLayer.prevLayer][:A]
    end
    cLayer.inputS = size(Ai)
    # Ai = padding(cLayer, Ai)
    padS = paddingSize(cLayer, Ai)
    n_Hi, n_Wi, n_Di, ci, m = paddedSize(cLayer, Ai)
    s_H, s_W, s_D = S = cLayer.s
    f_H, f_W, f_D = F = cLayer.f
    c = cLayer.channels

    n_H = (n_Hi - f_H) ÷ s_H + 1
    n_W = (n_Wi - f_W) ÷ s_W + 1
    n_D = (n_Di - f_D) ÷ s_D + 1

    Ao =
        zeros(eltype(Ai), n_H, n_W, n_D, c, m)

    if NNlib
        pooldims = PoolDims(Ai, cLayer.f, stride = cLayer.s, padding = padS)
        if cLayer isa MaxPoolLayer
            maxpool!(Ao, Ai, pooldims)
        elseif cLayer isa AveragePoolLayer
            meanpool!(Ao, Ai, pooldims)
        end #if cLayer isa MaxPoolLayer
    elseif F == S #to use the built-in reshape and maximum and mean
        fastPooling!(cLayer, Ai, Ao)
    else
        pooling!(cLayer, Ai, Ao)
    end #if NNlibConv

    cLayer.outputS = size(Ao)

    Ai = nothing
    cLayer.forwCount += 1

    # Base.GC.gc()
    return Dict(:A => Ao)
end #function layerForProp(cLayer::ThreeD) where {ThreeD <: Union{MaxPool3D, AveragePool3D}}
