

###convolution layers forprop

function layerForProp!(
    cLayer::Conv1D,
    Ai::AbstractArray = Array{Any,1}(undef,0);
    img2colConvolve::Bool = false,
    NNlib::Bool = true,
) 
    if length(Ai) == 0
        Ai = cLayer.prevLayer.A
    end
    cLayer.inputS = size(Ai)
    # Ai = padding(cLayer, Ai)
    n_Hi, ci, m = paddedSize(cLayer, Ai)
    s_H = cLayer.s
    f_H = cLayer.f
    c = cLayer.channels
    n_H = (n_Hi - f_H) ÷ s_H + 1

    if NNlib
        NNConv!(cLayer, Ai)
    elseif img2colConvolve
        ## in case input different size than previous time
        if (n_H * c, n_Hi * ci) != size(cLayer.K)
            cLayer.K = unroll(cLayer, (n_Hi, ci, m))
        end
        img2colConvolve!(cLayer, Ai)
    else
        cLayer.Z = zeros(eltype(Ai), n_H, cLayer.channels, m)
        convolve!(cLayer, Ai)
    end

    cLayer.outputS = size(cLayer.A)

    Ai = nothing
    cLayer.forwCount += 1
    # Base.GC.gc()
    return nothing

end #function layerForProp!(cLayer::Conv1D)

function layerForProp!(
    cLayer::Conv2D,
    Ai::AbstractArray = Array{Any,1}(undef,0);
    img2colConvolve::Bool = false,
    NNlib::Bool = true,
)
    if length(Ai) == 0
        Ai = cLayer.prevLayer.A
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
        NNConv!(cLayer, Ai)
    elseif img2colConvolve
        ## in case input different size than previous time
        if (n_W * n_H * c, n_Hi * n_Wi * ci) != size(cLayer.K)
            cLayer.K = unroll(cLayer, (n_Hi, n_Wi, ci, m))
        end #if (n_W*n_H* c, n_Hi*n_Wi*ci) != size(cLayer.K)
        img2colConvolve!(cLayer, Ai)
    else
        cLayer.Z = zeros(eltype(Ai), n_H, n_W, cLayer.channels, m)
        convolve!(cLayer, Ai)
    end #if img2colConvolve

    cLayer.outputS = size(cLayer.A)

    Ai = nothing
    cLayer.forwCount += 1

    # Base.GC.gc()
    return nothing
end #function layerForProp!(cLayer::Conv2D)

function layerForProp!(
    cLayer::Conv3D,
    Ai::AbstractArray = Array{Any,1}(undef,0);
    img2colConvolve::Bool = false,
    NNlib::Bool = true,
)
    if length(Ai) == 0
        Ai = cLayer.prevLayer.A
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
        NNConv!(cLayer, Ai)
    elseif img2colConvolve
        ## in case input different size than previous time
        if (n_W * n_H * n_D * c, n_Hi * n_Wi * n_Di * ci) != size(cLayer.K)
            cLayer.K = unroll(cLayer, (n_Hi, n_Wi, n_Di, ci, m))
        end #if (n_W*n_H* c, n_Hi*n_Wi*ci) != size(cLayer.K)
        img2colConvolve!(cLayer, Ai)
    else
        cLayer.Z = zeros(eltype(Ai), n_H, n_W, n_D, cLayer.channels, m)
        convolve!(cLayer, Ai)
    end #if img2colConvolve

    cLayer.outputS = size(cLayer.A)

    cLayer.forwCount += 1

    Ai = nothing

    # Base.GC.gc()
    return nothing
end #function layerForProp!(cLayer::Conv3D)



### Pooling Layers

#import only the needed parts not to have conflict
import NNlib.maxpool, NNlib.meanpool, NNlib.maxpool!, NNlib.meanpool!, NNlib.PoolDims

function layerForProp!(
    cLayer::OneD,
    Ai::AbstractArray = Array{Any,1}(undef,0);
    NNlib::Bool = true,
) where {OneD<:Union{MaxPool1D,AveragePool1D},AoN<:Union{AbstractArray,Nothing}}
    if length(Ai) == 0
        Ai = cLayer.prevLayer.A
    end
    cLayer.inputS = size(Ai)
    # Ai = padding(cLayer, Ai)
    padS = paddingSize(cLayer, Ai)
    n_Hi, ci, m = paddedSize(cLayer, Ai)
    s_H = cLayer.s
    f_H = cLayer.f
    c = cLayer.channels

    n_H = (n_Hi - f_H) ÷ s_H + 1

    cLayer.A = zeros(eltype(Ai), n_H, c, m)

    if NNlib
        pooldims = PoolDims(Ai, cLayer.f, stride = cLayer.s, padding = padS)
        if cLayer isa MaxPoolLayer
            maxpool!(cLayer.A, Ai, pooldims)
        elseif cLayer isa AveragePoolLayer
            meanpool!(cLayer.A, Ai, pooldims)
        end #if cLayer isa MaxPoolLayer
    elseif f_H == s_H #to use the built-in reshape and maximum and mean
        fastPooling!(cLayer, Ai)
    else
        pooling!(cLayer, Ai)
    end #if NNlibConv

    cLayer.outputS = size(cLayer.A)
    Ai = nothing
    cLayer.forwCount += 1
    # Base.GC.gc()
    return nothing
end #unction layerForProp!(cLayer::OneD) where {OneD <: Union{MaxPool1D, AveragePool1D}}

function layerForProp!(
    cLayer::TwoD,
    Ai::AbstractArray = Array{Any,1}(undef,0);
    NNlib::Bool = true,
) where {TwoD<:Union{MaxPool2D,AveragePool2D},AoN<:Union{AbstractArray,Nothing}}
    if length(Ai) == 0
        Ai = cLayer.prevLayer.A
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

    cLayer.A =
        zeros(eltype(Ai), n_H, n_W, c, m)

    if NNlib
        pooldims = PoolDims(Ai, cLayer.f, stride = cLayer.s, padding = padS)
        if cLayer isa MaxPoolLayer
            maxpool!(cLayer.A, Ai, pooldims)
        elseif cLayer isa AveragePoolLayer
            meanpool!(cLayer.A, Ai, pooldims)
        end #if cLayer isa MaxPoolLayer
    elseif F == S #to use the built-in reshape, maximum and mean
        fastPooling!(cLayer, Ai)
    else
        pooling!(cLayer, Ai)
    end #if NNlibConv

    cLayer.outputS = size(cLayer.A)

    Ai = nothing
    cLayer.forwCount += 1
    # Base.GC.gc()
    return nothing

end #function layerForProp!(cLayer::TwoD) where {TwoD <: Union{MaxPool2D, AveragePool2D}}

function layerForProp!(
    cLayer::ThreeD,
    Ai::AbstractArray = Array{Any,1}(undef,0);
    NNlib::Bool = true,
) where {
    ThreeD<:Union{MaxPool3D,AveragePool3D},
    AoN<:Union{AbstractArray,Nothing},
}
    if length(Ai) == 0
        Ai = cLayer.prevLayer.A
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

    cLayer.A =
        zeros(eltype(Ai), n_H, n_W, n_D, c, m)

    if NNlib
        pooldims = PoolDims(Ai, cLayer.f, stride = cLayer.s, padding = padS)
        if cLayer isa MaxPoolLayer
            maxpool!(cLayer.A, Ai, pooldims)
        elseif cLayer isa AveragePoolLayer
            meanpool!(cLayer.A, Ai, pooldims)
        end #if cLayer isa MaxPoolLayer
    elseif F == S #to use the built-in reshape and maximum and mean
        fastPooling!(cLayer, Ai)
    else
        pooling!(cLayer, Ai)
    end #if NNlibConv

    cLayer.outputS = size(cLayer.A)

    Ai = nothing
    cLayer.forwCount += 1

    # Base.GC.gc()
    return nothing
end #function layerForProp!(cLayer::ThreeD) where {ThreeD <: Union{MaxPool3D, AveragePool3D}}
