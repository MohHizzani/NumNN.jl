#import only the needed parts not to have conflict
import NNlib.conv, NNlib.conv!

### NNConv
function NNConv(cLayer::CL, Ai::AbstractArray{T,N}) where {T,N, CL <: ConvLayer}
    padS = paddingSize(cLayer, Ai)
    axW = axes(cLayer.W)[1:end-2]
    raxW = reverse.(axW)
    Z = conv(Ai, cLayer.W[raxW..., :, :], stride = cLayer.s, pad = padS)
    axB = axes(cLayer.B)[1:end-2]
    Z .+= cLayer.B[axB..., 1, :]
    actFun = cLayer.actFun
    A = eval(:($actFun($Z)))

    return Dict(:Z => Z, :A => A)
end #function img2colConvolve(cLayer::CL

export NNConv

### dNNConv!

#import only the needed parts not to have conflict
import NNlib.∇conv_data, NNlib.∇conv_filter, NNlib.DenseConvDims

function dNNConv!(
    cLayer::CL,
    Ai::AbstractArray{T1,N},
    dAi::AbstractArray{T2,N},
    dZ::AbstractArray{T3,N},
) where {T1, T2, T3, N, CL <: ConvLayer}


    W = cLayer.W
    padS = paddingSize(cLayer, Ai)
    convdim = DenseConvDims(Ai, W, stride = cLayer.s, padding = padS)
    axW = axes(cLayer.W)[1:end-2]
    raxW = reverse.(axW)
    dAi .= ∇conv_data(dZ, W[raxW..., :, :], convdim)
    cLayer.dW[raxW..., :, :] = ∇conv_filter(Ai, dZ, convdim)

    cLayer.dB = sum(permutedims(dZ, [(1:N-2)..., N, N-1]), dims = 1:(N-1))

    return nothing
end #function img2colConvolve(cLayer::Conv1D

export dNNConv!
