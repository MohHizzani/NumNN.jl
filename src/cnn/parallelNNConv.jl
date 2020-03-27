#import only the needed parts not to have conflict
import NNlib.conv, NNlib.conv!

### NNConv
function NNConv(cLayer::Conv1D, Ai::AbstractArray{T,3}) where {T}
    padS = paddingSize(cLayer, Ai)
    Z = conv(Ai, cLayer.W[end:-1:1, :, :], stride = cLayer.s, pad = padS)
    Z .+= cLayer.B[:, 1, :]
    actFun = cLayer.actFun
    A = eval(:($actFun($Z)))

    return Dict(:Z => Z, :A => A)
end #function img2colConvolve(cLayer::Conv1D


function NNConv(cLayer::Conv2D, Ai::AbstractArray{T,4}) where {T}
    padS = paddingSize(cLayer, Ai)
    Z = conv(Ai, cLayer.W[end:-1:1, end:-1:1, :, :], stride = cLayer.s, pad = padS)
    Z = Z .+= cLayer.B[:, :, 1, :]
    actFun = cLayer.actFun
    A = eval(:($actFun($Z)))

    return Dict(:Z => Z, :A => A)
end #function img2colConvolve(cLayer::Conv2D

function NNConv(cLayer::Conv3D, Ai::AbstractArray{T,5}) where {T}
    padS = paddingSize(cLayer, Ai)
    Z = conv(
        Ai,
        cLayer.W[end:-1:1, end:-1:1, end:-1:1, :, :],
        stride = cLayer.s,
        pad = padS,
    )
    Z .+= cLayer.B[:, :, :, 1, :]
    actFun = cLayer.actFun
    A = eval(:($actFun($Z)))

    return Dict(:Z => Z, :A => A)
end #function img2colConvolve(cLayer::Conv3D


export NNConv

### dNNConv!

#import only the needed parts not to have conflict
import NNlib.∇conv_data, NNlib.∇conv_filter, NNlib.DenseConvDims

function dNNConv!(
    cLayer::Conv1D,
    Ai::AbstractArray{T1,3},
    dAi::AbstractArray{T2,3},
    dZ::AbstractArray{T3,3},
) where {T1, T2, T3}


    W = cLayer.W
    padS = paddingSize(cLayer, Ai)
    convdim = DenseConvDims(Ai, W, stride = cLayer.s, padding = padS)
    dAi .= ∇conv_data(dZ, W[end:-1:1, :, :], convdim)
    cLayer.dW[end:-1:1, :, :] = ∇conv_filter(Ai, dZ, convdim)
    cLayer.dB = sum(permutedims(dZ, [1, 3, 2]), dims = 1:2)

    return nothing
end #function img2colConvolve(cLayer::Conv1D


function dNNConv!(
    cLayer::Conv2D,
    Ai::AbstractArray{T1,4},
    dAi::AbstractArray{T2,4},
    dZ::AbstractArray{T3,4},
) where {T1, T2, T3}

    W = cLayer.W
    convdim = DenseConvDims(Ai, W, stride = cLayer.s)
    padS = paddingSize(cLayer, Ai)
    convdim = DenseConvDims(Ai, W, stride = cLayer.s, padding = padS)
    dAi = ∇conv_data(dZ, W[end:-1:1, end:-1:1, :, :], convdim)
    cLayer.dW[end:-1:1, end:-1:1, :, :] = ∇conv_filter(Ai, dZ, convdim)
    cLayer.dB = sum(permutedims(dZ, [1, 2, 4, 3]), dims = 1:3)

    return nothing
end #function img2colConvolve(cLayer::Conv2D

function dNNConv!(
    cLayer::Conv3D,
    Ai::AbstractArray{T1,5},
    dAi::AbstractArray{T2,5},
    dZ::AbstractArray{T3,5},
) where {T1, T2, T3}

    W = cLayer.W
    convdim = DenseConvDims(Ai, W, stride = cLayer.s)
    padS = paddingSize(cLayer, Ai)
    convdim = DenseConvDims(Ai, W, stride = cLayer.s, padding = padS)
    dAi = ∇conv_data(dZ, W[end:-1:1, end:-1:1, end:-1:1, :, :], convdim)
    cLayer.dW[end:-1:1, end:-1:1, end:-1:1, :, :] = ∇conv_filter(Ai, dZ, convdim)
    cLayer.dB = sum(permutedims(dZ, [1, 2, 3, 5, 4]), dims = 1:4)

    return nothing
end #function img2colConvolve(cLayer::Conv3D


export dNNConv!
