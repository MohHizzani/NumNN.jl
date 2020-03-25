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
