#import only the needed parts not to have conflict
import NNlib.conv, NNlib.conv!

### NNConv
@doc raw"""
    NNConv(cLayer::CL, Ai::AbstractArray{T,N}) where {T,N, CL <: ConvLayer}

Perform the forward propagation for `cLayer::ConvLayer` using fast implementation of `NNlib`

# Return

- `Dict(:Z => Z, :A => A)`
"""
function NNConv(cLayer::CL, Ai::AbstractArray{T,N}) where {T,N, CL <: ConvLayer}
    padS = paddingSize(cLayer, Ai)
    # axW = axes(cLayer.W)[1:end-2]
    # raxW = reverse.(axW)
    # Z = conv(Ai, cLayer.W[raxW..., :, :], stride = cLayer.s, pad = padS)
    Z = conv(Ai, cLayer.W, stride = cLayer.s, pad = padS, flipped=true)
    axB = axes(cLayer.B)[1:end-2]
    Z .+= cLayer.B[axB..., 1, :]
    actFun = cLayer.actFun
    A = eval(:($actFun($Z)))

    return Dict(:Z => Z, :A => A)
end #function img2colConvolve(cLayer::CL

export NNConv

### dNNConv!

#import only the needed parts not to have conflict
import NNlib.∇conv_data!, NNlib.∇conv_filter, NNlib.DenseConvDims

@doc raw"""
    function dNNConv!(
        cLayer::CL,
        Ai::AbstractArray{T1,N},
        dAi::AbstractArray{T2,N},
        dZ::AbstractArray{T3,N},
    ) where {T1, T2, T3, N, CL <: ConvLayer}

Performs the back propagation for `cLayer::ConvLayer` and save values to the pre-allocated `Array` `dAi` and trainable parameters `W` & `B`

# Arguments

- `cLayer::ConvLayer`

- `Ai::AbstractArray{T1,N}` := the input activation of `cLayer`

- `dAi::AbstractArray{T2,N}` := pre-allocated to hold the derivative of the activation

- `dZ::AbstractArray{T3,N}` := the derivative of the cost to the input of the activation function

# Return

`nothing`

"""
function dNNConv!(
    cLayer::CL,
    Ai::AbstractArray{T1,N},
    dAi::AbstractArray{T2,N},
    dZ::AbstractArray{T3,N},
) where {T1, T2, T3, N, CL <: ConvLayer}


    W = cLayer.W
    padS = paddingSize(cLayer, Ai)
    convdim = DenseConvDims(Ai, W, stride = cLayer.s, padding = padS, flipkernel=true)
    # axW = axes(cLayer.W)[1:end-2]
    # raxW = reverse.(axW)
    # dAi .= ∇conv_data(dZ, W[raxW..., :, :], convdim)
    ∇conv_data!(dAi, dZ, W, convdim)
    cLayer.dW = ∇conv_filter(Ai, dZ, convdim)

    cLayer.dB = sum(permutedims(dZ, [(1:N-2)..., N, N-1]), dims = 1:(N-1))

    return nothing
end #function img2colConvolve(cLayer::Conv1D

export dNNConv!
