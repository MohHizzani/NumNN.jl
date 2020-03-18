import NNlib.conv, NNlib.conv!, NNlib.maxpool, NNlib.meanpool, NNlib.DenseConvDims


### NNConv
function NNConv!(cLayer::Conv1D, Ai::AbstractArray{T,3})

    cLayer.Z = conv(Ai, cLayer.W[end:-1:1,:,:], stride=cLayer.s)
    Z = cLayer.Z .+ B[:,1,:]
    actFun = cLayer.actFun
    cLayer.A = eval(:($actFun($Z)))

    return nothing
end #function fastConvolve(cLayer::Conv1D


function NNConv!(cLayer::Conv2D, Ai::AbstractArray{T,4})

    cLayer.Z = conv(Ai, cLayer.W[end:-1:1,end:-1:1,:,:], stride=cLayer.s)
    Z = cLayer.Z .+ B[:,:,1,:]
    actFun = cLayer.actFun
    cLayer.A = eval(:($actFun($Z)))

    return nothing
end #function fastConvolve(cLayer::Conv2D

function NNConv!(cLayer::Conv3D, Ai::AbstractArray{T,5})

    cLayer.Z = conv(Ai, cLayer.W[end:-1:1,end:-1:1,end:-1:1,:,:], stride=cLayer.s)
    Z = cLayer.Z .+ B[:,:,:,1,:]
    actFun = cLayer.actFun
    cLayer.A = eval(:($actFun($Z)))

    return nothing
end #function fastConvolve(cLayer::Conv3D


export NNConv!


### dNNConv!

import NNlib.∇conv_data, NNlib.∇conv_filter

function dNNConv!(cLayer::Conv1D, dZ::AbstractArray{T,3}, Ai::AoN=nothing, A::AoN=nothing) where {AoN <: Union{AbstractArray, Nothing}}

    if Ai == nothing
        Ai = cLayer.prevLayer.A
    end
    if A==nothing
        A = cLayer.A
    end
    W = cLayer.W
    padS = paddingSize(cLayer,Ai)
    convdim = DenseConvDims(Ai, W, stride=cLayer.s, padding=padS)
    dA = ∇conv_data(dZ, W[end:-1:1,:,:], convdim)
    cLayer.dW[end:-1:1,:,:] = ∇conv_filter(Ai,A, convdim)
    cLayer.dB = sum(permutedims(dZ, [1,3,2]), dims=1:2)

    return nothing
end #function fastConvolve(cLayer::Conv1D


function dNNConv!(cLayer::Conv2D, dZ::AbstractArray, Ai::AoN=nothing, A::AoN=nothing) where {AoN <: Union{AbstractArray, Nothing}}

    if Ai == nothing
        Ai = cLayer.prevLayer.A
    end
    if A==nothing
        A = cLayer.A
    end
    W = cLayer.W
    convdim = DenseConvDims(Ai, W, stride=cLayer.s)
    padS = paddingSize(cLayer,Ai)
    convdim = DenseConvDims(Ai, W, stride=cLayer.s, padding=padS)
    dA = ∇conv_data(dZ, W[end:-1:1,end:-1:1,:,:], convdim)
    cLayer.dW[end:-1:1,end:-1:1,:,:] = ∇conv_filter(Ai,A,convdim)
    cLayer.dB = sum(permutedims(dZ, [1,2,4,3]), dims=1:3)

    return nothing
end #function fastConvolve(cLayer::Conv2D

function dNNConv!(cLayer::Conv3D, dZ::AbstractArray, Ai::AoN=nothing, A::AoN=nothing) where {AoN <: Union{AbstractArray, Nothing}}

    if Ai == nothing
        Ai = cLayer.prevLayer.A
    end
    if A==nothing
        A = cLayer.A
    end
    W = cLayer.W
    convdim = DenseConvDims(Ai, W, stride=cLayer.s)
    padS = paddingSize(cLayer,Ai)
    convdim = DenseConvDims(Ai, W, stride=cLayer.s, padding=padS)
    dA = ∇conv_data(dZ, W[end:-1:1,end:-1:1,end:-1:1,:,:], convdim)
    cLayer.dW[end:-1:1,end:-1:1,end:-1:1,:,:] = ∇conv_filter(Ai,A,convdim)
    cLayer.dB = sum(permutedims(dZ, [1,2,3,5,4]), dims=1:4)

    return nothing
end #function fastConvolve(cLayer::Conv3D


export dNNConv!
