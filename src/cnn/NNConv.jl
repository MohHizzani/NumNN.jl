import NNlib.conv, NNlib.conv!, NNlib.maxpool, NNlib.meanpool


###
function NNConv!(cLayer::Conv1D, Ai::AbstractArray)

    cLayer.Z = conv(Ai, cLayer.W[end:-1:1,:,:], stride=cLayer.s)
    Z = cLayer.Z .+ B[:,1,:]
    actFun = cLayer.actFun
    cLayer.A = eval(:($actFun($Z)))

    return nothing
end #function fastConvolve(cLayer::Conv1D


function NNConv!(cLayer::Conv2D, Ai::AbstractArray)

    cLayer.Z = conv(Ai, cLayer.W[end:-1:1,end:-1:1,:,:], stride=cLayer.s)
    Z = cLayer.Z .+ B[:,1,:]
    actFun = cLayer.actFun
    cLayer.A = eval(:($actFun($Z)))

    return nothing
end #function fastConvolve(cLayer::Conv2D

function NNConv!(cLayer::Conv3D, Ai::AbstractArray)

    cLayer.Z = conv(Ai, cLayer.W[end:-1:1,end:-1:1,end:-1:1,:,:], stride=cLayer.s)
    Z = cLayer.Z .+ B[:,1,:]
    actFun = cLayer.actFun
    cLayer.A = eval(:($actFun($Z)))

    return nothing
end #function fastConvolve(cLayer::Conv3D


export NNConv!
