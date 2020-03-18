
function fastConvolve!(cLayer::Conv1D, Ai::AbstractArray)

    Z = cLayer.Z = col2img1D(cLayer.K*img2col(Ai), cLayer.outputS) .+ cLayer.B[:,1,:]

    actFun = cLayer.actFun
    cLayer.A = eval(:($actFun($Z)))

    return nothing
end #function fastConvolve(cLayer::Conv1D


function fastConvolve!(cLayer::Conv2D, Ai::AbstractArray)

    Z = cLayer.Z = col2img2D(cLayer.K*img2col(Ai), cLayer.outputS) .+ cLayer.B[:,:,1,:]

    actFun = cLayer.actFun
    cLayer.A = eval(:($actFun($Z)))

    return nothing
end #function fastConvolve(cLayer::Conv2D

function fastConvolve!(cLayer::Conv3D, Ai::AbstractArray)

    Z = cLayer.Z = col2img3D(cLayer.K*img2col(Ai), cLayer.outputS) .+ cLayer.B[:,:,:,1,:]

    actFun = cLayer.actFun
    cLayer.A = eval(:($actFun($Z)))

    return nothing
end #function fastConvolve(cLayer::Conv3D


export fastConvolve!
