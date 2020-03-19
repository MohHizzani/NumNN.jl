
function img2colConvolve!(cLayer::Conv1D, Ai::AbstractArray{T,3}) where {T}

    Z =
        cLayer.Z =
            col2img1D(cLayer.K * img2col(Ai), cLayer.outputS) .+
            cLayer.B[:, 1, :]

    actFun = cLayer.actFun
    cLayer.A = eval(:($actFun($Z)))

    return nothing
end #function img2colConvolve(cLayer::Conv1D


function img2colConvolve!(cLayer::Conv2D, Ai::AbstractArray{T,4}) where {T}

    Z =
        cLayer.Z =
            col2img2D(cLayer.K * img2col(Ai), cLayer.outputS) .+
            cLayer.B[:, :, 1, :]

    actFun = cLayer.actFun
    cLayer.A = eval(:($actFun($Z)))

    return nothing
end #function img2colConvolve(cLayer::Conv2D

function img2colConvolve!(cLayer::Conv3D, Ai::AbstractArray{T,5}) where {T}

    Z =
        cLayer.Z =
            col2img3D(cLayer.K * img2col(Ai), cLayer.outputS) .+
            cLayer.B[:, :, :, 1, :]

    actFun = cLayer.actFun
    cLayer.A = eval(:($actFun($Z)))

    return nothing
end #function img2colConvolve(cLayer::Conv3D


export img2colConvolve!
