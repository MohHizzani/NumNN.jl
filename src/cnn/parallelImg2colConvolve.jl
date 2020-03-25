
function img2colConvolve(cLayer::Conv1D, Ai::AbstractArray{T,3}) where {T}

    Aip = padding(cLayer, Ai)

    Z = col2img1D(cLayer.K * img2col(Aip), cLayer.outputS) .+
            cLayer.B[:, 1, :]

    actFun = cLayer.actFun
    Ao = eval(:($actFun($Z)))

    return Dict(:Z => Z, :A => Ao)
end #function img2colConvolve(cLayer::Conv1D


function img2colConvolve(cLayer::Conv2D, Ai::AbstractArray{T,4}) where {T}

    Aip = padding(cLayer, Ai)

    Z = col2img2D(cLayer.K * img2col(Aip), cLayer.outputS) .+
            cLayer.B[:, :, 1, :]

    actFun = cLayer.actFun
    Ao = eval(:($actFun($Z)))

    return Dict(:Z => Z, :A => Ao)
end #function img2colConvolve(cLayer::Conv2D

function img2colConvolve(cLayer::Conv3D, Ai::AbstractArray{T,5}) where {T}

    Aip = padding(cLayer, Ai)

    Z = col2img3D(cLayer.K * img2col(Aip), cLayer.outputS) .+
            cLayer.B[:, :, :, 1, :]

    actFun = cLayer.actFun
    Ao = eval(:($actFun($Z)))

    return Dict(:Z => Z, :A => Ao)
end #function img2colConvolve(cLayer::Conv3D


export img2colConvolve
