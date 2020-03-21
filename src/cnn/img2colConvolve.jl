
function img2colConvolve!(cLayer::Conv1D, Ai::AbstractArray{T,3}) where {T}

    Aip = padding(cLayer, Ai)

    Z =
        cLayer.Z =
            col2img1D(cLayer.K * img2col(Aip), cLayer.outputS) .+
            cLayer.B[:, 1, :]

    actFun = cLayer.actFun
    cLayer.A = eval(:($actFun($Z)))

    return nothing
end #function img2colConvolve(cLayer::Conv1D


function img2colConvolve!(cLayer::Conv2D, Ai::AbstractArray{T,4}) where {T}

    Aip = padding(cLayer, Ai)

    Z =
        cLayer.Z =
            col2img2D(cLayer.K * img2col(Aip), cLayer.outputS) .+
            cLayer.B[:, :, 1, :]

    actFun = cLayer.actFun
    cLayer.A = eval(:($actFun($Z)))

    return nothing
end #function img2colConvolve(cLayer::Conv2D

function img2colConvolve!(cLayer::Conv3D, Ai::AbstractArray{T,5}) where {T}

    Aip = padding(cLayer, Ai)

    Z =
        cLayer.Z =
            col2img3D(cLayer.K * img2col(Aip), cLayer.outputS) .+
            cLayer.B[:, :, :, 1, :]

    actFun = cLayer.actFun
    cLayer.A = eval(:($actFun($Z)))

    return nothing
end #function img2colConvolve(cLayer::Conv3D


export img2colConvolve!

### dimg2colConvolve!

export dimg2colConvolve!

function dimg2colConvolve!(
    cLayer::Conv1D,
    Ai::AbstractArray{T,3},
    dAi::AbstractArray{T,3},
    dZ::AbstractArray{T,3},
) where {T}

    Aip = padding(cLayer, Ai)

    cLayer.dA = col2img3D(cLayer.K' * img2col(dZ), cLayer.outputS)
    cLayer.dK = dZ * Aip'
    cLayer.dB = sum(permutedims(dZ, [1, 3, 2]), dims = 1:2)

    return nothing
end # function dimg2colConvolve!(
    #     cLayer::Conv1D,
    #     Ai::AbstractArray{T,3},
    #     dA::AbstractArray{T,3},
    #     dZ::AbstractArray{T,3},
    # ) where {T}


function dimg2colConvolve!(
    cLayer::Conv2D,
    Ai::AbstractArray{T,4},
    dAi::AbstractArray{T,4},
    dZ::AbstractArray{T,4},
) where {T}

    Aip = padding(cLayer, Ai)

    cLayer.dA = col2img3D(cLayer.K' * img2col(dZ), cLayer.outputS)
    cLayer.dK = dZ * Aip'
    cLayer.dB = sum(permutedims(dZ, [1, 2, 4, 3]), dims = 1:3)

    return nothing
end # function dimg2colConvolve!(
    #     cLayer::Conv2D,
    #     Ai::AbstractArray{T,4},
    #     dA::AbstractArray{T,4},
    #     dZ::AbstractArray{T,4},
    # ) where {T}

function dimg2colConvolve!(
    cLayer::Conv3D,
    Ai::AbstractArray{T,5},
    dAi::AbstractArray{T,5},
    dZ::AbstractArray{T,5},
) where {T}

    Aip = padding(cLayer, Ai)

    cLayer.dA = col2img3D(cLayer.K' * img2col(dZ), cLayer.outputS)
    cLayer.dK = dZ * Aip'
    cLayer.dB = sum(permutedims(dZ, [1, 2, 3, 5, 4]), dims = 1:4)

    return nothing
end # function dimg2colConvolve!(
    #     cLayer::Conv3D,
    #     Ai::AbstractArray{T,5},
    #     dA::AbstractArray{T,5},
    #     dZ::AbstractArray{T,5},
    # ) where {T}
