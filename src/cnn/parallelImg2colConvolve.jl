
function img2colConvolve(cLayer::CL, Ai::AbstractArray{T,N}) where {T,N, CL <: ConvLayer}

    Aip = padding(cLayer, Ai)
    axBT = axes(cLayer.B)
    axB = axBT[1:end-2]
    axBend = axBT[end]
    Z = col2img(cLayer.K * img2col(Aip), cLayer.outputS) .+
            view(cLayer.B, axB..., 1, axBend)

    actFun = cLayer.actFun
    Ao = eval(:($actFun($Z)))

    return Dict(:Z => Z, :A => Ao)
end #function img2colConvolve(cLayer::CL

export img2colConvolve


### dimg2colConvolve!

export dimg2colConvolve!

function dimg2colConvolve!(
    cLayer::Conv1D,
    Ai::AbstractArray{T1,3},
    dAi::AbstractArray{T2,3},
    dZ::AbstractArray{T3,3},
) where {T1,T2,T3}

    padS = paddingSize(cLayer, Ai)
    Aip = padding(cLayer, Ai)
    Aipv = img2col(Aip)
    dZv = img2col(dZ)

    n_Hi, c, m = size(Aip)
    s_H = cLayer.s
    f_H = cLayer.f
    n_H = (n_Hi - f_H) ÷ s_H + 1

    dAi .= col2img(cLayer.K' * dZv, (n_Hi,c,m))[1+padS[1]:end-padS[2], :, :]
    cLayer.dK = dZv * Aipv'
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
    Ai::AbstractArray{T1,4},
    dAi::AbstractArray{T2,4},
    dZ::AbstractArray{T3,4},
) where {T1,T2,T3}


    padS = paddingSize(cLayer, Ai)
    Aip = padding(cLayer, Ai)
    Aipv = img2col(Aip)
    dZv = img2col(dZ)
    n_Hi, n_Wi, c, m = size(Aip)
    s_H, s_W = cLayer.s
    f_H, f_W = cLayer.f
    n_H = (n_Hi - f_H) ÷ s_H + 1
    n_W = (n_Wi - f_W) ÷ s_W + 1

    dAi .= col2img(cLayer.K' * dZv, (n_Hi,n_Wi,c,m))[1+padS[1]:end-padS[2], 1+padS[3]:end-padS[4], :, :]
    cLayer.dK = dZv * Aipv'
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
    Ai::AbstractArray{T1,5},
    dAi::AbstractArray{T2,5},
    dZ::AbstractArray{T3,5},
) where {T1,T2,T3}

    padS = paddingSize(cLayer, Ai)
    Aip = padding(cLayer, Ai)
    Aipv = img2col(Aip)
    dZv = img2col(dZ)
    n_Hi, n_Wi, n_Di, c, m = size(Aip)
    s_H, s_W, s_D = cLayer.s
    f_H, f_W, f_D = cLayer.f
    n_H = (n_Hi - f_H) ÷ s_H + 1
    n_W = (n_Wi - f_W) ÷ s_W + 1
    n_D = (n_Di - f_D) ÷ s_D + 1

    dAi .= col2img(cLayer.K' * dZv, (n_Hi, n_Wi, n_Di, c, m))[1+padS[1]:end-padS[2],
        1+padS[3]:end-padS[4],
        1+padS[5]:end-padS[6],
        :,
        :,
    ]
    cLayer.dK = dZv * Aipv'
    cLayer.dB = sum(permutedims(dZ, [1, 2, 3, 5, 4]), dims = 1:4)

    return nothing
end # function dimg2colConvolve!(
    #     cLayer::Conv3D,
    #     Ai::AbstractArray{T,5},
    #     dA::AbstractArray{T,5},
    #     dZ::AbstractArray{T,5},
    # ) where {T}
