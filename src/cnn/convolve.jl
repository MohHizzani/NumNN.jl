
function convolve!(cLayer::Conv1D,
                  Ai)

    f_H = cLayer.f
    s_H = cLayer.s
    lastDim = ndims(Ai)
    n_H, c, m = size(cLayer.Z)
    W = cLayer.W
    B = cLayer.B
    for ci=1:c
        for mi=1:m, hi=1:n_H
            h_start = hi*s_H - (s_H == 1 ? 0 : 1)
            h_end = hi*s_H - (s_H == 1 ? 0 : 1) + f_H -1
            ai = Ai[h_start:h_end, :, mi]
            cLayer.Z[hi, ci, mi] = W[:,:,ci] ⋅ ai
        end #for mi=1:m, hi=1:n_H
        cLayer.Z .+= B[:,:,ci]
    end #for ci=1:c
    Z = cLayer.Z
    actFun = cLayer.actFun
    cLayer.A = eval(:($actFun($Z)))

    return nothing
end #function convolve(cLayer::Conv1D


function convolve!(cLayer::Conv2D,
                  Ai)

    f_H, f_W = cLayer.f
    s_H, s_W = cLayer.s
    lastDim = ndims(Ai)
    n_H, n_W, c, m = size(cLayer.Z)
    W = cLayer.W
    B = cLayer.B
    for ci=1:c
        for mi=1:m, wi=1:n_W, hi=1:n_H
            h_start = hi* s_H - (s_H == 1 ? 0 : 1)
            h_end = hi*s_H - (s_H == 1 ? 0 : 1) + f_H -1
            w_start = wi*s_W - (s_W == 1 ? 0 : 1)
            w_end = wi*s_W - (s_W == 1 ? 0 : 1) + f_W -1
            ai = Ai[h_start:h_end, w_start:w_end, :, mi]
            cLayer.Z[hi, wi, ci, mi] = W[:,:,:,ci] ⋅ ai
        end #for mi=1:m, wi=1:n_W, hi=1:n_H
        cLayer.Z .+= B[:,:,:,ci]
    end #for ci=1:c
    Z = cLayer.Z
    actFun = cLayer.actFun
    cLayer.A = eval(:($actFun($Z)))

    return nothing
end #function convolve(cLayer::Conv2D

function convolve!(cLayer::Conv3D,
                  Ai)

    f_H, f_W, f_D = cLayer.f
    s_H, s_W, s_D = cLayer.s
    lastDim = ndims(Ai)
    n_H, n_W, n_D, c, m = size(cLayer.Z)
    W = cLayer.W
    B = cLayer.B
    for ci=1:c
        for mi=1:m, wi=1:n_W, hi=1:n_H, di=1:n_D
            h_start = hi*s_H - (s_H == 1 ? 0 : 1)
            h_end = hi*s_H - (s_H == 1 ? 0 : 1) + f_H -1
            w_start = wi*s_W - (s_W == 1 ? 0 : 1)
            w_end = wi*s_W - (s_W == 1 ? 0 : 1) + f_W -1
            d_start = di*s_D - (s_D == 1 ? 0 : 1)
            d_end = di*s_D - (s_D == 1 ? 0 : 1) + f_D -1
            ai = Ai[h_start:h_end, w_start:w_end, d_start:d_end, :, mi]
            cLayer.Z[hi, wi, di, ci, mi] = W[:,:,:,:,ci] ⋅ ai
        end #for
        cLayer.Z .+= B[:,:,:,:, ci]
    end #for ci=1:c
    Z = cLayer.Z
    actFun = cLayer.actFun
    cLayer.A = eval(:($actFun($Z)))

    return nothing
end #function convolve(cLayer::Conv3D


export convolve!

### dconvolve


function dconvolve!(
                    cLayer::Conv1D,
                    Ai::AbstractArray,
                    dAi::AbstractArray,
                    dZ::AbstractArray,
                    )

    f_H = cLayer.f
    s_H = cLayer.s
    lastDim = ndims(Ai)
    n_H, c, m = size(cLayer.Z)
    W = cLayer.W
    cLayer.dW = similar(W)
    cLayer.dW .= 0
    B = cLayer.B
    cLayer.dB = similar(B)
    cLayer.dB .= 0
    for mi=1:m, ci=1:c, hi=1:n_H
        h_start = hi*s_H - (s_H == 1 ? 0 : 1)
        h_end = hi*s_H - (s_H == 1 ? 0 : 1) + f_H -1
        ai = Ai[h_start:h_end, :, mi]
        dAi[h_start:h_end, :, mi] .+= cLayer.Z[hi, ci, mi] .* W[:,:,ci]

        cLayer.dW[:,:,ci] .+= (ai .* cLayer.dZ[hi, ci, mi])
        cLayer.dB[:,:,ci] .+= cLayer.dZ[hi,ci,mi]
    end #for

    cLayer.dA = dAi

    return nothing
end #function dconvolve(cLayer::Conv1D


function dconvolve!(
                    cLayer::Conv2D,
                    Ai::AbstractArray,
                    dAi::AbstractArray,
                    dZ::AbstractArray,
                    )

    f_H, f_W = cLayer.f
    s_H, s_W = cLayer.s
    lastDim = ndims(Ai)
    n_H, n_W, c, m = size(cLayer.Z)
    W = cLayer.W
    cLayer.dW = similar(W)
    cLayer.dW .= 0
    B = cLayer.B
    cLayer.dB = similar(B)
    cLayer.dB .= 0
    for mi=1:m, ci=1:c, wi=1:n_W, hi=1:n_H
        h_start = hi* s_H - (s_H == 1 ? 0 : 1)
        h_end = hi*s_H - (s_H == 1 ? 0 : 1) + f_H -1
        w_start = wi*s_W - (s_W == 1 ? 0 : 1)
        w_end = wi*s_W - (s_W == 1 ? 0 : 1) + f_W -1
        ai = Ai[h_start:h_end, w_start:w_end, :, mi]
        dAi[h_start:h_end, w_start:w_end, :, mi] .+= cLayer.Z[hi, wi, ci, mi] .* W[:,:,:,ci]

        cLayer.dW[:,:,:,ci] .+= (ai .* cLayer.dZ[hi, wi, ci, mi])
        cLayer.dB[:,:.:,ci] .+= cLayer.dZ[hi, wi,ci,mi]
    end #for

    cLayer.dA = dAi

    return nothing
end #function dconvolve(cLayer::Conv2D


function dconvolve!(
                    cLayer::Conv3D,
                    Ai::AbstractArray,
                    dAi::AbstractArray,
                    dZ::AbstractArray,
                    )

    f_H, f_W, f_D = cLayer.f
    s_H, s_W, s_D = cLayer.s
    lastDim = ndims(Ai)
    n_H, n_W, n_D, c, m = size(cLayer.Z)
    W = cLayer.W
    cLayer.dW = similar(W)
    cLayer.dW .= 0
    B = cLayer.B
    cLayer.dB = similar(B)
    cLayer.dB .= 0
    for mi=1:m, ci=1:c, wi=1:n_W, hi=1:n_H, di=1:n_D
        h_start = hi*s_H - (s_H == 1 ? 0 : 1)
        h_end = hi*s_H - (s_H == 1 ? 0 : 1) + f_H -1
        w_start = wi*s_W - (s_W == 1 ? 0 : 1)
        w_end = wi*s_W - (s_W == 1 ? 0 : 1) + f_W -1
        d_start = di*s_D - (s_D == 1 ? 0 : 1)
        d_end = di*s_D - (s_D == 1 ? 0 : 1) + f_D -1
        ai = Ai[h_start:h_end, w_start:w_end, d_start:d_end, :, mi]
        dAi[h_start:h_end, w_start:w_end, d_start:d_end, :, mi] .+= cLayer.Z[hi, wi, di, ci, mi] .* W[:,:,:,:,ci]

        cLayer.dW[:,:,:,:,ci] .+= (ai .* cLayer.dZ[hi, wi, ci, mi])
        cLayer.dB[:,:.:,:,ci] .+= cLayer.dZ[hi,wi,di,ci,mi]
    end #for

    cLayer.dA = dAi

    return nothing
end #function dconvolve(cLayer::Conv3D


export dconvolve!
