
function convolve!(cLayer::Conv1D,
                  Ai)

    f_H = cLayer.f
    s_H = cLayer.s
    lastDim = ndims(Ai)
    n_H, c, m = size(cLayer.Z)
    W = cLayer.W
    B = cLayer.B
    for mi=1:m, ci=1:c, hi=1:n_H
        h_start = hi*s_H
        h_end = hi*s_H + f_H -1
        ai = Ai[h_start:h_end, :, mi]
        cLayer.Z[hi, ci, mi] = W[ci] ⋅ ai
    end #for
    cLayer.Z .+= B
    Z = cLayer.Z
    actFun = cLayer.actFun
    cLayer.A = eval(:($actFun($Z)))

    return nothing
end #function convolve(cLayer::Conv2D


function convolve!(cLayer::Conv2D,
                  Ai)

    f_H, f_W = cLayer.f
    s_H, s_W = cLayer.s
    lastDim = ndims(Ai)
    n_H, n_W, c, m = size(cLayer.Z)
    W = cLayer.W
    B = cLayer.B
    for mi=1:m, ci=1:c, wi=1:n_W, hi=1:n_H
        h_start = hi*s_H
        h_end = hi*s_H + f_H -1
        w_start = wi*s_W
        w_end = wi*s_W + f_W -1
        ai = Ai[h_start:h_end, w_start:w_end, :, mi]
        cLayer.Z[hi, wi, ci, mi] = W[ci] ⋅ ai
    end #for
    cLayer.Z .+= B
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
    for mi=1:m, ci=1:c, wi=1:n_W, hi=1:n_H, di=1:n_D
        h_start = hi*s_H
        h_end = hi*s_H + f_H -1
        w_start = wi*s_W
        w_end = wi*s_W + f_W -1
        d_start = di*s_D
        d_end = di*s_D + f_D -1
        ai = Ai[h_start:h_end, w_start:w_end, d_start:d_end, :, mi]
        cLayer.Z[hi, wi, di, ci, mi] = W[ci] ⋅ ai
    end #for
    cLayer.Z .+= B
    Z = cLayer.Z
    actFun = cLayer.actFun
    cLayer.A = eval(:($actFun($Z)))

    return nothing
end #function convolve(cLayer::Conv3D


export convolve!
