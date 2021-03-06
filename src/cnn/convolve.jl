#Naive convolution method using for loops with some parallel using
#@simd and @inbounds

function convolve!(cLayer::Conv1D, Ai::AbstractArray{T,3}) where {T}

    Aip = padding(cLayer, Ai)

    f_H = cLayer.f
    s_H = cLayer.s
    lastDim = ndims(Ai)
    n_H, c, m = size(cLayer.Z)
    W = cLayer.W
    B = cLayer.B
    @simd for mi = 1:m
        @simd for ci = 1:c
            @simd for hi = 1:n_H
                h_start = hi * s_H - (s_H == 1 ? 0 : s_H - 1)
                h_end = hi * s_H - (s_H == 1 ? 0 : s_H - 1) + f_H - 1
                @inbounds ai = Aip[h_start:h_end, :, mi]
                @inbounds cLayer.Z[hi, ci, mi] = W[:, :, ci] ⋅ ai
                # ai = nothing
                # Base.GC.gc()
            end #for mi=1:m, hi=1:n_H
        end #for ci=1:c
    end #for mi=1:m
    @inbounds cLayer.Z .+= B[:, 1, :]
    Z = cLayer.Z
    actFun = cLayer.actFun
    cLayer.A = eval(:($actFun($Z)))

    W = B = Z = nothing

    Ai = nothing

    # Base.GC.gc()

    return nothing
end #function convolve(cLayer::Conv1D


function convolve!(cLayer::Conv2D, Ai::AbstractArray{T,4}) where {T}

    Aip = padding(cLayer, Ai)

    f_H, f_W = cLayer.f
    s_H, s_W = cLayer.s
    lastDim = ndims(Ai)
    n_H, n_W, c, m = size(cLayer.Z)
    W = cLayer.W
    B = cLayer.B
    @simd for mi = 1:m
        @simd for ci = 1:c
            @simd for wi = 1:n_W
                @simd for hi = 1:n_H
                    h_start = hi * s_H - (s_H == 1 ? 0 : s_H - 1)
                    h_end = hi * s_H - (s_H == 1 ? 0 : s_H - 1) + f_H - 1
                    w_start = wi * s_W - (s_W == 1 ? 0 : s_W - 1)
                    w_end = wi * s_W - (s_W == 1 ? 0 : s_W - 1) + f_W - 1
                    @inbounds ai = Aip[h_start:h_end, w_start:w_end, :, mi]
                    @inbounds cLayer.Z[hi, wi, ci, mi] = W[:, :, :, ci] ⋅ ai
                end #for hi=1:n_H
            end #for mi=1:m, wi=1:n_W, hi=1:n_H
        end #for ci=1:c
    end #for mi=1:m
    @inbounds cLayer.Z .+= B[:, :, 1, :]
    Z = cLayer.Z
    actFun = cLayer.actFun
    cLayer.A = eval(:($actFun($Z)))



    # Base.GC.gc()

    return nothing
end #function convolve(cLayer::Conv2D

function convolve!(cLayer::Conv3D, Ai::AbstractArray{T,5}) where {T}

    Aip = padding(cLayer, Ai)

    f_H, f_W, f_D = cLayer.f
    s_H, s_W, s_D = cLayer.s
    lastDim = ndims(Ai)
    n_H, n_W, n_D, c, m = size(cLayer.Z)
    W = cLayer.W
    B = cLayer.B
    @simd for mi = 1:m
        @simd for ci = 1:c
            @simd for di = 1:n_D
                @simd for wi = 1:n_W
                    @simd for hi = 1:n_H
                        h_start = hi * s_H - (s_H == 1 ? 0 : s_H - 1)
                        h_end = hi * s_H - (s_H == 1 ? 0 : s_H - 1) + f_H - 1
                        w_start = wi * s_W - (s_W == 1 ? 0 : s_W - 1)
                        w_end = wi * s_W - (s_W == 1 ? 0 : s_W - 1) + f_W - 1
                        d_start = di * s_D - (s_D == 1 ? 0 : s_D - 1)
                        d_end = di * s_D - (s_D == 1 ? 0 : s_D - 1) + f_D - 1
                        @inbounds ai = Aip[
                            h_start:h_end,
                            w_start:w_end,
                            d_start:d_end,
                            :,
                            mi,
                        ]
                        @inbounds cLayer.Z[hi, wi, di, ci, mi] =
                            W[:, :, :, :, ci] ⋅ ai
                        # ai = nothing
                        # Base.GC.gc()
                    end #for hi=1:n_H
                end #for wi=1:n_W
            end #for di=1:n_D
        end #for ci=1:c
    end #for mi=1:m
    @inbounds cLayer.Z .+= B[:, :, :, 1, :]
    Z = cLayer.Z
    actFun = cLayer.actFun
    cLayer.A = eval(:($actFun($Z)))

    W = B = Z = nothing

    Ai = nothing

    # Base.GC.gc()

    return nothing
end #function convolve(cLayer::Conv3D


export convolve!

### dconvolve

#Naive convolution method using for loops with some parallel using
#@simd and @inbounds

function dconvolve!(
    cLayer::Conv1D,
    Ai::AbstractArray{T,3},
    dAi::AbstractArray{T,3},
    dZ::AbstractArray{T,3},
) where {T}

    Aip = padding(cLayer, Ai)
    padS = paddingSize(cLayer, Ai)
    dAip = similar(Aip)
    dAip .= 0

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
    @simd for mi = 1:m
        @simd for ci = 1:c
            for hi = 1:n_H
                h_start = hi * s_H - (s_H == 1 ? 0 : s_H - 1)
                h_end = hi * s_H - (s_H == 1 ? 0 : s_H - 1) + f_H - 1
                @inbounds ai = Aip[h_start:h_end, :, mi]
                @inbounds dAip[h_start:h_end, :, mi] .+=
                    dZ[hi, ci, mi] .* W[:, :, ci]

                @inbounds cLayer.dW[:, :, ci] .+= (ai .* dZ[hi, ci, mi])
                @inbounds cLayer.dB[:, :, ci] .+= dZ[hi, ci, mi]
            end #for
        end #for ci=1:c
    end #for mi=1:m

    dAi .= dAip[1+padS[1]:end-padS[2], :, :]

    return dAi
end #function dconvolve(cLayer::Conv1D


function dconvolve!(
    cLayer::Conv2D,
    Ai::AbstractArray{T,4},
    dAi::AbstractArray{T,4},
    dZ::AbstractArray{T,4},
) where {T}

    Aip = padding(cLayer, Ai)
    padS = paddingSize(cLayer, Ai)
    dAip = similar(Aip)
    dAip .= 0

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
    @simd for mi = 1:m
        @simd for ci = 1:c
            for wi = 1:n_W, hi = 1:n_H
                h_start = hi * s_H - (s_H == 1 ? 0 : s_H - 1)
                h_end = hi * s_H - (s_H == 1 ? 0 : s_H - 1) + f_H - 1
                w_start = wi * s_W - (s_W == 1 ? 0 : s_W - 1)
                w_end = wi * s_W - (s_W == 1 ? 0 : s_W - 1) + f_W - 1
                @inbounds ai = Aip[h_start:h_end, w_start:w_end, :, mi]
                @inbounds dAip[h_start:h_end, w_start:w_end, :, mi] .+=
                    dZ[hi, wi, ci, mi] .* W[:, :, :, ci]

                @inbounds cLayer.dW[:, :, :, ci] .+= (ai .* dZ[hi, wi, ci, mi])
                @inbounds cLayer.dB[:, :, :, ci] .+= dZ[hi, wi, ci, mi]
            end #for
        end #for ci=1:c
    end #for mi=1:m

    dAi .= dAip[1+padS[1]:end-padS[2], 1+padS[3]:end-padS[4], :, :]

    return dAi
end #function dconvolve(cLayer::Conv2D


function dconvolve!(
    cLayer::Conv3D,
    Ai::AbstractArray{T,5},
    dAi::AbstractArray{T,5},
    dZ::AbstractArray{T,5},
) where {T}

    Aip = padding(cLayer, Ai)
    padS = paddingSize(cLayer, Ai)
    dAip = similar(Aip)
    dAip .= 0

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
    @simd for mi = 1:m
        @simd for ci = 1:c
            for wi = 1:n_W, hi = 1:n_H, di = 1:n_D
                h_start = hi * s_H - (s_H == 1 ? 0 : s_H - 1)
                h_end = hi * s_H - (s_H == 1 ? 0 : s_H - 1) + f_H - 1
                w_start = wi * s_W - (s_W == 1 ? 0 : s_W - 1)
                w_end = wi * s_W - (s_W == 1 ? 0 : s_W - 1) + f_W - 1
                d_start = di * s_D - (s_D == 1 ? 0 : s_D - 1)
                d_end = di * s_D - (s_D == 1 ? 0 : s_D - 1) + f_D - 1
                @inbounds ai =
                    Aip[h_start:h_end, w_start:w_end, d_start:d_end, :, mi]
                @inbounds dAip[
                    h_start:h_end,
                    w_start:w_end,
                    d_start:d_end,
                    :,
                    mi,
                ] .+= dZ[hi, wi, di, ci, mi] .* W[:, :, :, :, ci]

                @inbounds cLayer.dW[:, :, :, :, ci] .+=
                    (ai .* dZ[hi, wi, ci, mi])
                @inbounds cLayer.dB[:, :, :, :, ci] .+= dZ[hi, wi, di, ci, mi]
            end #for
        end #for ci=1:c
    end #for mi=1:m

    dAi .= dAip[1+padS[1]:end-padS[2],
        1+padS[3]:end-padS[4],
        1+padS[5]:end-padS[6],
        :,
        :,
    ]

    return nothing
end #function dconvolve(cLayer::Conv3D


export dconvolve!
