#Naive convolution method using for loops with some parallel using
#@simd and @inbounds

function convolve(cLayer::Conv1D, Ai::AbstractArray{T1,3}) where {T1}

    Aip = padding(cLayer, Ai)

    f_H = cLayer.f
    s_H = cLayer.s
    lastDim = ndims(Ai)
    paddedS = paddedSize(cLayer, Ai)[1:end-2]
    s = cLayer.s
    f = cLayer.f
    outputS = outDims(cLayer, Ai)[1:end-1]
    ci, m = cLayer.inputS[end], size(Ai)[end]
    co = cLayer.channels
    Z = zeros(eltype(Ai), outputS..., co, m)
    n_H, c, m = size(Z)
    W = cLayer.W
    B = cLayer.B
    # @simd
    Threads.@threads for mi = 1:m
        # @simd for
        @simd for ci = 1:c
            # @simd
            for hi = 1:n_H
                h_start = hi * s_H - (s_H == 1 ? 0 : s_H - 1)
                h_end = hi * s_H - (s_H == 1 ? 0 : s_H - 1) + f_H - 1
                @inbounds ai = view(Aip,h_start:h_end, :, mi)
                @inbounds Z[hi, ci, mi] = W[:, :, ci] ⋅ ai + B[1,1,ci]
                # ai = nothing
                # Base.GC.gc()
            end #for mi=1:m, hi=1:n_H
        end #for ci=1:c
    end #for mi=1:m
    # @inbounds Z .+= B[:, 1, :]
    actFun = cLayer.actFun
    Ao = eval(:($actFun($Z)))

    # Base.GC.gc()

    return Dict(:Z => Z, :A => Ao)
end #function convolve(cLayer::Conv1D


function convolve(cLayer::Conv2D, Ai::AbstractArray{T1,4}) where {T1}

    Aip = padding(cLayer, Ai)

    f_H, f_W = cLayer.f
    s_H, s_W = cLayer.s
    lastDim = ndims(Ai)
    cLayer.inputS = size(Ai)[1:end-1]
    paddedS = paddedSize(cLayer, Ai)[1:end-2]
    s = cLayer.s
    f = cLayer.f
    outputS = outDims(cLayer, Ai)[1:end-1]
    ci, m = cLayer.inputS[end], size(Ai)[end]
    co = cLayer.channels
    Z = zeros(eltype(Ai), outputS..., co, m)
    n_H, n_W, c, m = size(Z)
    W = cLayer.W
    B = cLayer.B
    # @simd
    Threads.@threads for mi = 1:m
        @simd for ci = 1:c
        # Threads.@threads for ci = 1:c
            # @simd for
            # wi = 1:n_W
            for wi = 1:n_W, hi = 1:n_H
                # @simd
                # for hi = 1:n_H
                    h_start = hi * s_H - (s_H == 1 ? 0 : s_H - 1)
                    h_end = hi * s_H - (s_H == 1 ? 0 : s_H - 1) + f_H - 1
                    w_start = wi * s_W - (s_W == 1 ? 0 : s_W - 1)
                    w_end = wi * s_W - (s_W == 1 ? 0 : s_W - 1) + f_W - 1
                    @inbounds ai = view(Aip,h_start:h_end, w_start:w_end, :, mi)
                    # @inbounds ai = Aip[h_start:h_end, w_start:w_end, :, mi]
                    @inbounds Z[hi, wi, ci, mi] = W[:, :, :, ci] ⋅ ai + B[1,1,1,ci]
                # end #for hi=1:n_H
            end #for wi=1:n_W, hi=1:n_H
        end #for ci=1:c
    end #for mi=1:m
    # @inbounds Z .+= B[:, :, 1, :]
    actFun = cLayer.actFun
    Ao = eval(:($actFun($Z)))

    # Base.GC.gc()

    return Dict(:Z => Z, :A => Ao)
end #function convolve(cLayer::Conv2D

function convolve(cLayer::Conv3D, Ai::AbstractArray{T1,5}) where {T1}

    Aip = padding(cLayer, Ai)

    f_H, f_W, f_D = cLayer.f
    s_H, s_W, s_D = cLayer.s
    lastDim = ndims(Ai)
    paddedS = paddedSize(cLayer, Ai)[1:end-2]
    s = cLayer.s
    f = cLayer.f
    outputS = outDims(cLayer, Ai)[1:end-1]
    ci, m = cLayer.inputS[end], size(Ai)[end]
    co = cLayer.channels
    Z = zeros(eltype(Ai), outputS..., co, m)
    n_H, n_W, n_D, c, m = size(Z)
    W = cLayer.W
    B = cLayer.B
    # @simd
    Threads.@threads for mi = 1:m
        # @simd for
        @simd for ci = 1:c
            # @simd for

                # @simd for

                    for di = 1:n_D, wi = 1:n_W, hi = 1:n_H
                        h_start = hi * s_H - (s_H == 1 ? 0 : s_H - 1)
                        h_end = hi * s_H - (s_H == 1 ? 0 : s_H - 1) + f_H - 1
                        w_start = wi * s_W - (s_W == 1 ? 0 : s_W - 1)
                        w_end = wi * s_W - (s_W == 1 ? 0 : s_W - 1) + f_W - 1
                        d_start = di * s_D - (s_D == 1 ? 0 : s_D - 1)
                        d_end = di * s_D - (s_D == 1 ? 0 : s_D - 1) + f_D - 1
                        @inbounds ai = view(Aip,
                            h_start:h_end,
                            w_start:w_end,
                            d_start:d_end,
                            :,
                            mi,
                        )
                        @inbounds Z[hi, wi, di, ci, mi] =
                            W[:, :, :, :, ci] ⋅ ai + B[1,1,1,1,ci]
                        # ai = nothing
                        # Base.GC.gc()
                    end #for hi=1:n_H
        #         end #for wi=1:n_W
        #     end #for di=1:n_D
        end #for ci=1:c
    end #for mi=1:m
    # @inbounds Z .+= B[:, :, :, 1, :]
    actFun = cLayer.actFun
    Ao = eval(:($actFun($Z)))

    # Base.GC.gc()

    return Dict(:Z => Z, :A => Ao)
end #function convolve(cLayer::Conv3D


export convolve

### dconvolve

#Naive convolution method using for loops with some parallel using
#@simd and @inbounds

function dconvolve!(
    cLayer::Conv1D,
    Ai::AbstractArray{T1,3},
    dAi::AbstractArray{T2,3},
    dZ::AbstractArray{T3,3},
) where {T1,T2,T3}

    Aip = padding(cLayer, Ai)
    padS = paddingSize(cLayer, Ai)
    dAip = zeros(promote_type(eltype(dZ),eltype(Ai)), size(Aip))


    f_H = cLayer.f
    s_H = cLayer.s
    lastDim = ndims(Ai)
    n_H, c, m = size(dZ)
    W = cLayer.W
    cLayer.dW = similar(W)
    cLayer.dW .= 0
    B = cLayer.B
    cLayer.dB = similar(B)
    cLayer.dB .= 0
    # @simd
    Threads.@threads for mi = 1:m
        # @simd
        @simd for ci = 1:c
            for hi = 1:n_H
                h_start = hi * s_H - (s_H == 1 ? 0 : s_H - 1)
                h_end = hi * s_H - (s_H == 1 ? 0 : s_H - 1) + f_H - 1
                @inbounds ai = view(Aip, h_start:h_end, :, mi)
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
    Ai::AbstractArray{T1,4},
    dAi::AbstractArray{T2,4},
    dZ::AbstractArray{T3,4},
) where {T1,T2,T3}

    Aip = padding(cLayer, Ai)
    padS = paddingSize(cLayer, Ai)
    dAip = zeros(promote_type(eltype(dZ),eltype(Ai)), size(Aip))

    f_H, f_W = cLayer.f
    s_H, s_W = cLayer.s
    lastDim = ndims(Ai)
    n_H, n_W, c, m = size(dZ)
    W = cLayer.W
    cLayer.dW = similar(W)
    cLayer.dW .= 0
    B = cLayer.B
    cLayer.dB = similar(B)
    cLayer.dB .= 0
    # @simd
    Threads.@threads for mi = 1:m
        # @simd
        @simd for ci = 1:c
            for wi = 1:n_W, hi = 1:n_H
                h_start = hi * s_H - (s_H == 1 ? 0 : s_H - 1)
                h_end = hi * s_H - (s_H == 1 ? 0 : s_H - 1) + f_H - 1
                w_start = wi * s_W - (s_W == 1 ? 0 : s_W - 1)
                w_end = wi * s_W - (s_W == 1 ? 0 : s_W - 1) + f_W - 1
                @inbounds ai = view(Aip, h_start:h_end, w_start:w_end, :, mi)
                @inbounds dAip[h_start:h_end, w_start:w_end, :, mi] .+=
                    dZ[hi, wi, ci, mi] .* W[:, :, :, ci]

                @inbounds cLayer.dW[:, :, :, ci] .+= (ai .* dZ[hi, wi, ci, mi])
                @inbounds cLayer.dB[:, :, :, ci] .+= dZ[hi, wi, ci, mi]
            end #for
        end #for ci=1:c
    end #for mi=1:m

    dAi .= dAip[1+padS[1]:end-padS[2], 1+padS[3]:end-padS[4], :, :]

    return dAi
end #function dconvolve(cLayer::Conv2D@


function dconvolve!(
    cLayer::Conv3D,
    Ai::AbstractArray{T1,5},
    dAi::AbstractArray{T2,5},
    dZ::AbstractArray{T3,5},
) where {T1,T2,T3}

    Aip = padding(cLayer, Ai)
    padS = paddingSize(cLayer, Ai)
    dAip = zeros(promote_type(eltype(dZ),eltype(Ai)), size(Aip))

    f_H, f_W, f_D = cLayer.f
    s_H, s_W, s_D = cLayer.s
    lastDim = ndims(Ai)
    n_H, n_W, n_D, c, m = size(dZ)
    W = cLayer.W
    cLayer.dW = similar(W)
    cLayer.dW .= 0
    B = cLayer.B
    cLayer.dB = similar(B)
    cLayer.dB .= 0
    # @simd
    Threads.@threads for mi = 1:m
        # @simd
        @simd for ci = 1:c
            for di = 1:n_D, wi = 1:n_W, hi = 1:n_H
                h_start = hi * s_H - (s_H == 1 ? 0 : s_H - 1)
                h_end = hi * s_H - (s_H == 1 ? 0 : s_H - 1) + f_H - 1
                w_start = wi * s_W - (s_W == 1 ? 0 : s_W - 1)
                w_end = wi * s_W - (s_W == 1 ? 0 : s_W - 1) + f_W - 1
                d_start = di * s_D - (s_D == 1 ? 0 : s_D - 1)
                d_end = di * s_D - (s_D == 1 ? 0 : s_D - 1) + f_D - 1
                @inbounds ai =
                    view(Aip, h_start:h_end, w_start:w_end, d_start:d_end, :, mi)
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
