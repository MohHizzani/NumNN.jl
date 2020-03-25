#Naive convolution method using for loops with some parallel using
#@simd and @inbounds

function convolve(cLayer::Conv1D, Ai::AbstractArray{T,3}, Z::AbstractArray{T,3}) where {T}

    Aip = padding(cLayer, Ai)

    f_H = cLayer.f
    s_H = cLayer.s
    lastDim = ndims(Ai)
    n_H, c, m = size(Z)
    W = cLayer.W
    B = cLayer.B
    @simd for mi = 1:m
        @simd for ci = 1:c
            @simd for hi = 1:n_H
                h_start = hi * s_H - (s_H == 1 ? 0 : s_H - 1)
                h_end = hi * s_H - (s_H == 1 ? 0 : s_H - 1) + f_H - 1
                @inbounds ai = view(Aip,h_start:h_end, :, mi)
                @inbounds Z[hi, ci, mi] = W[:, :, ci] ⋅ ai
                # ai = nothing
                # Base.GC.gc()
            end #for mi=1:m, hi=1:n_H
        end #for ci=1:c
    end #for mi=1:m
    @inbounds Z .+= B[:, 1, :]
    actFun = cLayer.actFun
    Ao = eval(:($actFun($Z)))

    # Base.GC.gc()

    return Dict(:Z => Z, :A => Ao)
end #function convolve(cLayer::Conv1D


function convolve(cLayer::Conv2D, Ai::AbstractArray{T,4}, Z::AbstractArray{T,4}) where {T}

    Aip = padding(cLayer, Ai)

    f_H, f_W = cLayer.f
    s_H, s_W = cLayer.s
    lastDim = ndims(Ai)
    n_H, n_W, c, m = size(Z)
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
                    @inbounds ai = view(Aip,h_start:h_end, w_start:w_end, :, mi)
                    @inbounds Z[hi, wi, ci, mi] = W[:, :, :, ci] ⋅ ai
                end #for hi=1:n_H
            end #for mi=1:m, wi=1:n_W, hi=1:n_H
        end #for ci=1:c
    end #for mi=1:m
    @inbounds Z .+= B[:, :, 1, :]
    actFun = cLayer.actFun
    Ao = eval(:($actFun($Z)))

    # Base.GC.gc()

    return Dict(:Z => Z, :A => Ao)
end #function convolve(cLayer::Conv2D

function convolve(cLayer::Conv3D, Ai::AbstractArray{T,5}, Z::AbstractArray{T,5}) where {T}

    Aip = padding(cLayer, Ai)

    f_H, f_W, f_D = cLayer.f
    s_H, s_W, s_D = cLayer.s
    lastDim = ndims(Ai)
    n_H, n_W, n_D, c, m = size(Z)
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
                        @inbounds ai = view(Aip,
                            h_start:h_end,
                            w_start:w_end,
                            d_start:d_end,
                            :,
                            mi,
                        )
                        @inbounds Z[hi, wi, di, ci, mi] =
                            W[:, :, :, :, ci] ⋅ ai
                        # ai = nothing
                        # Base.GC.gc()
                    end #for hi=1:n_H
                end #for wi=1:n_W
            end #for di=1:n_D
        end #for ci=1:c
    end #for mi=1:m
    @inbounds Z .+= B[:, :, :, 1, :]
    actFun = cLayer.actFun
    Ao = eval(:($actFun($Z)))

    # Base.GC.gc()

    return Dict(:Z => Z, :A => Ao)
end #function convolve(cLayer::Conv3D


export convolve
