
function convolve!(cLayer::Conv1D,
                  Ai)

    f_H = cLayer.f
    s_H = cLayer.s
    lastDim = ndims(Ai)
    n_H, c, m = size(cLayer.Z)
    W = cLayer.W
    B = cLayer.B
    @simd for mi=1:m
        @simd for ci=1:c
            @simd for hi=1:n_H
                h_start = hi* s_H - (s_H == 1 ? 0 : s_H-1)
                h_end = hi*s_H - (s_H == 1 ? 0 : s_H-1) + f_H -1
                @inbounds ai = Ai[h_start:h_end, :, mi]
                @inbounds cLayer.Z[hi, ci, mi] = W[:,:,ci] ⋅ ai
                # ai = nothing
                # Base.GC.gc()
            end #for mi=1:m, hi=1:n_H
        end #for ci=1:c
    end #for mi=1:m
    @inbounds cLayer.Z .+= B[:,1,:]
    Z = cLayer.Z
    actFun = cLayer.actFun
    cLayer.A = eval(:($actFun($Z)))

    W = B = Z = nothing

    Ai = nothing

    # Base.GC.gc()

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
    @simd for mi=1:m
      @simd for ci=1:c
          @simd for wi=1:n_W
              @simd for hi=1:n_H
                  h_start = hi* s_H - (s_H == 1 ? 0 : s_H-1)
                  h_end = hi*s_H - (s_H == 1 ? 0 : s_H-1) + f_H -1
                  w_start = wi*s_W - (s_W == 1 ? 0 : s_W-1)
                  w_end = wi*s_W - (s_W == 1 ? 0 : s_W-1) + f_W -1
                  @inbounds ai = Ai[h_start:h_end, w_start:w_end, :, mi]
                  @inbounds cLayer.Z[hi, wi, ci, mi] = W[:,:,:,ci] ⋅ ai
              end #for hi=1:n_H
          end #for mi=1:m, wi=1:n_W, hi=1:n_H
      end #for ci=1:c
    end #for mi=1:m
    @inbounds cLayer.Z .+= B[:,:,1, :]
    Z = cLayer.Z
    actFun = cLayer.actFun
    cLayer.A = eval(:($actFun($Z)))



    # Base.GC.gc()

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
    @simd for mi=1:m
        @simd for ci=1:c
            @simd for di=1:n_D
                @simd for wi=1:n_W
                    @simd for hi=1:n_H
                        h_start = hi* s_H - (s_H == 1 ? 0 : s_H-1)
                        h_end = hi*s_H - (s_H == 1 ? 0 : s_H-1) + f_H -1
                        w_start = wi*s_W - (s_W == 1 ? 0 : s_W-1)
                        w_end = wi*s_W - (s_W == 1 ? 0 : s_W-1) + f_W -1
                        d_start = di*s_D - (s_D == 1 ? 0 : s_D-1)
                        d_end = di*s_D - (s_D == 1 ? 0 : s_D-1) + f_D -1
                        @inbounds ai = Ai[h_start:h_end, w_start:w_end, d_start:d_end, :, mi]
                        @inbounds cLayer.Z[hi, wi, di, ci, mi] = W[:,:,:,:,ci] ⋅ ai
                        # ai = nothing
                        # Base.GC.gc()
                    end #for hi=1:n_H
                end #for wi=1:n_W
            end #for di=1:n_D
        end #for ci=1:c
    end #for mi=1:m
    @inbounds cLayer.Z .+= B[:,:,:,1, :]
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


function dconvolve!(
                   cLayer::Conv1D,
                   Ai::AbstractArray,
                   dA::AbstractArray,
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
    @simd for mi=1:m
        @simd for ci=1:c
            for hi=1:n_H
                h_start = hi* s_H - (s_H == 1 ? 0 : s_H-1)
                h_end = hi*s_H - (s_H == 1 ? 0 : s_H-1) + f_H -1
                @inbounds ai = Ai[h_start:h_end, :, mi]
                @inbounds dA[h_start:h_end, :, mi] .+= dZ[hi, ci, mi] .* W[:,:,ci]

                @inbounds cLayer.dW[:,:,ci] .+= (ai .* dZ[hi, ci, mi])
                @inbounds cLayer.dB[:,:,ci] .+= dZ[hi,ci,mi]
            end #for
        end #for ci=1:c
    end #for mi=1:m
    # n_Hi, ci, m = cLayer.inputS
    # n_Hj, ci, m = size(dA)
    # p_H_hi, p_H_lo = paddingSize(cLayer,Ai)
    #
    # cLayer.dA = dA[1+p_H_hi:end-p_H_lo,:,:]
    #
    # @assert cLayer.inputS == size(cLayer.dA)

    return dA
end #function dconvolve(cLayer::Conv1D


function dconvolve!(
                    cLayer::Conv2D,
                    Ai::AbstractArray,
                    dA::AbstractArray,
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
    @simd for mi=1:m
        @simd for ci=1:c
            for wi=1:n_W, hi=1:n_H
                h_start = hi* s_H - (s_H == 1 ? 0 : s_H-1)
                h_end = hi*s_H - (s_H == 1 ? 0 : s_H-1) + f_H -1
                w_start = wi*s_W - (s_W == 1 ? 0 : s_W-1)
                w_end = wi*s_W - (s_W == 1 ? 0 : s_W-1) + f_W -1
                @inbounds ai = Ai[h_start:h_end, w_start:w_end, :, mi]
                @inbounds dA[h_start:h_end, w_start:w_end, :, mi] .+= dZ[hi, wi, ci, mi] .* W[:,:,:,ci]

                @inbounds cLayer.dW[:,:,:,ci] .+= (ai .* dZ[hi, wi, ci, mi])
                @inbounds cLayer.dB[:,:,:,ci] .+= dZ[hi, wi,ci,mi]
            end #for
        end #for ci=1:c
    end #for mi=1:m
    # n_Hi, n_Wi, ci, m = cLayer.inputS
    # n_Hj, n_Wj, ci, m = size(dA)
    # p_H_hi, p_H_lo, p_W_hi, p_W_lo = paddingSize(cLayer, Ai)
    #
    # cLayer.dA = dA[1+p_H_hi:end-p_H,1+p_W:end-p_W,:,:]
    #
    # @assert cLayer.inputS == size(cLayer.dA)

    return dA
end #function dconvolve(cLayer::Conv2D


function dconvolve!(
                    cLayer::Conv3D,
                    Ai::AbstractArray,
                    dA::AbstractArray,
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
    @simd for mi=1:m
        @simd for ci=1:c
            for wi=1:n_W, hi=1:n_H, di=1:n_D
                h_start = hi* s_H - (s_H == 1 ? 0 : s_H-1)
                h_end = hi*s_H - (s_H == 1 ? 0 : s_H-1) + f_H -1
                w_start = wi*s_W - (s_W == 1 ? 0 : s_W-1)
                w_end = wi*s_W - (s_W == 1 ? 0 : s_W-1) + f_W -1
                d_start = di*s_D - (s_D == 1 ? 0 : s_D-1)
                d_end = di*s_D - (s_D == 1 ? 0 : s_D-1) + f_D -1
                @inbounds ai = Ai[h_start:h_end, w_start:w_end, d_start:d_end, :, mi]
                @inbounds dA[h_start:h_end, w_start:w_end, d_start:d_end, :, mi] .+= dZ[hi, wi, di, ci, mi] .* W[:,:,:,:,ci]

                @inbounds cLayer.dW[:,:,:,:,ci] .+= (ai .* dZ[hi, wi, ci, mi])
                @inbounds cLayer.dB[:,:,:,:,ci] .+= dZ[hi,wi,di,ci,mi]
            end #for
        end #for ci=1:c
    end #for mi=1:m

    # n_Hi, n_Wi, n_Di, ci, m = cLayer.inputS
    # n_Hj, n_Wj, n_Dj, ci, m = size(dA)
    # p_H = abs(n_Hi - n_Hj) ÷ 2
    # p_W = abs(n_Wi - n_Wj) ÷ 2
    # p_D = abs(n_Di - n_Dj) ÷ 2
    #
    # cLayer.dA = dA[1+p_H:end-p_H,1+p_W:end-p_W,1+p_D:end-p_D,:,:]
    #
    # @assert cLayer.inputS == size(cLayer.dA)

    return nothing
end #function dconvolve(cLayer::Conv3D


export dconvolve!
