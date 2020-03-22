using Statistics


function pooling!(
    cLayer::OneD,
    Ai::AbstractArray{T,3},
) where {OneD<:Union{MaxPool1D,AveragePool1D},T}

    Aip = padding(cLayer, Ai)
    n_Hi, c, m = size(Aip)
    s_H = cLayer.s
    f_H = cLayer.f
    n_H = (n_Hi - f_H) ÷ s_H + 1

    if cLayer isa MaxPoolLayer
        pool = maximum
    else
        pool = mean
    end #if cLayer isa MaxPoolLayer

    @simd for mi = 1:m
        @simd for ci = 1:c
            @simd for hi = 1:n_H
                h_start = hi * s_H - (s_H == 1 ? 0 : s_H - 1)
                h_end = hi * s_H - (s_H == 1 ? 0 : s_H - 1) + f_H - 1
                @inbounds ai = Aip[h_start:h_end, ci, mi]
                @inbounds cLayer.A[hi, ci, mi] = pool(ai)
                # ai = nothing
                # Base.GC.gc()
            end #for
        end #for ci=1:c
    end #for mi=1:m, ci=1:c

    return nothing
end #function pooling!(cLayer::OneD


function pooling!(
    cLayer::TwoD,
    Ai::AbstractArray{T,4},
) where {TwoD<:Union{MaxPool2D,AveragePool2D},T}

    Aip = padding(cLayer, Ai)
    n_Hi, n_Wi, c, m = size(Aip)
    s_H, s_W = cLayer.s
    f_H, f_W = cLayer.f
    n_H = (n_Hi - f_H) ÷ s_H + 1
    n_W = (n_Wi - f_W) ÷ s_W + 1
    if cLayer isa MaxPoolLayer
        pool = maximum
    else
        pool = mean
    end #if cLayer isa MaxPoolLayer
    @simd for mi = 1:m
        @simd for ci = 1:c
            @simd for wi = 1:n_W
                @simd for hi = 1:n_H
                    h_start = hi * s_H - (s_H == 1 ? 0 : s_H - 1)
                    h_end = hi * s_H - (s_H == 1 ? 0 : s_H - 1) + f_H - 1
                    w_start = wi * s_W - (s_W == 1 ? 0 : s_W - 1)
                    w_end = wi * s_W - (s_W == 1 ? 0 : s_W - 1) + f_W - 1
                    @inbounds ai =
                        view(Aip, h_start:h_end, w_start:w_end, ci, mi)
                    @inbounds cLayer.A[hi, wi, ci, mi] = pool(ai)
                    # ai = nothing
                    # Base.GC.gc()
                end #for hi=1:n_H
            end #for wi=1:n_W
        end #for ci=1:c
    end #for mi=1:m,
    return nothing
end #function pooling!(cLayer::TwoD,


function pooling!(
    cLayer::ThreeD,
    Ai::AbstractArray{T,5},
) where {ThreeD<:Union{MaxPool3D,AveragePool3D},T}

    Aip = padding(cLayer, Ai)
    n_Hi, n_Wi, n_Di, c, m = size(Aip)
    s_H, s_W, s_D = cLayer.s
    f_H, f_W, f_D = cLayer.f

    n_H = (n_Hi - f_H) ÷ s_H + 1
    n_W = (n_Wi - f_W) ÷ s_W + 1
    n_D = (n_Di - f_D) ÷ s_D + 1

    if cLayer isa MaxPoolLayer
        pool = maximum
    else
        pool = mean
    end #if cLayer isa MaxPoolLayer

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
                            ci,
                            mi,
                        ]
                        @inbounds cLayer.A[hi, wi, di, ci, mi] = pool(ai)
                        # ai = nothing
                        # Base.GC.gc()
                    end #for hi=1:n_H
                end #for wi=1:n_W
            end #for di=1:n_D
        end #for ci=1:c
    end #for mi=1:m, ci=1:c

    return nothing
end #function pooling!(cLayer::ThreeD,

export pooling!


###dpooling

#TODO to edit the dpooling! to be as Ai, dAi, Ao, dAo

function dpooling!(
    cLayer::OneD,
    Ai::AbstractArray{T,3},
    dAi::AbstractArray{T,3},
    Ao::AbstractArray{T,3},
    dAo::AbstractArray{T,3},
) where {OneD<:Union{MaxPool1D,AveragePool1D},T}

    padS = paddingSize(cLayer, Ai)
    paddedS = paddedSize(cLayer, Ai)
    dAip = zeros(eltype(Ai), paddedS)
    n_Hi, c, m = paddedS
    s_H = sS = cLayer.s
    f_H = fS = cLayer.f
    n_H = (n_Hi - f_H) ÷ s_H + 1

    if sS == fS #to use the @simd the parallele the operation
        @simd for mi = 1:m
            @simd for ci = 1:c
                @simd for hi = 1:n_H
                    h_start = hi * s_H - (s_H == 1 ? 0 : s_H - 1)
                    h_end = hi * s_H - (s_H == 1 ? 0 : s_H - 1) + f_H - 1
                    @inbounds ai = Aip[h_start:h_end, ci, mi]
                    if cLayer isa MaxPoolLayer
                        mask = (Ao[hi,ci,mi] .== ai)
                        @inbounds dAip[h_start:h_end, ci, mi] .+=
                            (dAo[hi, ci, mi] .* mask)
                    else
                        mask = similar(ai)
                        mask .= 1 / prod(size(ai))
                        @inbounds dAip[h_start:h_end, ci, mi] .+=
                            (dAo[hi, ci, mi] .* mask)
                    end #if cLayer isa MaxPoolLayer
                end #for hi=1:n_H
            end #for ci=1:c
        end #for mi=1:m
    else
        @simd for mi = 1:m
            @simd for ci = 1:c
                for hi = 1:n_H
                    h_start = hi * s_H - (s_H == 1 ? 0 : s_H - 1)
                    h_end = hi * s_H - (s_H == 1 ? 0 : s_H - 1) + f_H - 1
                    @inbounds ai = Aip[h_start:h_end, ci, mi]
                    if cLayer isa MaxPoolLayer
                        mask = (Ao[hi,ci,mi] .== ai)
                        @inbounds dAip[h_start:h_end, ci, mi] .+=
                            (dAo[hi, ci, mi] .* mask)
                    else
                        mask = similar(ai)
                        mask .= 1 / prod(size(ai))
                        @inbounds dAip[h_start:h_end, ci, mi] .+=
                            (dAo[hi, ci, mi] .* mask)
                    end #if cLayer isa MaxPoolLayer
                end #for hi=1:n_H
            end #for ci=1:c
        end #for mi=1:m
    end #if sS == fS

    dAi .= dAip[1+padS[1]:end-padS[2], :, :]

    return nothing
end #function dpooling!(cLayer::OneD


function dpooling!(
    cLayer::TwoD,
    Ai::AbstractArray{T,4},
    dAi::AbstractArray{T,4},
    Ao::AbstractArray{T,4},
    dAo::AbstractArray{T,4},
) where {TwoD<:Union{MaxPool2D,AveragePool2D},T}

    padS = paddingSize(cLayer, Ai)
    paddedS = paddedSize(cLayer, Ai)
    dAip = zeros(eltype(Ai), paddedS)
    n_Hi, n_Wi, c, m = paddedS
    s_H, s_W = sS = cLayer.s
    f_H, f_W = fS = cLayer.f
    n_H = (n_Hi - f_H) ÷ s_H + 1
    n_W = (n_Wi - f_W) ÷ s_W + 1
    if sS == fS
        @simd for mi = 1:m
            @simd for ci = 1:c
                @simd for wi = 1:n_W
                    @simd for hi = 1:n_H
                        h_start = hi * s_H - (s_H == 1 ? 0 : s_H - 1)
                        h_end = hi * s_H - (s_H == 1 ? 0 : s_H - 1) + f_H - 1
                        w_start = wi * s_W - (s_W == 1 ? 0 : s_W - 1)
                        w_end = wi * s_W - (s_W == 1 ? 0 : s_W - 1) + f_W - 1
                        @inbounds ai = Aip[h_start:h_end, w_start:w_end, ci, mi]
                        if cLayer isa MaxPoolLayer
                            mask = (Ao[hi,wi,ci,mi] .== ai)
                            @inbounds dAip[
                                h_start:h_end,
                                w_start:w_end,
                                ci,
                                mi,
                            ] .+= (dAo[hi, wi, ci, mi] .* mask)
                        else
                            mask = similar(ai)
                            mask .= 1 / prod(size(ai))
                            @inbounds dAip[
                                h_start:h_end,
                                w_start:w_end,
                                ci,
                                mi,
                            ] .+= (dAo[hi, wi, ci, mi] .* mask)
                        end #if cLayer isa MaxPoolLayer
                    end #for hi=1:n_H
                end #for wi=1:n_W
            end #for ci=1:c
        end #for mi=1:m
    else
        @simd for mi = 1:m
            @simd for ci = 1:c
                for wi = 1:n_W, hi = 1:n_H
                    h_start = hi * s_H - (s_H == 1 ? 0 : s_H - 1)
                    h_end = hi * s_H - (s_H == 1 ? 0 : s_H - 1) + f_H - 1
                    w_start = wi * s_W - (s_W == 1 ? 0 : s_W - 1)
                    w_end = wi * s_W - (s_W == 1 ? 0 : s_W - 1) + f_W - 1
                    @inbounds ai = Aip[h_start:h_end, w_start:w_end, ci, mi]
                    if cLayer isa MaxPoolLayer
                        mask = (Ao[hi,wi,ci,mi] .== ai)
                        @inbounds dAip[h_start:h_end, w_start:w_end, ci, mi] .+=
                            (dAo[hi, wi, ci, mi] .* mask)
                    else
                        mask = similar(ai)
                        mask .= 1 / prod(size(ai))
                        @inbounds dAip[h_start:h_end, w_start:w_end, ci, mi] .+=
                            (dAo[hi, wi, ci, mi] .* mask)
                    end #if cLayer isa MaxPoolLayer
                end #for wi=1:n_W, hi=1:n_H
            end #for ci=1:c
        end #for mi=1:m
    end #if sS == fS

    dAi .= dAip[1+padS[1]:end-padS[2], 1+padS[3]:end-padS[4], :, :]

    return nothing
end #function dpooling!(cLayer::TwoD,


function dpooling!(
    cLayer::ThreeD,
    Ai::AbstractArray{T,5},
    dAi::AbstractArray{T,5},
    Ao::AbstractArray{T,5},
    dAo::AbstractArray{T,5},
) where {ThreeD<:Union{MaxPool3D,AveragePool3D},T}

    padS = paddingSize(cLayer, Ai)
    paddedS = paddedSize(cLayer, Ai)
    dAip = zeros(eltype(Ai), paddedS)
    n_Hi, n_Wi, n_Di, c, m = paddedS
    s_H, s_W, s_D = sS = cLayer.s
    f_H, f_W, f_D = fS = cLayer.f

    n_H = (n_Hi - f_H) ÷ s_H + 1
    n_W = (n_Wi - f_W) ÷ s_W + 1
    n_D = (n_Di - f_D) ÷ s_D + 1

    if sS == fS
        @simd for mi = 1:m
            @simd for ci = 1:c
                @simd for di = 1:n_D
                    @simd for wi = 1:n_W
                        @simd for hi = 1:n_H
                            h_start = hi * s_H - (s_H == 1 ? 0 : s_H - 1)
                            h_end =
                                hi * s_H - (s_H == 1 ? 0 : s_H - 1) + f_H - 1
                            w_start = wi * s_W - (s_W == 1 ? 0 : s_W - 1)
                            w_end =
                                wi * s_W - (s_W == 1 ? 0 : s_W - 1) + f_W - 1
                            d_start = di * s_D - (s_D == 1 ? 0 : s_D - 1)
                            d_end =
                                di * s_D - (s_D == 1 ? 0 : s_D - 1) + f_D - 1
                            @inbounds ai = Aip[
                                h_start:h_end,
                                w_start:w_end,
                                d_start:d_end,
                                ci,
                                mi,
                            ]
                            if cLayer isa MaxPoolLayer
                                mask = (Ao[hi,wi,di,ci,mi] .== ai)
                                @inbounds dAip[
                                    h_start:h_end,
                                    w_start:w_end,
                                    d_start:d_end,
                                    ci,
                                    mi,
                                ] .+= (dAo[hi, wi, di, ci, mi] .* mask)
                            else
                                mask = similar(ai)
                                mask .= 1 / prod(size(ai))
                                @inbounds dAip[
                                    h_start:h_end,
                                    w_start:w_end,
                                    d_start:d_end,
                                    ci,
                                    mi,
                                ] .+= (dAo[hi, wi, di, ci, mi] .* mask)
                            end #if cLayer isa MaxPoolLayer
                        end #for hi=1:n_H
                    end #for wi=1:n_W
                end #for di=1:n_D
            end #for ci=1:c
        end #for mi=1:m

    else
        @simd for mi = 1:m
            @simd for ci = 1:c
                for di = 1:n_D, wi = 1:n_W, hi = 1:n_H
                    h_start = hi * s_H - (s_H == 1 ? 0 : s_H - 1)
                    h_end = hi * s_H - (s_H == 1 ? 0 : s_H - 1) + f_H - 1
                    w_start = wi * s_W - (s_W == 1 ? 0 : s_W - 1)
                    w_end = wi * s_W - (s_W == 1 ? 0 : s_W - 1) + f_W - 1
                    d_start = di * s_D - (s_D == 1 ? 0 : s_D - 1)
                    d_end = di * s_D - (s_D == 1 ? 0 : s_D - 1) + f_D - 1
                    @inbounds ai =
                        Aip[h_start:h_end, w_start:w_end, d_start:d_end, ci, mi]
                    if cLayer isa MaxPoolLayer
                        mask = (Ao[hi,wi,di,ci,mi] .== ai)
                        @inbounds dAip[
                            h_start:h_end,
                            w_start:w_end,
                            d_start:d_end,
                            ci,
                            mi,
                        ] .+= (dAo[hi, wi, di, ci, mi] .* mask)
                    else
                        mask = similar(ai)
                        mask .= 1 / prod(size(ai))
                        @inbounds dAip[
                            h_start:h_end,
                            w_start:w_end,
                            d_start:d_end,
                            ci,
                            mi,
                        ] .+= (dAo[hi, wi, di, ci, mi] .* mask)
                    end #if cLayer isa MaxPoolLayer
                end #for wi=1:n_W, hi=1:n_H, di=1:n_D
            end #for ci=1:c
        end #for mi=1:m
    end #if sS == fS

    dAi .= dAip[1+padS[1]:end-padS[2],
        1+padS[3]:end-padS[4],
        1+padS[5]:end-padS[6],
        :,
        :,
    ]

    return nothing
end #function dpooling!(cLayer::ThreeD,

export dpooling!
