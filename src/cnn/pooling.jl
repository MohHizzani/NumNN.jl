using Statistics


function pooling!(
    cLayer::OneD,
    Ai::AbstractArray{T,3},
) where {OneD<:Union{MaxPool1D,AveragePool1D},T}

    n_Hi, c, m = size(Ai)
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
                @inbounds ai = Ai[h_start:h_end, ci, mi]
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

    n_Hi, n_Wi, c, m = size(Ai)
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
                        view(Ai, h_start:h_end, w_start:w_end, ci, mi)
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

    n_Hi, n_Wi, n_Di, c, m = size(Ai)
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
                        @inbounds ai = Ai[
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
    dA::AbstractArray{T,3},
) where {OneD<:Union{MaxPool1D,AveragePool1D},T}

    n_Hi, c, m = size(Ai)
    s_H = sS = cLayer.s
    f_H = fS = cLayer.f
    n_H = (n_Hi - f_H) ÷ s_H + 1

    if sS == fS #to use the @simd the parallele the operation
        @simd for mi = 1:m
            @simd for ci = 1:c
                @simd for hi = 1:n_H
                    h_start = hi * s_H - (s_H == 1 ? 0 : s_H - 1)
                    h_end = hi * s_H - (s_H == 1 ? 0 : s_H - 1) + f_H - 1
                    @inbounds ai = Ai[h_start:h_end, ci, mi]
                    if cLayer isa MaxPoolLayer
                        mask = (maximum(ai) .== ai)
                        @inbounds dAi[h_start:h_end, ci, mi] .+=
                            (dA[hi, ci, mi] .* mask)
                    else
                        mask = similar(ai)
                        mask .= 1 / prod(size(ai))
                        @inbounds dAi[h_start:h_end, ci, mi] .+=
                            (dA[hi, ci, mi] .* mask)
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
                    @inbounds ai = Ai[h_start:h_end, ci, mi]
                    if cLayer isa MaxPoolLayer
                        mask = (maximum(ai) .== ai)
                        @inbounds dAi[h_start:h_end, ci, mi] .+=
                            (dA[hi, ci, mi] .* mask)
                    else
                        mask = similar(ai)
                        mask .= 1 / prod(size(ai))
                        @inbounds dAi[h_start:h_end, ci, mi] .+=
                            (dA[hi, ci, mi] .* mask)
                    end #if cLayer isa MaxPoolLayer
                end #for hi=1:n_H
            end #for ci=1:c
        end #for mi=1:m
    end #if sS == fS

    return nothing
end #function dpooling!(cLayer::OneD


function dpooling!(
    cLayer::TwoD,
    Ai::AbstractArray{T,4},
    dAi::AbstractArray{T,4},
    dA::AbstractArray{T,4},
) where {TwoD<:Union{MaxPool2D,AveragePool2D},T}

    n_Hi, n_Wi, c, m = size(Ai)
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
                        @inbounds ai = Ai[h_start:h_end, w_start:w_end, ci, mi]
                        if cLayer isa MaxPoolLayer
                            mask = (maximum(ai) .== ai)
                            @inbounds dAi[
                                h_start:h_end,
                                w_start:w_end,
                                ci,
                                mi,
                            ] .+= (dA[hi, wi, ci, mi] .* mask)
                        else
                            mask = similar(ai)
                            mask .= 1 / prod(size(ai))
                            @inbounds dAi[
                                h_start:h_end,
                                w_start:w_end,
                                ci,
                                mi,
                            ] .+= (dA[hi, wi, ci, mi] .* mask)
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
                    @inbounds ai = Ai[h_start:h_end, w_start:w_end, ci, mi]
                    if cLayer isa MaxPoolLayer
                        mask = (maximum(ai) .== ai)
                        @inbounds dAi[h_start:h_end, w_start:w_end, ci, mi] .+=
                            (dA[hi, wi, ci, mi] .* mask)
                    else
                        mask = similar(ai)
                        mask .= 1 / prod(size(ai))
                        @inbounds dAi[h_start:h_end, w_start:w_end, ci, mi] .+=
                            (dA[hi, wi, ci, mi] .* mask)
                    end #if cLayer isa MaxPoolLayer
                end #for wi=1:n_W, hi=1:n_H
            end #for ci=1:c
        end #for mi=1:m
    end #if sS == fS

    return nothing
end #function dpooling!(cLayer::TwoD,


function dpooling!(
    cLayer::ThreeD,
    Ai::AbstractArray{T,5},
    dAi::AbstractArray{T,5},
    dA::AbstractArray{T,5},
) where {ThreeD<:Union{MaxPool3D,AveragePool3D},T}

    n_Hi, n_Wi, n_Di, c, m = size(Ai)
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
                            @inbounds ai = Ai[
                                h_start:h_end,
                                w_start:w_end,
                                d_start:d_end,
                                ci,
                                mi,
                            ]
                            if cLayer isa MaxPoolLayer
                                mask = (maximum(ai) .== ai)
                                @inbounds dAi[
                                    h_start:h_end,
                                    w_start:w_end,
                                    d_start:d_end,
                                    ci,
                                    mi,
                                ] .+= (dA[hi, wi, di, ci, mi] .* mask)
                            else
                                mask = similar(ai)
                                mask .= 1 / prod(size(ai))
                                @inbounds dAi[
                                    h_start:h_end,
                                    w_start:w_end,
                                    d_start:d_end,
                                    ci,
                                    mi,
                                ] .+= (dA[hi, wi, di, ci, mi] .* mask)
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
                        Ai[h_start:h_end, w_start:w_end, d_start:d_end, ci, mi]
                    if cLayer isa MaxPoolLayer
                        mask = (maximum(ai) .== ai)
                        @inbounds dAi[
                            h_start:h_end,
                            w_start:w_end,
                            d_start:d_end,
                            ci,
                            mi,
                        ] .+= (dA[hi, wi, di, ci, mi] .* mask)
                    else
                        mask = similar(ai)
                        mask .= 1 / prod(size(ai))
                        @inbounds dAi[
                            h_start:h_end,
                            w_start:w_end,
                            d_start:d_end,
                            ci,
                            mi,
                        ] .+= (dA[hi, wi, di, ci, mi] .* mask)
                    end #if cLayer isa MaxPoolLayer
                end #for wi=1:n_W, hi=1:n_H, di=1:n_D
            end #for ci=1:c
        end #for mi=1:m
    end #if sS == fS

    return nothing
end #function dpooling!(cLayer::ThreeD,

export dpooling!
