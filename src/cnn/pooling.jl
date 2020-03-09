using Statistics


function pooling!(cLayer::OneD,
                 Ai) where {OneD <: Union{MaxPool1D, AveragePool1D}}

    n_Hi, c, m = size(Ai)
    s_H = cLayer.s
    f_H = cLayer.f
    n_H = (n_Hi - f_H) ÷ s_H + 1

    for mi=1:m, ci=1:c
        @async for hi=1:n_H
            h_start = hi*s_H - (s_H == 1 ? 0 : 1)
            h_end = hi*s_H - (s_H == 1 ? 0 : 1) + f_H -1
            ai = Ai[h_start:h_end, ci, mi]
            if cLayer isa MaxPoolLayer
                pool = maximum
            else
                pool = mean
            end #if cLayer isa MaxPoolLayer
            cLayer.A[hi, ci, mi] = pool(ai)
        end #for
    end #for mi=1:m, ci=1:c

    return nothing
end #function pooling!(cLayer::OneD


function pooling!(cLayer::TwoD,
                  Ai) where {TwoD <: Union{MaxPool2D, AveragePool2D}}

    n_Hi, n_Wi, c, m = size(Ai)
    s_H, s_W = cLayer.s
    f_H, f_W = cLayer.f
    n_H = (n_Hi - f_H) ÷ s_H + 1
    n_W = (n_Wi - f_W) ÷ s_W + 1
    for mi=1:m, ci=1:c
        @async for wi=1:n_W, hi=1:n_H
            h_start = hi* s_H - (s_H == 1 ? 0 : 1)
            h_end = hi*s_H - (s_H == 1 ? 0 : 1) + f_H -1
            w_start = wi*s_W - (s_W == 1 ? 0 : 1)
            w_end = wi*s_W - (s_W == 1 ? 0 : 1) + f_W -1
            ai = Ai[h_start:h_end, w_start:w_end, ci, mi]
            if cLayer isa MaxPoolLayer
                pool = maximum
            else
                pool = mean
            end #if cLayer isa MaxPoolLayer
            cLayer.A[hi, wi, ci, mi] = pool(ai)
        end #for
    end #for mi=1:m, ci=1:c
    return nothing
end #function pooling!(cLayer::TwoD,


function pooling!(cLayer::ThreeD,
                  Ai) where {ThreeD <: Union{MaxPool3D, AveragePool3D}}

    n_Hi, n_Wi, n_Di, c, m = size(Ai)
    s_H, s_W, s_D = cLayer.s
    f_H, f_W, f_D = cLayer.f

    n_H = (n_Hi - f_H) ÷ s_H + 1
    n_W = (n_Wi - f_W) ÷ s_W + 1
    n_D = (n_Di - f_D) ÷ s_D + 1

    for mi=1:m, ci=1:c
        @async for wi=1:n_W, hi=1:n_H, di=1:n_D
            h_start = hi*s_H - (s_H == 1 ? 0 : 1)
            h_end = hi*s_H - (s_H == 1 ? 0 : 1) + f_H -1
            w_start = wi*s_W - (s_W == 1 ? 0 : 1)
            w_end = wi*s_W - (s_W == 1 ? 0 : 1) + f_W -1
            d_start = di*s_D - (s_D == 1 ? 0 : 1)
            d_end = di*s_D - (s_D == 1 ? 0 : 1) + f_D -1
            ai = Ai[h_start:h_end, w_start:w_end, d_start:d_end, ci, mi]
            if cLayer isa MaxPoolLayer
                pool = maximum
            else
                pool = mean
            end #if cLayer isa MaxPoolLayer
            cLayer.A[hi, wi, di, ci, mi] = pool(ai)
        end #for
    end #for mi=1:m, ci=1:c
    return nothing
end #function pooling!(cLayer::ThreeD,

export pooling!


###dpooling

function dpooling!(
                   cLayer::OneD,
                   Ai,
                   dAi,
                   ) where {OneD <: Union{MaxPool1D, AveragePool1D}}

     n_Hi, c, m = size(Ai)
     s_H = cLayer.s
     f_H = cLayer.f
     n_H = (n_Hi - f_H) ÷ s_H + 1

    for mi=1:m, ci=1:c, hi=1:n_H
        h_start = hi*s_H - (s_H == 1 ? 0 : 1)
        h_end = hi*s_H - (s_H == 1 ? 0 : 1) + f_H -1
        ai = Ai[h_start:h_end, ci, mi]
        if cLayer isa MaxPoolLayer
            mask = (maximum(ai) .== ai)
            dAi[h_start:h_end, ci, mi] .+= (dAi[hi, ci, mi] .* mask)
        else
            mask = similar(ai)
            mask .= 1/prod(size(ai))
            dAi[h_start:h_end, ci, mi] .+= (dAi[hi, ci, mi] .* mask)
        end #if cLayer isa MaxPoolLayer
    end #for

    n_Hi, ci, m = size(cLayer.prevLayer.A)
    n_Hj, ci, m = size(dAi)
    p_H = abs(n_Hi - n_Hj) ÷ 2

    cLayer.dA = dAi[1+p_H:end-p_H,:,:]

    @assert size(cLayer.prevLayer.A) == size(cLayer.dA)

    return nothing
end #function dpooling!(cLayer::OneD


function dpooling!(
                   cLayer::TwoD,
                   Ai,
                   dAi,
                   ) where {TwoD <: Union{MaxPool2D, AveragePool2D}}

    n_Hi, n_Wi, c, m = size(Ai)
    s_H, s_W = cLayer.s
    f_H, f_W = cLayer.f
    n_H = (n_Hi - f_H) ÷ s_H + 1
    n_W = (n_Wi - f_W) ÷ s_W + 1

    for mi=1:m, ci=1:c, wi=1:n_W, hi=1:n_H
        h_start = hi* s_H - (s_H == 1 ? 0 : 1)
        h_end = hi*s_H - (s_H == 1 ? 0 : 1) + f_H -1
        w_start = wi*s_W - (s_W == 1 ? 0 : 1)
        w_end = wi*s_W - (s_W == 1 ? 0 : 1) + f_W -1
        ai = Ai[h_start:h_end, w_start:w_end, ci, mi]
        if cLayer isa MaxPoolLayer
            mask = (maximum(ai) .== ai)
            dAi[h_start:h_end, w_start:w_end, ci, mi] .+= (dAi[hi, wi, ci, mi] .* mask)
        else
            mask = similar(ai)
            mask .= 1/prod(size(ai))
            dAi[h_start:h_end, w_start:w_end, ci, mi] .+= (dAi[hi, wi, ci, mi] .* mask)
        end #if cLayer isa MaxPoolLayer
    end #for

    n_Hi, n_Wi, ci, m = size(cLayer.prevLayer.A)
    n_Hj, n_Wj, ci, m = size(dAi)
    p_H = abs(n_Hi - n_Hj) ÷ 2
    p_W = abs(n_Wi - n_Wj) ÷ 2

    cLayer.dA = dAi[1+p_H:end-p_H,1+p_W:end-p_W,:,:]

    @assert size(cLayer.prevLayer.A) == size(cLayer.dA)

    return nothing
end #function dpooling!(cLayer::TwoD,


function dpooling!(
                   cLayer::ThreeD,
                   Ai,
                   dAi,
                   ) where {ThreeD <: Union{MaxPool3D, AveragePool3D}}

    n_Hi, n_Wi, n_Di, c, m = size(Ai)
    s_H, s_W, s_D = cLayer.s
    f_H, f_W, f_D = cLayer.f

    n_H = (n_Hi - f_H) ÷ s_H + 1
    n_W = (n_Wi - f_W) ÷ s_W + 1
    n_D = (n_Di - f_D) ÷ s_D + 1

    for mi=1:m, ci=1:c, wi=1:n_W, hi=1:n_H, di=1:n_D
        h_start = hi*s_H - (s_H == 1 ? 0 : 1)
        h_end = hi*s_H - (s_H == 1 ? 0 : 1) + f_H -1
        w_start = wi*s_W - (s_W == 1 ? 0 : 1)
        w_end = wi*s_W - (s_W == 1 ? 0 : 1) + f_W -1
        d_start = di*s_D - (s_D == 1 ? 0 : 1)
        d_end = di*s_D - (s_D == 1 ? 0 : 1) + f_D -1
        ai = Ai[h_start:h_end, w_start:w_end, d_start:d_end, ci, mi]
        if cLayer isa MaxPoolLayer
            mask = (maximum(ai) .== ai)
            dAi[h_start:h_end, w_start:w_end, d_start:d_end, ci, mi] .+= (dAi[hi, wi, di, ci, mi] .* mask)
        else
            mask = similar(ai)
            mask .= 1/prod(size(ai))
            dAi[h_start:h_end, w_start:w_end, d_start:d_end, ci, mi] .+= (dAi[hi, wi, di, ci, mi] .* mask)
        end #if cLayer isa MaxPoolLayer
    end #for


    n_Hi, n_Wi, n_Di, ci, m = size(cLayer.prevLayer.A)
    n_Hj, n_Wj, n_Dj, ci, m = size(dAi)
    p_H = abs(n_Hi - n_Hj) ÷ 2
    p_W = abs(n_Wi - n_Wj) ÷ 2
    p_D = abs(n_Di - n_Dj) ÷ 2

    cLayer.dA = dAi[1+p_H:end-p_H,1+p_W:end-p_W,p_D:end-p_D,:,:]

    @assert size(cLayer.prevLayer.A) == size(cLayer.dA)


    return nothing
end #function dpooling!(cLayer::ThreeD,

export dpooling!
