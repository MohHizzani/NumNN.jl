using Statistics


function fastPooling!(
    cLayer::OneD,
    Ai::AbstractArray{T,3},
) where {OneD<:Union{MaxPool1D,AveragePool1D},T}

    Aip = padding(cLayer, Ai)
    n_Hi, c, m = size(Aip)
    s_H = cLayer.s
    f_H = cLayer.f
    n_H = (n_Hi - f_H) ÷ s_H + 1

    if cLayer isa MaxPoolLayer
        cLayer.A = reshape(
            maximum(reshape(Aip, f_H, n_H, c, m), dims = 1),
            n_H,
            c,
            m,
        )
    else
        cLayer.A =
            reshape(mean(reshape(Aip, f_H, n_H, c, m), dims = 1), n_H, c, m)
    end #if cLayer isa MaxPoolLayer

    return nothing
end #function fastPooling!(cLayer::OneD


function fastPooling!(
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
        cLayer.A = reshape(
            maximum(
                permutedims(
                    reshape(Aip, f_H, n_H, f_W, n_W, c, m),
                    [1, 3, 2, 4, 5, 6],
                ),
                dims = 1:2,
            ),
            n_H,
            n_W,
            c,
            m,
        )
    else
        cLayer.A = reshape(
            mean(
                permutedims(
                    reshape(Aip, f_H, n_H, f_W, n_W, c, m),
                    [1, 3, 2, 4, 5, 6],
                ),
                dims = 1:2,
            ),
            n_H,
            n_W,
            c,
            m,
        )
    end #if cLayer isa MaxPoolLayer

    return nothing
end #function fastPooling!(cLayer::TwoD,


function fastPooling!(
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
        cLayer.A = reshape(
            maximum(
                permutedims(
                    reshape(Aip, f_H, n_H, f_W, n_W, f_D, n_D, c, m),
                    [1, 3, 5, 2, 4, 6, 7, 8],
                ),
                dims = 1:3,
            ),
            n_H,
            n_W,
            n_D,
            c,
            m,
        )
    else
        cLayer.A = reshape(
            mean(
                permutedims(
                    reshape(Aip, f_H, n_H, f_W, n_W, f_D, n_D, c, m),
                    [1, 3, 5, 2, 4, 6, 7, 8],
                ),
                dims = 1:3,
            ),
            n_H,
            n_W,
            n_D,
            c,
            m,
        )
    end #if cLayer isa MaxPoolLayer

    return nothing
end #function fastPooling!(cLayer::ThreeD,

export fastPooling!

### dfastPooling


function dfastPooling!(
    cLayer::OneD,
    Ai::AbstractArray{T,3},
    dAi::AbstractArray{T,3},
    Ao::AbstractArray{T,3},
    dAo::AbstractArray{T,3},
) where {OneD<:Union{MaxPool1D,AveragePool1D},T}

    padS = paddingSize(cLayer, Ai)
    Aip = padding(cLayer, Ai)
    n_Hi, c, m = size(Aip)
    s_H = cLayer.s
    f_H = cLayer.f
    n_H = (n_Hi - f_H) ÷ s_H + 1

    Aipr = reshape(Aip, f_H, n_H, c, m)
    if cLayer isa MaxPoolLayer
        mask = Aipr .== reshape(Ao, 1, n_H, c, m)
    else
        mask = similar(Aipr) .= 1 / prod((cLayer.f))
    end #if cLayer isa MaxPoolLayer
    dAi .= reshape(reshape(dAo, 1, n_H, c, m) .* mask, n_Hi, c, m)[1+padS[1]:end-padS[2], :, :]

    mask = nothing
    return nothing
end #function dfastPooling!(cLayer::OneD


function dfastPooling!(
    cLayer::TwoD,
    Ai::AbstractArray{T,4},
    dAi::AbstractArray{T,4},
    Ao::AbstractArray{T,4},
    dAo::AbstractArray{T,4},
) where {TwoD<:Union{MaxPool2D,AveragePool2D},T}

    padS = paddingSize(cLayer, Ai)
    Aip = padding(cLayer, Ai)
    n_Hi, n_Wi, c, m = size(Aip)
    s_H, s_W = cLayer.s
    f_H, f_W = cLayer.f
    n_H = (n_Hi - f_H) ÷ s_H + 1
    n_W = (n_Wi - f_W) ÷ s_W + 1

    Aipr =
        permutedims(reshape(Aip, f_H, n_H, f_W, n_W, c, m), [1, 3, 2, 4, 5, 6])

    if cLayer isa MaxPoolLayer
        mask = Aipr .== reshape(Ao, 1, 1, n_H, n_W, c, m)
    else
        mask = similar(Aipr) .= 1 / prod((cLayer.f))
    end
    dAi .= reshape(
        permutedims(
            reshape(dAo, 1, 1, n_H, n_W, c, m) .* mask,
            [1, 3, 2, 4, 5, 6],
        ),
        n_Hi,
        n_Wi,
        c,
        m,
    )[1+padS[1]:end-padS[2], 1+padS[3]:end-padS[4], :, :]

    mask = nothing
    return nothing
end #function dfastPooling!(cLayer::TwoD,


function dfastPooling!(
    cLayer::ThreeD,
    Ai::AbstractArray{T,5},
    dAi::AbstractArray{T,5},
    Ao::AbstractArray{T,5},
    dAo::AbstractArray{T,5},
) where {ThreeD<:Union{MaxPool3D,AveragePool3D},T}

    padS = paddingSize(cLayer, Ai)
    Aip = padding(cLayer, Ai)
    n_Hi, n_Wi, n_Di, c, m = size(Aip)
    s_H, s_W, s_D = cLayer.s
    f_H, f_W, f_D = cLayer.f

    n_H = (n_Hi - f_H) ÷ s_H + 1
    n_W = (n_Wi - f_W) ÷ s_W + 1
    n_D = (n_Di - f_D) ÷ s_D + 1

    Aipr = permutedims(
        reshape(Aip, f_H, n_H, f_W, n_W, f_D, n_D, c, m),
        [1, 3, 5, 2, 4, 6, 7, 8],
    )

    if cLayer isa MaxPoolLayer
        mask = Aipr .== reshape(Ao, 1, 1, 1, n_H, n_W, n_D, c, m)
    else
        mask = similar(Aipr) .= 1 / prod((cLayer.f))
    end

    dAi .= reshape(
        permutedims(
            reshape(dAo, 1, 1, 1, n_H, n_W, n_D, c, m) .* mask,
            [1, 4, 2, 5, 3, 6, 7, 8],
        ),
        n_Hi,
        n_Wi,
        n_Di,
        c,
        m,
    )[1+padS[1]:end-padS[2],
        1+padS[3]:end-padS[4],
        1+padS[5]:end-padS[6],
        :,
        :,
    ]

    mask = nothing
    return nothing
end #function dfastPooling!(cLayer::ThreeD,

export dfastPooling!
