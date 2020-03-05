

###convolution layers forprop

function layerForProp!(cLayer::Conv1D)
    Ai = padding(cLayer)
    n_Hi, ci, m = size(Ai)
    s_H = cLayer.s
    f_H = cLayer.f
    n_H = (n_Hi - f_H) ÷ s_H + 1

    cLayer.Z = zeros(eltype(cLayer.prevLayer.A),
                     n_H, cLayer.channels, m)
    convolve!(cLayer, Ai)

    cLayer.forwCount += 1
    return nothing

end #function layerForProp!(cLayer::Conv1D)

function layerForProp!(cLayer::Conv2D)
    Ai = padding(cLayer)
    n_Hi, n_Wi, ci, m = size(Ai)
    s_H, s_W = cLayer.s
    f_H, f_W = cLayer.f
    n_H = (n_Hi - f_H) ÷ s_H + 1
    n_W = (n_Wi - f_W) ÷ s_W + 1

    cLayer.Z = zeros(eltype(cLayer.prevLayer.A),
                     n_H, n_W, cLayer.channels, m)
    convolve!(cLayer, Ai)

    cLayer.forwCount += 1
    return nothing
end #function layerForProp!(cLayer::Conv2D)

function layerForProp!(cLayer::Conv3D)
    Ai = padding(cLayer)

    n_Hi, n_Wi, n_Di, ci, m = size(Ai)
    s_H, s_W, s_D = cLayer.s
    f_H, f_W, f_D = cLayer.f

    n_H = (n_Hi - f_H) ÷ s_H + 1
    n_W = (n_Wi - f_W) ÷ s_W + 1
    n_D = (n_Di - f_D) ÷ s_D + 1
    cLayer.Z = zeros(eltype(cLayer.prevLayer.A),
                 n_H, n_W, n_D, cLayer.channels, m)

    cLayer.forwCount += 1
    convolve!(cLayer, Ai)

    return nothing
end #function layerForProp!(cLayer::Conv3D)



### Pooling Layers

function layerForProp!(cLayer::OneD) where {OneD <: Union{MaxPool1D, AveragePool1D}}
    Ai = padding(cLayer)
    n_Hi, ci, m = size(Ai)
    s_H = cLayer.s
    f_H = cLayer.f
    n_H = (n_Hi - f_H) ÷ s_H + 1

    cLayer.A = zeros(eltype(cLayer.prevLayer.A),
                     n_H, cLayer.channels, m)
    pooling!(cLayer, Ai)

    cLayer.forwCount += 1
    return nothing
end #unction layerForProp!(cLayer::OneD) where {OneD <: Union{MaxPool1D, AveragePool1D}}

function layerForProp!(cLayer::TwoD) where {TwoD <: Union{MaxPool2D, AveragePool2D}}
    Ai = padding(cLayer)
    n_Hi, n_Wi, ci, m = size(Ai)
    s_H, s_W = cLayer.s
    f_H, f_W = cLayer.f
    n_H = (n_Hi - f_H) ÷ s_H + 1
    n_W = (n_Wi - f_W) ÷ s_W + 1

    cLayer.A = zeros(eltype(cLayer.prevLayer.A),
                     n_H, n_W, cLayer.channels, m)
    pooling!(cLayer, Ai)

    cLayer.forwCount += 1
    return nothing

end #function layerForProp!(cLayer::TwoD) where {TwoD <: Union{MaxPool2D, AveragePool2D}}

function layerForProp!(cLayer::ThreeD) where {ThreeD <: Union{MaxPool3D, AveragePool3D}}
    Ai = padding(cLayer)

    n_Hi, n_Wi, n_Di, ci, m = size(Ai)
    s_H, s_W, s_D = cLayer.s
    f_H, f_W, f_D = cLayer.f

    n_H = (n_Hi - f_H) ÷ s_H + 1
    n_W = (n_Wi - f_W) ÷ s_W + 1
    n_D = (n_Di - f_D) ÷ s_D + 1

    cLayer.A = zeros(eltype(cLayer.prevLayer.A),
                 n_H, n_W, n_D, cLayer.channels, m)

    pooling!(cLayer, Ai)

    cLayer.forwCount += 1
    return nothing
end #function layerForProp!(cLayer::ThreeD) where {ThreeD <: Union{MaxPool3D, AveragePool3D}}
