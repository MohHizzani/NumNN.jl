using Statistics


function fastPooling!(cLayer::OneD,
                      Ai) where {OneD <: Union{MaxPool1D, AveragePool1D}}

    n_Hi, c, m = size(Ai)
    s_H = cLayer.s
    f_H = cLayer.f
    n_H = (n_Hi - f_H) ÷ s_H + 1

    if cLayer isa MaxPoolLayer
        cLayer.A = reshape(maximum(reshape(A, f_H,n_H,c,m),dims=1),n_H,n_W,c,m)
    else
        cLayer.A = reshape(mean(reshape(A, f_H,n_H,c,m),dims=1),n_H,n_W,c,m)
    end #if cLayer isa MaxPoolLayer

    return nothing
end #function fastPooling!(cLayer::OneD


function fastPooling!(cLayer::TwoD,
                      Ai) where {TwoD <: Union{MaxPool2D, AveragePool2D}}

    n_Hi, n_Wi, c, m = size(Ai)
    s_H, s_W = cLayer.s
    f_H, f_W = cLayer.f
    n_H = (n_Hi - f_H) ÷ s_H + 1
    n_W = (n_Wi - f_W) ÷ s_W + 1

    if cLayer isa MaxPoolLayer
        cLayer.A = reshape(maximum(permutedims(reshape(A, f_H,n_H,f_W,n_W,c,m),[1,3,2,4,5,6]),dims=1:2),n_H,n_W,c,m)
    else
        cLayer.A = reshape(mean(permutedims(reshape(A, f_H,n_H,f_W,n_W,c,m),[1,3,2,4,5,6]),dims=1:2),n_H,n_W,c,m)
    end #if cLayer isa MaxPoolLayer

    return nothing
end #function fastPooling!(cLayer::TwoD,


function fastPooling!(cLayer::ThreeD,
                      Ai) where {ThreeD <: Union{MaxPool3D, AveragePool3D}}

    n_Hi, n_Wi, n_Di, c, m = size(Ai)
    s_H, s_W, s_D = cLayer.s
    f_H, f_W, f_D = cLayer.f

    n_H = (n_Hi - f_H) ÷ s_H + 1
    n_W = (n_Wi - f_W) ÷ s_W + 1
    n_D = (n_Di - f_D) ÷ s_D + 1

    if cLayer isa MaxPoolLayer
        cLayer.A = reshape(maximum(permutedims(reshape(A, f_H,n_H,f_W,n_W,f_D,n_D,c,m),[1,3,5,2,4,6,7,8]),dims=1:3),n_H,n_W,n_D,c,m)
    else
        cLayer.A = reshape(mean(permutedims(reshape(A, f_H,n_H,f_W,n_W,f_D,n_D,c,m),[1,3,5,2,4,6,7,8]),dims=1:3),n_H,n_W,n_D,c,m)
    end #if cLayer isa MaxPoolLayer

    return nothing
end #function fastPooling!(cLayer::ThreeD,

export fastPooling!
