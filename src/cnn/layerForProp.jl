

###convolution layers forprop

function layerForProp!(cLayer::Conv1D, Ai::AoN; fastConvolve=false, NNlibConv = true) where {AoN <: Union{AbstractArray, Nothing}}
    if Ai == nothing
        Ai = cLayer.prevLayer.A
    end
    cLayer.inputS = size(Ai)
    Ai = padding(cLayer, Ai)
    n_Hi, ci, m = size(Ai)
    s_H = cLayer.s
    f_H = cLayer.f
    c = cLayer.channels
    n_H = (n_Hi - f_H) ÷ s_H + 1

    if NNlibConv
        NNConv!(cLayer, Ai)
    elseif fastConvolve
        ## in case input different size than previous time
        if (n_H* c, n_Hi*ci) != size(cLayer.K)
            cLayer.K = unroll(cLayer, (n_Hi, ci, m))
        end
        fastConvolve!(cLayer, Ai)
    else
        cLayer.Z = zeros(eltype(Ai),
                         n_H, cLayer.channels, m)
        convolve!(cLayer, Ai)
    end

    cLayer.outputS = size(cLayer.A)

    Ai = nothing
    cLayer.forwCount += 1
    # Base.GC.gc()
    return nothing

end #function layerForProp!(cLayer::Conv1D)

function layerForProp!(cLayer::Conv2D, Ai::AoN; fastConvolve=false, NNlibConv = true) where {AoN <: Union{AbstractArray, Nothing}}
    if Ai == nothing
        Ai = cLayer.prevLayer.A
    end
    cLayer.inputS = size(Ai)
    Ai = padding(cLayer, Ai)
    n_Hi, n_Wi, ci, m = size(Ai)
    c = cLayer.channels
    s_H, s_W = cLayer.s
    f_H, f_W = cLayer.f
    c = cLayer.channels
    n_H = (n_Hi - f_H) ÷ s_H + 1
    n_W = (n_Wi - f_W) ÷ s_W + 1



    if NNlibConv
        NNConv!(cLayer, Ai)
    elseif fastConvolve
        ## in case input different size than previous time
        if (n_W*n_H* c, n_Hi*n_Wi*ci) != size(cLayer.K)
            cLayer.K = unroll(cLayer, (n_Hi, n_Wi, ci, m))
        end #if (n_W*n_H* c, n_Hi*n_Wi*ci) != size(cLayer.K)
        fastConvolve!(cLayer, Ai)
    else
        cLayer.Z = zeros(eltype(Ai),
                         n_H, n_W, cLayer.channels, m)
        convolve!(cLayer, Ai)
    end #if fastConvolve

    cLayer.outputS = size(cLayer.A)

    Ai = nothing
    cLayer.forwCount += 1

    # Base.GC.gc()
    return nothing
end #function layerForProp!(cLayer::Conv2D)

function layerForProp!(cLayer::Conv3D, Ai::AoN; fastConvolve=false, NNlibConv = true) where {AoN <: Union{AbstractArray, Nothing}}
    if Ai == nothing
        Ai = cLayer.prevLayer.A
    end
    cLayer.inputS = size(Ai)
    Ai = padding(cLayer, Ai)

    n_Hi, n_Wi, n_Di, ci, m = size(Ai)
    s_H, s_W, s_D = cLayer.s
    f_H, f_W, f_D = cLayer.f
    c = cLayer.channels

    n_H = (n_Hi - f_H) ÷ s_H + 1
    n_W = (n_Wi - f_W) ÷ s_W + 1
    n_D = (n_Di - f_D) ÷ s_D + 1


    if NNlibConv
        NNConv!(cLayer, Ai)
    elseif fastConvolve
        ## in case input different size than previous time
        if (n_W*n_H*n_D* c, n_Hi*n_Wi*n_Di*ci) != size(cLayer.K)
            cLayer.K = unroll(cLayer, (n_Hi, n_Wi, n_Di, ci, m))
        end #if (n_W*n_H* c, n_Hi*n_Wi*ci) != size(cLayer.K)
        fastConvolve!(cLayer, Ai)
    else
        cLayer.Z = zeros(eltype(Ai),
                     n_H, n_W, n_D, cLayer.channels, m)
        convolve!(cLayer, Ai)
    end #if fastConvolve

    cLayer.outputS = size(cLayer.A)

    cLayer.forwCount += 1

    Ai = nothing

    # Base.GC.gc()
    return nothing
end #function layerForProp!(cLayer::Conv3D)



### Pooling Layers

function layerForProp!(cLayer::OneD, Ai::AoN; NNlibPool = true) where {OneD <: Union{MaxPool1D, AveragePool1D}, AoN <: Union{AbstractArray, Nothing}}
    if Ai == nothing
        Ai = cLayer.prevLayer.A
    end
    cLayer.inputS = size(Ai)
    Ai = padding(cLayer, Ai)
    n_Hi, ci, m = size(Ai)
    s_H = cLayer.s
    f_H = cLayer.f
    c = cLayer.channels

    n_H = (n_Hi - f_H) ÷ s_H + 1

    if NNlibConv
        if cLayer isa MaxPoolLayer
            cLayer.A = maxpool(Ai, cLayer.f, stride=s_H)
        elseif cLayer isa AveragePoolLayer
            cLayer.A = meanpool(Ai, cLayer.f, stride=s_H)
        end #if cLayer isa MaxPoolLayer
    elseif f_H == s_H #to use the built-in reshape and maximum and mean
        cLayer.A = fastPooling(cLayer, Ai)
    else
        cLayer.A = zeros(eltype(cLayer.prevLayer.A),
                         n_H, cLayer.channels, m)
        pooling!(cLayer, Ai)
    end #if NNlibConv

    cLayer.outputS = size(cLayer.A)
    Ai = nothing
    cLayer.forwCount += 1
    # Base.GC.gc()
    return nothing
end #unction layerForProp!(cLayer::OneD) where {OneD <: Union{MaxPool1D, AveragePool1D}}

function layerForProp!(cLayer::TwoD, Ai::AoN; NNlibPool = true) where {TwoD <: Union{MaxPool2D, AveragePool2D}, AoN <: Union{AbstractArray, Nothing}}
    if Ai == nothing
        Ai = cLayer.prevLayer.A
    end
    cLayer.inputS = size(Ai)
    Ai = padding(cLayer, Ai)
    n_Hi, n_Wi, ci, m = size(Ai)
     s_H, s_W = S = cLayer.s
    f_H, f_W = F = cLayer.f
    c = cLayer.channels
    n_H = (n_Hi - f_H) ÷ s_H + 1
    n_W = (n_Wi - f_W) ÷ s_W + 1

    if NNlibConv
        if cLayer isa MaxPoolLayer
            cLayer.A = maxpool(Ai, cLayer.f, stride=S)
        elseif cLayer isa AveragePoolLayer
            cLayer.A = meanpool(Ai, cLayer.f, stride=S)
        end #if cLayer isa MaxPoolLayer
    elseif F == S #to use the built-in reshape and maximum and mean
        cLayer.A = fastPooling(cLayer, Ai)
    else
        cLayer.A = zeros(eltype(cLayer.prevLayer.A),
                         n_H, n_W, cLayer.channels, m)
        pooling!(cLayer, Ai)
    end #if NNlibConv

    cLayer.outputS = size(cLayer.A)

    Ai = nothing
    cLayer.forwCount += 1
    # Base.GC.gc()
    return nothing

end #function layerForProp!(cLayer::TwoD) where {TwoD <: Union{MaxPool2D, AveragePool2D}}

function layerForProp!(cLayer::ThreeD, Ai::AoN; NNlibPool = true) where {ThreeD <: Union{MaxPool3D, AveragePool3D}, AoN <: Union{AbstractArray, Nothing}}
    if Ai == nothing
        Ai = cLayer.prevLayer.A
    end
    cLayer.inputS = size(Ai)
    Ai = padding(cLayer, Ai)

    n_Hi, n_Wi, n_Di, ci, m = size(Ai)
    s_H, s_W, s_D = S = cLayer.s
    f_H, f_W, f_D = F = cLayer.f
    c = cLayer.channels

    n_H = (n_Hi - f_H) ÷ s_H + 1
    n_W = (n_Wi - f_W) ÷ s_W + 1
    n_D = (n_Di - f_D) ÷ s_D + 1

    if NNlibConv
        if cLayer isa MaxPoolLayer
            cLayer.A = maxpool(Ai, cLayer.f, stride=S)
        elseif cLayer isa AveragePoolLayer
            cLayer.A = meanpool(Ai, cLayer.f, stride=S)
        end #if cLayer isa MaxPoolLayer
    elseif F == S #to use the built-in reshape and maximum and mean
        cLayer.A = fastPooling(cLayer, Ai)
    else
        cLayer.A = zeros(eltype(cLayer.prevLayer.A),
                     n_H, n_W, n_D, cLayer.channels, m)
        pooling!(cLayer, Ai)
    end #if NNlibConv

    cLayer.outputS = size(cLayer.A)

    Ai = nothing
    cLayer.forwCount += 1

    # Base.GC.gc()
    return nothing
end #function layerForProp!(cLayer::ThreeD) where {ThreeD <: Union{MaxPool3D, AveragePool3D}}
