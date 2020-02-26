
###FCLayer forprop

function layerForProp!(cLayer::FCLayer)
    cLayer.Z = cLayer.W * cLayer.prevLayer.A .+ cLayer.B
    actFun = cLayer.actFun
    Z = cLayer.Z
    cLayer.A = eval(:($actFun($Z)))
    return nothing
end #function layerForProp!(cLayer::FCLayer)


###AddLayer forprop

function layerForProp!(cLayer::AddLayer)
    cLayer.A = similar(cLayer.prevLayer.A)
    cLayer.A .= 0
    for prevLayer in cLayer.prevLayer
        cLayer.A .+= prevLayer.A
    end
    return nothing
end #function layerForProp!(cLayer::AddLayer)

###Activation forprop


function layerForProp!(cLayer::Activation)
    actFun = cLayer.actFun
    Ai = cLayer.prevLayer.A
    cLayer.A = eval(:($actFun($Ai)))

    return nothing
end #function layerForProp!(cLayer::Activation)



###convolution layers forprop

function layerForProp!(cLayer::Conv1D)
    n_Hi, ci, m = size(cLayer.prevLayer.A)
    s_H = cLayer.s
    f_H = cLayer.f
    if cLayer.padding == :same
        p_H = Integer(ceil((s_H*(n_Hi-1)-n_Hi+f_H)/2))
        n_H = n_Hi
        Ai = padding(cLayer.prevLayer.A, p_H)
    elseif cLayer.padding == :valid
        Ai = cLayer.prevLayer.A
        n_H = (n_Hi - f_H) ÷ s_H + 1
    end #if cLayer.padding == :same
    cLayer.Z = zeros(eltype(cLayer.prevLayer.A),
                     n_H, cLayer.channels, m)

    convolve!(cLayer, Ai)

    return nothing

end #function layerForProp!(cLayer::Conv1D)

function layerForProp!(cLayer::Conv2D)
    n_Hi, n_Wi, ci, m = size(cLayer.prevLayer.A)
    s_H, s_W = cLayer.s
    f_H, f_W = cLayer.f
    if cLayer.padding == :same
        p_H = Integer(ceil((s_H*(n_Hi-1)-n_Hi+f_H)/2))
        p_W = Integer(ceil((s_W*(n_Wi-1)-n_Wi+f_W)/2))
        n_H, n_W = n_Hi, n_Wi
        Ai = padding(cLayer.prevLayer.A, p_H, p_W)
    elseif cLayer.padding == :valid
        Ai = cLayer.prevLayer.A
        n_H = (n_Hi - f_H) ÷ s_H + 1
        n_W = (n_Wi - f_W) ÷ s_W + 1
    end #if cLayer.padding == :same
    cLayer.Z = zeros(eltype(cLayer.prevLayer.A),
                     n_H, n_W, cLayer.channels, m)

    convolve!(cLayer, Ai)

    return nothing
end #function layerForProp!(cLayer::Conv2D)

function layerForProp!(cLayer::Conv3D)
    n_Hi, n_Wi, n_Di, ci, m = size(cLayer.prevLayer.A)
    s_H, s_W, s_D = cLayer.s
    f_H, f_W, f_D = cLayer.f
    if cLayer.padding == :same
        p_H = Integer(ceil((s_H*(n_Hi-1)-n_Hi+f_H)/2))
        p_W = Integer(ceil((s_W*(n_Wi-1)-n_Wi+f_W)/2))
        p_D = Integer(ceil((s_D*(n_Di-1)-n_Di+f_D)/2))
        n_H, n_W, n_D = n_Hi, n_Wi, n_D
        Ai = padding(cLayer.prevLayer.A, p_H, p_W, p_D)
    elseif cLayer.padding == :valid
        Ai = cLayer.prevLayer.A
        n_H = (n_Hi - f_H) ÷ s_H + 1
        n_W = (n_Wi - f_W) ÷ s_W + 1
        n_D = (n_Di - f_D) ÷ s_D + 1
    end #if cLayer.padding == :same
    cLayer.Z = zeros(eltype(cLayer.prevLayer.A),
                     n_H, n_W, n_D, cLayer.channels, m)

    convolve!(cLayer, Ai)

    return nothing
end #function layerForProp!(cLayer::Conv3D)





export layerForProp!
