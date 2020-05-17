

function (l::Conv1D)(li_1::Layer)
    l.prevLayer = li_1
    if ! in(l,li_1.nextLayers)
        push!(li_1.nextLayers, l)
    end
    padding = l.padding
    c = l.channels
    n_Hi, ci = l.inputS = li_1.outputS
    s_H = l.s
    f_H = l.f
    if l.padding == :same
        l.outputS = (n_Hi, c)
    elseif l.padding == :valid
        n_H = ((n_Hi - f_H) ÷ s_H) + 1
        l.outputS = (n_H, c)
    end
    return l
end

function (l::Conv2D)(li_1::Layer)
    l.prevLayer = li_1
    if ! in(l,li_1.nextLayers)
        push!(li_1.nextLayers, l)
    end
    padding = l.padding
    c = l.channels
    n_Hi, n_Wi, ci = l.inputS = li_1.outputS
    s_H, s_W = l.s
    f_H, f_W = l.f
    if l.padding == :same
        l.outputS = (n_Hi, n_Wi, c)
    elseif l.padding == :valid
        n_H = ((n_Hi - f_H) ÷ s_H) + 1
        n_W = ((n_Wi - f_W) ÷ s_W) + 1
        l.outputS = (n_H, n_W, c)
    end

    return l
end

function (l::Conv3D)(li_1::Layer)
    l.prevLayer = li_1
    if ! in(l,li_1.nextLayers)
        push!(li_1.nextLayers, l)
    end
    padding = l.padding
    c = l.channels
    n_Hi, n_Wi, n_Di, ci = l.inputS = li_1.outputS
    s_H, s_W, s_D = l.s
    f_H, f_W, f_D = l.f
    if l.padding == :same
        l.outputS = (n_Hi, n_Wi, n_Di, c)
    elseif l.padding == :valid
        n_H = ((n_Hi - f_H) ÷ s_H) + 1
        n_W = ((n_Wi - f_W) ÷ s_W) + 1
        n_D = ((n_Di - f_D) ÷ s_D) + 1
        l.outputS = (n_H, n_W, n_D, c)
    end
    return l
end

function (l::MaxPool1D)(li_1::Layer)
    l.prevLayer = li_1
    if ! in(l,li_1.nextLayers)
        push!(li_1.nextLayers, l)
    end
    padding = l.padding
    c = l.channels = li_1.channels
    n_Hi, ci = l.inputS = li_1.outputS
    s_H = l.s
    f_H = l.f
    if l.padding == :same
        l.outputS = (n_Hi, c)
    elseif l.padding == :valid
        n_H = ((n_Hi - f_H) ÷ s_H) + 1
        l.outputS = (n_H, c)
    end
    return l
end

function (l::MaxPool2D)(li_1::Layer)
    l.prevLayer = li_1
    if ! in(l,li_1.nextLayers)
        push!(li_1.nextLayers, l)
    end
    padding = l.padding
    c = l.channels = li_1.channels
    n_Hi, n_Wi, ci = l.inputS = li_1.outputS
    s_H, s_W = l.s
    f_H, f_W = l.f
    if l.padding == :same
        l.outputS = (n_Hi, n_Wi, c)
    elseif l.padding == :valid
        n_H = ((n_Hi - f_H) ÷ s_H) + 1
        n_W = ((n_Wi - f_W) ÷ s_W) + 1
        l.outputS = (n_H, n_W, c)
    end

    return l
end

function (l::MaxPool3D)(li_1::Layer)
    l.prevLayer = li_1
    if ! in(l,li_1.nextLayers)
        push!(li_1.nextLayers, l)
    end
    padding = l.padding
    c = l.channels = li_1.channels
    n_Hi, n_Wi, n_Di, ci = l.inputS = li_1.outputS
    s_H, s_W, s_D = l.s
    f_H, f_W, f_D = l.f
    if l.padding == :same
        l.outputS = (n_Hi, n_Wi, n_Di, c)
    elseif l.padding == :valid
        n_H = ((n_Hi - f_H) ÷ s_H) + 1
        n_W = ((n_Wi - f_W) ÷ s_W) + 1
        n_D = ((n_Di - f_D) ÷ s_D) + 1
        l.outputS = (n_H, n_W, n_D, c)
    end
    return l
end

function (l::AveragePool1D)(li_1::Layer)
    l.prevLayer = li_1
    if ! in(l,li_1.nextLayers)
        push!(li_1.nextLayers, l)
    end
    padding = l.padding
    c = l.channels = li_1.channels
    n_Hi, ci = l.inputS = li_1.outputS
    s_H = l.s
    f_H = l.f
    if l.padding == :same
        l.outputS = (n_Hi, c)
    elseif l.padding == :valid
        n_H = ((n_Hi - f_H) ÷ s_H) + 1
        l.outputS = (n_H, c)
    end
    return l
end

function (l::AveragePool2D)(li_1::Layer)
    l.prevLayer = li_1
    if ! in(l,li_1.nextLayers)
        push!(li_1.nextLayers, l)
    end
    padding = l.padding
    c = l.channels = li_1.channels
    n_Hi, n_Wi, ci = l.inputS = li_1.outputS
    s_H, s_W = l.s
    f_H, f_W = l.f
    if l.padding == :same
        l.outputS = (n_Hi, n_Wi, c)
    elseif l.padding == :valid
        n_H = ((n_Hi - f_H) ÷ s_H) + 1
        n_W = ((n_Wi - f_W) ÷ s_W) + 1
        l.outputS = (n_H, n_W, c)
    end

    return l
end

function (l::AveragePool3D)(li_1::Layer)
    l.prevLayer = li_1
    if ! in(l,li_1.nextLayers)
        push!(li_1.nextLayers, l)
    end
    padding = l.padding
    c = l.channels = li_1.channels
    n_Hi, n_Wi, n_Di, ci = l.inputS = li_1.outputS
    s_H, s_W, s_D = l.s
    f_H, f_W, f_D = l.f
    if l.padding == :same
        l.outputS = (n_Hi, n_Wi, n_Di, c)
    elseif l.padding == :valid
        n_H = ((n_Hi - f_H) ÷ s_H) + 1
        n_W = ((n_Wi - f_W) ÷ s_W) + 1
        n_D = ((n_Di - f_D) ÷ s_D) + 1
        l.outputS = (n_H, n_W, n_D, c)
    end
    return l
end
