

function (l::Conv1D)(li_1::Layer)
    l.prevLayer = li_1
    if ! in(l,li_1.nextLayers)
        push!(li_1.nextLayers, l)
    end
    return l
end

function (l::Conv2D)(li_1::Layer)
    l.prevLayer = li_1
    if ! in(l,li_1.nextLayers)
        push!(li_1.nextLayers, l)
    end
    return l
end

function (l::Conv3D)(li_1::Layer)
    l.prevLayer = li_1
    if ! in(l,li_1.nextLayers)
        push!(li_1.nextLayers, l)
    end
    return l
end

function (l::MaxPool1D)(li_1::Layer)
    l.prevLayer = li_1
    l.channels = li_1.channels
    if ! in(l,li_1.nextLayers)
        push!(li_1.nextLayers, l)
    end
    return l
end

function (l::MaxPool2D)(li_1::Layer)
    l.prevLayer = li_1
    l.channels = li_1.channels
    if ! in(l,li_1.nextLayers)
        push!(li_1.nextLayers, l)
    end
    return l
end

function (l::MaxPool3D)(li_1::Layer)
    l.prevLayer = li_1
    l.channels = li_1.channels
    if ! in(l,li_1.nextLayers)
        push!(li_1.nextLayers, l)
    end
    return l
end

function (l::AveragePool1D)(li_1::Layer)
    l.prevLayer = li_1
    l.channels = li_1.channels
    if ! in(l,li_1.nextLayers)
        push!(li_1.nextLayers, l)
    end
    return l
end

function (l::AveragePool2D)(li_1::Layer)
    l.prevLayer = li_1
    l.channels = li_1.channels
    if ! in(l,li_1.nextLayers)
        push!(li_1.nextLayers, l)
    end
    return l
end

function (l::AveragePool3D)(li_1::Layer)
    l.prevLayer = li_1
    l.channels = li_1.channels
    if ! in(l,li_1.nextLayers)
        push!(li_1.nextLayers, l)
    end
    return l
end
