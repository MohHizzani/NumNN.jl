

function layerBackProp(
    cLayer::ConvLayer,
    model::Model,
    actFun::SoS,
    labels::AbstractArray,
    ) where {SoS <: Union{Type{softmax}, Type{σ}}}


    lossFun = model.lossFun
    dlossFun = Symbol("d",lossFun)
    A = cLayer.A
    Y = labels
    dZ = eval(:($dlossFun($A, $Y)))

    return dZ
end #softmax or σ layerBackProp


function layerBackProp!(cLayer::ConvLayer, model::Model; labels=nothing)

    m = size(cLayer.A)[end]

    A = cLayer.A

    regulization, λ = model.regulization, model.λ

    actFun = cLayer.actFun

    dZ = []
    if cLayer.actFun == model.outLayer.actFun

        dZ = layerBackProp(cLayer, model, eval(:($actFun)), labels)

    elseif all(i->(i.backCount==cLayer.nextLayers[1].backCount), cLayer.nextLayers)
        dA = []
        for nextLayer in cLayer.nextLayers
            try
                dA .+= nextLayer.dA
            catch e
                dA = nextLayer.dA #need to initialize dA
            end #try/catch
        end #for

        dActFun = Symbol("d",cLayer.actFun)

        Z = cLayer.Z

        dZ = dA .* eval(:($dActFun($Z)))
    else #in case not every next layer done backprop
        return nothing

    end #if all(i->(i.backCount==cLayer.nextLayers[1].backCount), cLayer.nextLayers)

    Ai = padding(cLayer)

    dAi = similar(Ai)
    dAi .= 0

    dconvolve!(cLayer,Ai,dAi,dZ)

    cLayer.backCount += 1

    return nothing
end #function layerBackProp!(cLayer::Input


### Pooling Layers

function layerBackProp!(cLayer::OneD, model::Model; labels=nothing) where {OneD <: Union{MaxPool1D, AveragePool1D}}

    Ai = padding(cLayer)
    dAi = similar(Ai)
    dAi .= 0

    dpooling!(cLayer, Ai, dAi)

    cLayer.forwCount += 1

    return nothing

end #unction layerBackProp!(cLayer::OneD) where {OneD <: Union{MaxPool1D, AveragePool1D}}

function layerBackProp!(cLayer::TwoD, model::Model; labels=nothing) where {TwoD <: Union{MaxPool2D, AveragePool2D}}

    Ai = padding(cLayer)
    dAi = similar(Ai)
    dAi .= 0

    dpooling!(cLayer, Ai, dAi)

    cLayer.forwCount += 1

    return nothing

end #function layerBackProp!(cLayer::TwoD) where {TwoD <: Union{MaxPool2D, AveragePool2D}}

function layerBackProp!(cLayer::ThreeD, model::Model; labels=nothing) where {ThreeD <: Union{MaxPool3D, AveragePool3D}}

    Ai = padding(cLayer)
    dAi = similar(Ai)
    dAi .= 0

    dpooling!(cLayer, Ai, dAi)

    cLayer.forwCount += 1

    return nothing

end #function layerBackProp!(cLayer::ThreeD) where {ThreeD <: Union{MaxPool3D, AveragePool3D}}
