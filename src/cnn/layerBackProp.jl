

### convlayers
function layerBackProp(
    cLayer::ConvLayer,
    model::Model,
    actFun::SoS,
    labels::AbstractArray,
) where {SoS<:Union{Type{softmax},Type{σ}}}


    lossFun = model.lossFun
    dlossFun = Symbol("d", lossFun)
    A = cLayer.A
    Y = labels
    dZ = eval(:($dlossFun($A, $Y)))

    return dZ
end #softmax or σ layerBackProp


function layerBackProp!(
    cLayer::ConvLayer,
    model::Model,
    Ai::AoN = nothing,
    Ao::AoN = nothing,
    dAo::AoN = nothing;
    labels::AoN = nothing,
    img2colConvolve::Bool = false,
    NNlib::Bool = true,
) where {AoN<:Union{AbstractArray,Nothing}}

    if Ai == nothing
        Ai = cLayer.prevLayer.A
    end
    if Ao == nothing
        Ao = cLayer.A
    end
    m = size(Ao)[end]

    regulization, λ = model.regulization, model.λ

    actFun = cLayer.actFun

    dZ = []
    if cLayer.actFun == model.outLayer.actFun
        dZ = layerBackProp(cLayer, model, eval(:($actFun)), labels)
    elseif dAo != nothing
        dActFun = Symbol("d", cLayer.actFun)

        Z = cLayer.Z

        dZ = dAo .* eval(:($dActFun($Z)))
    elseif all(
        i -> (i.backCount == cLayer.nextLayers[1].backCount),
        cLayer.nextLayers,
    )
        dAo = []
        for nextLayer in cLayer.nextLayers
            try
                dAo .+= nextLayer.dA
            catch e
                dAo = nextLayer.dA #need to initialize dA
            end #try/catch
        end #for


        dActFun = Symbol("d", cLayer.actFun)

        Z = cLayer.Z

        dZ = dAo .* eval(:($dActFun($Z)))
    else #in case not every next layer done backprop
        return nothing

    end #if all(i->(i.backCount==cLayer.nextLayers[1].backCount), cLayer.nextLayers)

    if NNlib

        dNNConv!(cLayer, dZ, Ai, Ao)

    else

        cLayer.dA = similar(Ai) #the size before padding
        cLayer.dA .= 0

        if img2colConvolve
            dimg2colConvolve!(cLayer, Ai, cLayer.dA, dZ)
        else
            dconvolve!(cLayer, Ai, cLayer.dA, dZ)
        end #if img2colConvolve

    end #if NNlib

    cLayer.backCount += 1

    return nothing
end #function layerBackProp!(cLayer::Input


### Pooling Layers

#import only the needed parts not to have conflict
import NNlib.∇maxpool, NNlib.∇meanpool, NNlib.∇maxpool!, NNlib.∇meanpool!, NNlib.PoolDims

function layerBackProp!(
    cLayer::OneD,
    model::Model,
    Ai::AoN = nothing,
    Ao::AoN = nothing,
    dAo::AoN = nothing;
    labels::AoN = nothing,
    NNlib::Bool = true,
) where {OneD<:Union{MaxPool1D,AveragePool1D},AoN<:Union{AbstractArray,Nothing}}
    if Ai == nothing
        Ai = cLayer.prevLayer.A
    end
    if Ao == nothing
        Ao = cLayer.A
    end

    padS = paddingSize(cLayer, Ai)
    cLayer.dA = similar(Ai) .= 0


    if dAo == nothing
        if all(
            i -> (i.backCount == cLayer.nextLayers[1].backCount),
            cLayer.nextLayers,
        )
            dAo = []
            for nextLayer in cLayer.nextLayers
                try
                    dAo .+= nextLayer.dA
                catch e
                    dAo = nextLayer.dA #need to initialize dA
                end #try/catch
            end #for
        else
            return nothing
        end #if all(i->(i.backCount==cLayer.nextLayers[1].backCount), cLayer.nextLayers)
    end #if dA==nothing

    if NNlib
        pooldims = PoolDims(Ai, cLayer.f, stride = cLayer.s, padding = padS)
        if cLayer isa MaxPoolLayer
            ∇maxpool!(cLayer.dA, dAo, Ao, Ai, pooldims)
        elseif cLayer isa AveragePoolLayer
            ∇meanpool!(cLayer.dA, dAo, Ao, Ai, pooldims)
        end #if cLayer isa MaxPoolLayer

    else

        if cLayer.s == cLayer.f
            dfastPooling!(cLayer, Ai, cLayer.dA, Ao, dAo)
        else
            dpooling!(cLayer, Ai, cLayer.dA, Ao, dAo)
        end #if cLayer.s == cLayer.f
    end #if NNlib

    cLayer.forwCount += 1
    return nothing

end #unction layerBackProp!(cLayer::OneD) where {OneD <: Union{MaxPool1D, AveragePool1D}}

function layerBackProp!(
    cLayer::TwoD,
    model::Model,
    Ai::AoN = nothing,
    Ao::AoN = nothing,
    dAo::AoN = nothing;
    labels::AoN = nothing,
    NNlib::Bool = true,
) where {TwoD<:Union{MaxPool2D,AveragePool2D},AoN<:Union{AbstractArray,Nothing}}
    if Ai == nothing
        Ai = cLayer.prevLayer.A
    end
    if Ao == nothing
        Ao = cLayer.A
    end

    padS = paddingSize(cLayer, Ai)
    cLayer.dA = similar(Ai) .= 0


    if dAo == nothing
        if all(
            i -> (i.backCount == cLayer.nextLayers[1].backCount),
            cLayer.nextLayers,
        )
            dAo = []
            for nextLayer in cLayer.nextLayers
                try
                    dAo .+= nextLayer.dA
                catch e
                    dAo = nextLayer.dA #need to initialize dA
                end #try/catch
            end #for
        else
            return nothing
        end #if all(i->(i.backCount==cLayer.nextLayers[1].backCount), cLayer.nextLayers)
    end #if dA==nothing

    if NNlib
        pooldims = PoolDims(Ai, cLayer.f, stride = cLayer.s, padding = padS)
        if cLayer isa MaxPoolLayer
            ∇maxpool!(cLayer.dA, dAo, Ao, Ai, pooldims)
        elseif cLayer isa AveragePoolLayer
            ∇meanpool!(cLayer.dA, dAo, Ao, Ai, pooldims)
        end #if cLayer isa MaxPoolLayer

    else

        if cLayer.s == cLayer.f
            dfastPooling!(cLayer, Ai, cLayer.dA, Ao, dAo)
        else
            dpooling!(cLayer, Ai, cLayer.dA, Ao, dAo)
        end #if cLayer.s == cLayer.f
    end #if NNlib

    cLayer.forwCount += 1
    return nothing

end #function layerBackProp!(cLayer::TwoD) where {TwoD <: Union{MaxPool2D, AveragePool2D}}

function layerBackProp!(
    cLayer::ThreeD,
    model::Model,
    Ai::AoN = nothing,
    Ao::AoN = nothing,
    dAo::AoN = nothing;
    labels::AoN = nothing,
    NNlib::Bool = true,
) where {
    ThreeD<:Union{MaxPool3D,AveragePool3D},
    AoN<:Union{AbstractArray,Nothing},
}
    if Ai == nothing
        Ai = cLayer.prevLayer.A
    end
    if Ao == nothing
        Ao = cLayer.A
    end

    padS = paddingSize(cLayer, Ai)
    cLayer.dA = similar(Ai) .= 0


    if dAo == nothing
        if all(
            i -> (i.backCount == cLayer.nextLayers[1].backCount),
            cLayer.nextLayers,
        )
            dAo = []
            for nextLayer in cLayer.nextLayers
                try
                    dAo .+= nextLayer.dA
                catch e
                    dAo = nextLayer.dA #need to initialize dA
                end #try/catch
            end #for
        else
            return nothing
        end #if all(i->(i.backCount==cLayer.nextLayers[1].backCount), cLayer.nextLayers)
    end #if dA==nothing

    if NNlib
        pooldims = PoolDims(Ai, cLayer.f, stride = cLayer.s, padding = padS)
        if cLayer isa MaxPoolLayer
            ∇maxpool!(cLayer.dA, dAo, Ao, Ai, pooldims)
        elseif cLayer isa AveragePoolLayer
            ∇meanpool!(cLayer.dA, dAo, Ao, Ai, pooldims)
        end #if cLayer isa MaxPoolLayer


    else

        if cLayer.s == cLayer.f
            dfastPooling!(cLayer, Ai, cLayer.dA, Ao, dAo)
        else
            dpooling!(cLayer, Ai, cLayer.dA, Ao, dAo)
        end #if cLayer.s == cLayer.f
    end #if NNlib

    cLayer.forwCount += 1
    return nothing

end #function layerBackProp!(cLayer::ThreeD) where {ThreeD <: Union{MaxPool3D, AveragePool3D}}
