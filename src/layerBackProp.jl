

function layerBackProp!(
    cLayer::Layer,
    model::Model,
    actFun::SoS,
    labels::AbstractArray,
    ) where {SoS <: Union{Type{softmax}, Type{σ}}}


    lossFun = model.lossFun
    dlossFun = Symbol("d",lossFun)
    A = cLayer.A
    Y = labels
    cLayer.dZ = eval(:($dlossFun($A, $Y)))
    return nothing
end #softmax or σ layerBackProp



function layerBackProp!(cLayer::FCLayer, model::Model; labels=nothing)
    prevLayer = cLayer.prevLayer
    lossFun = model.lossFun

    m = size(cLayer.A)[end]

    A = cLayer.A

    regulization, λ = model.regulization, model.λ

    actFun = cLayer.actFun

    if cLayer.actFun == model.outLayer.actFun

        layerBackProp!(cLayer, model, eval(:($actFun)), labels)

    elseif all(i->(i.backCount==cLayer.nextLayers[1].backCount), cLayer.nextLayers)
        dA = []
        for nextLayer in cLayer.nextLayers
            try
                dA .+= nextLayer.W'nextLayer.dZ
            catch e
                if e isa ErrorException #in case nextLayer does not has W parameter
                    try
                        dA .+= nextLayer.dZ
                    catch e1 #in case it is the first time
                        dA = nextLayer.dZ #need to initialize dA
                    end
                elseif e isa DimensionMismatch #in case W exists but first time
                    dA = nextLayer.W'nextLayer.dZ #to initialize dA
                end
            end #try/catch
        end #for

        keepProb = cLayer.keepProb
        if keepProb < 1.0 #to save time of multiplication in case keepProb was one
           D = rand(cLayer.numNodes,1) .< keepProb
           dA .*= D
           dA ./= keepProb
        end

        dActFun = Symbol("d",cLayer.actFun)

        Z = cLayer.Z

        cLayer.dZ = dA .* eval(:($dActFun($Z)))

    else #in case not every next layer done backprop
        return nothing

    end #if all(i->(i.backCount==cLayer.nextLayers[1].backCount), cLayer.nextLayers)

    cLayer.dW = cLayer.dZ*cLayer.prevLayer.A' ./m

    if regulization > 0
        if regulization==1
            cLayer.dW .+= (λ/2m)
        else
            cLayer.dW .+= (λ/m) .* cLayer.W
        end
    end

    cLayer.dB = 1/m .* sum(cLayer.dZ, dims=2)

    cLayer.backCount += 1

    return nothing

end #function layerBackProp(cLayer::FCLayer)



function layerBackProp!(cLayer::Activation, model::Model; labels=nothing)
    prevLayer = cLayer.prevLayer
    lossFun = model.lossFun

    m = size(cLayer.A)[end]

    regulization, λ = model.regulization, model.λ

    actFun = cLayer.actFun

    if cLayer.actFun == model.outLayer.actFun

        layerBackProp!(cLayer, model, eval(:($actFun)), labels)

    elseif all(i->(i.backCount==cLayer.nextLayers[1].backCount), cLayer.nextLayers)
        dA = []
        for nextLayer in cLayer.nextLayers
            try
                dA .+= nextLayer.W'nextLayer.dZ
            catch e
                if e isa ErrorException #in case nextLayer does not has W parameter
                    try
                        dA .+= nextLayer.dZ
                    catch e1 #in case it is the first time
                        dA = nextLayer.dZ #need to initialize dA
                    end
                elseif e isa DimensionMismatch #in case W exists but first time
                    dA = nextLayer.W'nextLayer.dZ #to initialize dA
                end
            end #try/catch
        end #for

        try
            keepProb = cLayer.keepProb
            if keepProb < 1.0 #to save time of multiplication in case keepProb was one
               D = rand(cLayer.numNodes,1) .< keepProb
               dA .*= D
               dA ./= keepProb
            end
        catch e

        end

        dActFun = Symbol("d",cLayer.actFun)

        Z = cLayer.prevLayer.A

        cLayer.dZ = dA .* eval(:($dActFun($Z)))

    else #in case not every next layer done backprop
        return nothing

    end #if all(i->(i.backCount==cLayer.nextLayers[1].backCount), cLayer.nextLayers)

    cLayer.backCount += 1

    return nothing
end #function layerBackProp!(cLayer::Activation



function layerBackProp!(cLayer::AddLayer, model::Model; labels=nothing)

    m = size(cLayer.A)[end]

    A = cLayer.A

    regulization, λ = model.regulization, model.λ

    if all(i->(i.backCount==cLayer.nextLayers[1].backCount), cLayer.nextLayers)
        dA = []
        for nextLayer in cLayer.nextLayers
            try
                dA .+= nextLayer.W'nextLayer.dZ
            catch e
                if e isa ErrorException #in case nextLayer does not has W parameter
                    try
                        dA .+= nextLayer.dZ
                    catch e1 #in case it is the first time
                        dA = nextLayer.dZ #need to initialize dA
                    end
                elseif e isa DimensionMismatch #in case W exists but first time
                    dA = nextLayer.W'nextLayer.dZ #to initialize dA
                end
            end #try/catch
        end #for

        cLayer.dZ = dA


    else #in case not every next layer done backprop
        return nothing

    end #if all(i->(i.backCount==cLayer.nextLayers[1].backCount), cLayer.nextLayers)

    cLayer.backCount += 1

    return nothing
end #function layerBackProp!(cLayer::AddLayer



function layerBackProp!(cLayer::Input, model::Model; labels=nothing)

    m = size(cLayer.A)[end]

    A = cLayer.A

    regulization, λ = model.regulization, model.λ

    if all(i->(i.backCount==cLayer.nextLayers[1].backCount), cLayer.nextLayers)
        dA = []
        for nextLayer in cLayer.nextLayers
            try
                dA .+= nextLayer.W'nextLayer.dZ
            catch e
                if e isa ErrorException #in case nextLayer does not has W parameter
                    try
                        dA .+= nextLayer.dZ
                    catch e1 #in case it is the first time
                        dA = nextLayer.dZ #need to initialize dA
                    end
                elseif e isa DimensionMismatch #in case W exists but first time
                    dA = nextLayer.W'nextLayer.dZ #to initialize dA
                end
            end #try/catch
        end #for

        cLayer.dZ = dA

    else #in case not every next layer done backprop
        return nothing

    end #if all(i->(i.backCount==cLayer.nextLayers[1].backCount), cLayer.nextLayers)

    cLayer.backCount += 1

    return nothing
end #function layerBackProp!(cLayer::Input


export layerBackProp!
