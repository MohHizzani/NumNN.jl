

function layerBackProp(
    cLayer::FCLayer,
    model::Model,
    actFun::SoS,
    labels::AbstractArray,
    ) where {SoS <: Union{Type{softmax}, Type{σ}}}


    lossFun = model.lossFun
    dlossFun = Symbol("d",lossFun)
    A = cLayer.A
    Y = labels
    dZ = eval(:($dlossFun($A, $Y)))

    cLayer.dA = cLayer.W'dZ
    return dZ
end #softmax or σ layerBackProp


function layerBackProp(
    cLayer::Activation,
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


function layerBackProp!(cLayer::FCLayer, model::Model, dA::AoN=nothing; labels=nothing) where {AoN <: Union{AbstractArray, Nothing}}
    prevLayer = cLayer.prevLayer
    lossFun = model.lossFun

    m = size(cLayer.A)[end]

    A = cLayer.A

    regulization, λ = model.regulization, model.λ

    actFun = cLayer.actFun

    dZ = []
    if cLayer.actFun == model.outLayer.actFun

        dZ = layerBackProp(cLayer, model, eval(:($actFun)), labels)

    elseif dA != nothing
        keepProb = cLayer.keepProb
        if keepProb < 1.0 #to save time of multiplication in case keepProb was one
           D = rand(cLayer.channels,1) .< keepProb
           dA .*= D
           dA ./= keepProb
        end

        dActFun = Symbol("d",cLayer.actFun)

        Z = cLayer.Z

        dZ = dA .* eval(:($dActFun($Z)))

    elseif all(i->(i.backCount==cLayer.nextLayers[1].backCount), cLayer.nextLayers)
        dA = []
        for nextLayer in cLayer.nextLayers
            try
                dA .+= nextLayer.dA
            catch e #in case first time DimensionMismatch
                dA = nextLayer.dA
            end
        end #for

        keepProb = cLayer.keepProb
        if keepProb < 1.0 #to save time of multiplication in case keepProb was one
           D = rand(cLayer.channels,1) .< keepProb
           dA .*= D
           dA ./= keepProb
        end

        dActFun = Symbol("d",cLayer.actFun)

        Z = cLayer.Z

        dZ = dA .* eval(:($dActFun($Z)))

    else #in case not every next layer done backprop
        return nothing

    end #if all(i->(i.backCount==cLayer.nextLayers[1].backCount), cLayer.nextLayers)

    cLayer.dW = dZ*cLayer.prevLayer.A' ./m

    if regulization > 0
        if regulization==1
            cLayer.dW .+= (λ/2m)
        else
            cLayer.dW .+= (λ/m) .* cLayer.W
        end
    end

    cLayer.dB = 1/m .* sum(dZ, dims=2)

    cLayer.dA = cLayer.W'dZ

    cLayer.backCount += 1

    return nothing

end #function layerBackProp(cLayer::FCLayer)



function layerBackProp!(cLayer::Activation, model::Model, dA::AoN=nothing; labels=nothing)  where {AoN <: Union{AbstractArray, Nothing}}
    prevLayer = cLayer.prevLayer
    lossFun = model.lossFun

    m = size(cLayer.A)[end]

    regulization, λ = model.regulization, model.λ

    actFun = cLayer.actFun

    dZ = []
    if cLayer.actFun == model.outLayer.actFun

        dZ = layerBackProp(cLayer, model, eval(:($actFun)), labels)

    elseif dA != nothing
        keepProb = cLayer.keepProb
        if keepProb < 1.0 #to save time of multiplication in case keepProb was one
           D = rand(cLayer.channels,1) .< keepProb
           dA .*= D
           dA ./= keepProb
        end

        dActFun = Symbol("d",cLayer.actFun)

        Z = cLayer.Z

        dZ = dA .* eval(:($dActFun($Z)))

    elseif all(i->(i.backCount==cLayer.nextLayers[1].backCount), cLayer.nextLayers)
        dA = []
        for nextLayer in cLayer.nextLayers
            try
                dA .+= nextLayer.dA
            catch e #if first time DimensionMismatch
                dA = nextLayer.dA
            end #try/catch
        end #for

        try
            keepProb = cLayer.keepProb
            if keepProb < 1.0 #to save time of multiplication in case keepProb was one
               D = rand(cLayer.channels,1) .< keepProb
               dA .*= D
               dA ./= keepProb
            end
        catch e

        end

        dActFun = Symbol("d",cLayer.actFun)

        Z = cLayer.prevLayer.A #this is the Activation layer

        dZ = dA .* eval(:($dActFun($Z)))

    else #in case not every next layer done backprop
        return nothing

    end #if all(i->(i.backCount==cLayer.nextLayers[1].backCount), cLayer.nextLayers)

    cLayer.dA = dZ

    cLayer.backCount += 1

    return nothing
end #function layerBackProp!(cLayer::Activation



function layerBackProp!(cLayer::AddLayer, model::Model, dA::AoN=nothing; labels=nothing)  where {AoN <: Union{AbstractArray, Nothing}}

    # m = size(cLayer.A)[end]
    #
    # A = cLayer.A
    #
    # regulization, λ = model.regulization, model.λ
    if dA != nothing
        keepProb = cLayer.keepProb
        if keepProb < 1.0 #to save time of multiplication in case keepProb was one
           D = rand(cLayer.channels,1) .< keepProb
           dA .*= D
           dA ./= keepProb
        end

        dActFun = Symbol("d",cLayer.actFun)

        Z = cLayer.Z

        dZ = dA .* eval(:($dActFun($Z)))
    elseif all(i->(i.backCount==cLayer.nextLayers[1].backCount), cLayer.nextLayers)
        dA = []
        for nextLayer in cLayer.nextLayers
            try
                dA .+= nextLayer.dA
            catch e
                dA = nextLayer.dA #to initialize dA
            end #try/catch
        end #for

        cLayer.dA = dA


    else #in case not every next layer done backprop
        return nothing

    end #if all(i->(i.backCount==cLayer.nextLayers[1].backCount), cLayer.nextLayers)

    cLayer.backCount += 1

    return nothing
end #function layerBackProp!(cLayer::AddLayer



function layerBackProp!(cLayer::Input, model::Model, dA::AoN=nothing; labels=nothing) where {AoN <: Union{AbstractArray, Nothing}}

    # m = size(cLayer.A)[end]
    #
    # A = cLayer.A
    #
    # regulization, λ = model.regulization, model.λ

    if dA != nothing
        cLayer.dA = dA
    elseif all(i->(i.backCount==cLayer.nextLayers[1].backCount), cLayer.nextLayers)
        dA = []
        for nextLayer in cLayer.nextLayers
            try
                dA .+= nextLayer.dA
            catch e
                dA = nextLayer.dA #need to initialize dA
            end #try/catch
        end #for

        cLayer.dA = dA

    else #in case not every next layer done backprop
        return nothing

    end #if all(i->(i.backCount==cLayer.nextLayers[1].backCount), cLayer.nextLayers)

    cLayer.backCount += 1

    return nothing
end #function layerBackProp!(cLayer::Input


function layerBackProp!(cLayer::BatchNorm, model::Model, dA::AoN=nothing; labels=nothing) where {AoN <: Union{AbstractArray, Nothing}}
    prevLayer = cLayer.prevLayer
    lossFun = model.lossFun

    m = size(cLayer.A)[end]

    A = cLayer.A

    normDim = cLayer.dim

    regulization, λ = model.regulization, model.λ
    dZ = []
    if dA != nothing
        dZ = cLayer.W .* dA
    elseif all(i->(i.backCount==cLayer.nextLayers[1].backCount), cLayer.nextLayers)
        dA = []
        for nextLayer in cLayer.nextLayers
            try
                dA .+= nextLayer.dA
            catch e
                dA = nextLayer.dA #need to initialize dA
            end #try/catch
        end #for


        Z = cLayer.Z

        dZ = cLayer.W .* dA

    else #in case not every next layer done backprop
        return nothing

    end #if all(i->(i.backCount==cLayer.nextLayers[1].backCount), cLayer.nextLayers)

    cLayer.dW = sum(dZ .* cLayer.Z, dims=1:normDim)

    if regulization > 0
        if regulization==1
            cLayer.dW .+= (λ/2m)
        else
            cLayer.dW .+= (λ/m) .* cLayer.W
        end
    end

    cLayer.dB = sum(dZ, dims=1:normDim)

    N = prod(size(dZ)[1:normDim])

    varep = cLayer.var .+ cLayer.ϵ

    dẐ = dZ ./ sqrt.(varep) .*
         (.-(cLayer.Ai_μ_s) .* ((N-1)/N^2) ./ (varep).^2 .+ 1) .*
         (1-1/N)

    cLayer.dA = dẐ

    cLayer.backCount += 1

    return nothing

end #function layerBackProp(cLayer::BatchNorm)


export layerBackProp!
