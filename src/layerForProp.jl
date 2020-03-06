using Statistics

###Input Layer
function layerForProp!(cLayer::Input, X::AoN=nothing) where {AoN <: Union{Array, Nothing}}
    if X != nothing
        cLayer(X)
    end
    cLayer.forwCount += 1
    return nothing
end


###FCLayer forprop

function layerForProp!(cLayer::FCLayer)
    cLayer.Z = cLayer.W * cLayer.prevLayer.A .+ cLayer.B
    actFun = cLayer.actFun
    Z = cLayer.Z
    cLayer.A = eval(:($actFun($Z)))
    cLayer.forwCount += 1
    return nothing
end #function layerForProp!(cLayer::FCLayer)


###AddLayer forprop

function layerForProp!(cLayer::AddLayer)
    cLayer.A = similar(cLayer.prevLayer[1].A)
    cLayer.A .= 0
    for prevLayer in cLayer.prevLayer
        cLayer.A .+= prevLayer.A
    end
    cLayer.forwCount += 1
    return nothing
end #function layerForProp!(cLayer::AddLayer)

###Activation forprop


function layerForProp!(cLayer::Activation)
    actFun = cLayer.actFun
    Ai = cLayer.prevLayer.A
    cLayer.A = eval(:($actFun($Ai)))
    cLayer.forwCount += 1
    return nothing
end #function layerForProp!(cLayer::Activation)


###BatchNorm

function layerForProp!(cLayer::BatchNorm)

    initWB!(cLayer)

    Ai = cLayer.prevLayer.A
    cLayer.μ = mean(Ai, dims=1:cLayer.dim)
    cLayer.Ai_μ = Ai .- cLayer.μ
    N = prod(size(Ai)[1:cLayer.dim])
    cLayer.Ai_μ_s = cLayer.Ai_μ .^ 2
    cLayer.var = sum(cLayer.Ai_μ_s, dims=1:cLayer.dim) ./ N
    cLayer.Z = cLayer.Ai_μ ./ sqrt.(cLayer.var .+ cLayer.ϵ)
    cLayer.A = cLayer.W .* cLayer.Z .+ cLayer.B
    cLayer.forwCount += 1
    return nothing
end #function layerForProp!(cLayer::BatchNorm)

export layerForProp!
