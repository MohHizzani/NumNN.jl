using Statistics

###Input Layer
function layerForProp!(cLayer::Input, X::AoN=nothing) where {AoN <: Union{Array, Nothing}}
    if X != nothing
        cLayer(X)
    end
    return nothing
end


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


###BatchNorm

function layerForProp!(cLayer::BatchNorm)
    Ai = cLayer.prevLayer.A
    cLayer.Z = Ai .- mean(Ai, dims=1:cLayer.dim)
    cLayer.Z ./= sqrt.(var(Ai, dims=1:cLayer.dim) .+ cLayer.ϵ)
    cLayer.A = cLayer.W .* cLayer.Z .+ cLayer.B

    return nothing
end #function layerForProp!(cLayer::BatchNorm)

export layerForProp!
