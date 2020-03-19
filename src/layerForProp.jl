using Statistics

###Input Layer
function layerForProp!(
    cLayer::Input,
    X::AoN = nothing,
) where {AoN<:Union{Array,Nothing}}
    if X != nothing
        cLayer.A = X
        cLayer.inputS = cLayer.outputS = size(X)
    end
    cLayer.forwCount += 1
    # Base.GC.gc()
    return nothing
end


###FCLayer forprop

function layerForProp!(
    cLayer::FCLayer,
    Ai::AoN,
) where {AoN<:Union{AbstractArray,Nothing}}
    if Ai == nothing
        Ai = cLayer.prevLayer.A
    end
    cLayer.inputS = cLayer.prevLayer.outputS
    cLayer.Z = cLayer.W * Ai .+ cLayer.B
    actFun = cLayer.actFun
    Z = cLayer.Z
    cLayer.outputS = size(Z)
    cLayer.A = eval(:($actFun($Z)))
    cLayer.forwCount += 1
    # Base.GC.gc()
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
    # Base.GC.gc()
    return nothing
end #function layerForProp!(cLayer::AddLayer)

###Activation forprop


function layerForProp!(
    cLayer::Activation,
    Ai::AoN,
) where {AoN<:Union{AbstractArray,Nothing}}
    if Ai == nothing
        Ai = cLayer.prevLayer.A
    end
    actFun = cLayer.actFun
    cLayer.A = eval(:($actFun($Ai)))
    cLayer.inputS = cLayer.outputS = size(Ai)
    cLayer.forwCount += 1
    # Ai = nothing
    # Base.GC.gc()
    return nothing
end #function layerForProp!(cLayer::Activation)


###BatchNorm

function layerForProp!(
    cLayer::BatchNorm,
    Ai::AoN,
) where {AoN<:Union{AbstractArray,Nothing}}
    if Ai == nothing
        Ai = cLayer.prevLayer.A
    end

    initWB!(cLayer)

    cLayer.μ = mean(Ai, dims = 1:cLayer.dim)
    Ai_μ = Ai .- cLayer.μ
    N = prod(size(Ai)[1:cLayer.dim])
    cLayer.Ai_μ_s = Ai_μ .^ 2
    cLayer.var = sum(cLayer.Ai_μ_s, dims = 1:cLayer.dim) ./ N
    cLayer.Z = Ai_μ ./ sqrt.(cLayer.var .+ cLayer.ϵ)
    cLayer.A = cLayer.W .* cLayer.Z .+ cLayer.B
    cLayer.forwCount += 1
    Ai_μ = nothing
    cLayer.inputS = cLayer.outputS = size(Ai)
    # Base.GC.gc()
    return nothing
end #function layerForProp!(cLayer::BatchNorm)

export layerForProp!
