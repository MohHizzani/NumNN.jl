using Statistics

###Input Layer
function layerForProp(
    cLayer::Input,
    X::AbstractArray = Array{Any,1}(undef,0);
    FCache::Dict{Layer,Dict{Symbol, AbstractArray}},
    kwargs...,
)
    if length(X) != 0
        cLayer.A = X
        cLayer.inputS = cLayer.outputS = size(X)
    end
    cLayer.forwCount += 1
    # Base.GC.gc()
    return Dict(:A=>X)
end


###FCLayer forprop

function layerForProp(
    cLayer::FCLayer,
    Ai::AbstractArray = Array{Any,1}(undef,0);
    FCache::Dict{Layer,Dict{Symbol, AbstractArray}},
    kwargs...,
)
    prevLayer = cLayer.prevLayer
    if length(Ai) == 0
        Ai = FCache[prevLayer][:A]
    end
    cLayer.inputS = cLayer.prevLayer.outputS
    Z = cLayer.W * Ai .+ cLayer.B
    actFun = cLayer.actFun
    # Z = cLayer.Z
    cLayer.outputS = size(Z)
    A = eval(:($actFun($Z)))
    cLayer.forwCount += 1
    # Base.GC.gc()
    return Dict(:Z=>Z, :A=>A)
end #function layerForProp(cLayer::FCLayer)


###AddLayer forprop

function layerForProp(
    cLayer::AddLayer;
    FCache::Dict{Layer,Dict{Symbol, AbstractArray}},
    kwargs...,
    )
    A = similar(FCache[cLayer.prevLayer[1]][:A]) .= 0
    # if all(
    #     i -> (i.forwCount == cLayer.prevLayer[1].forwCount),
    #     cLayer.prevLayer,
    # )
        for prevLayer in cLayer.prevLayer
            A .+= FCache[prevLayer][:A]
        end
    # end #if all
    cLayer.forwCount += 1
    # Base.GC.gc()
    return Dict(:A=>A)
end #function layerForProp(cLayer::AddLayer)

###Activation forprop


function layerForProp(
    cLayer::Activation,
    Ai::AbstractArray = Array{Any,1}(undef,0);
    FCache::Dict{Layer,Dict{Symbol, AbstractArray}},
    kwargs...,
)
    prevLayer = cLayer.prevLayer
    if length(Ai) == 0
        Ai = FCache[prevLayer][:A]
    end
    actFun = cLayer.actFun
    A = eval(:($actFun($Ai)))
    cLayer.inputS = cLayer.outputS = size(Ai)
    cLayer.forwCount += 1
    # Ai = nothing
    # Base.GC.gc()
    return Dict(:A=>A)
end #function layerForProp(cLayer::Activation)


###BatchNorm

function layerForProp(
    cLayer::BatchNorm,
    Ai::AbstractArray = Array{Any,1}(undef,0);
    FCache::Dict{Layer,Dict{Symbol, AbstractArray}},
    kwargs...,
)

    prediction = getindex(kwargs, :prediction; default=false)
    prevLayer = cLayer.prevLayer
    if length(Ai) == 0
        Ai = FCache[prevLayer][:A]
    end

    if prediction
        cLayer.forwCount += 1
        NDim = cLayer.dim
        μ = mean(Ai, dims = 1:NDim)
        Ai_μ = Ai .- μ
        Num = prod(size(Ai)[1:NDim])
        Ai_μ_s = Ai_μ .^ 2
        var = sum(Ai_μ_s, dims = 1:NDim) ./ Num
        Z = Ai_μ ./ sqrt.(var .+ cLayer.ϵ)
        Ao = cLayer.W .* Z .+ cLayer.B
        return Dict(:A => Ao)
    end #prediction

    # initWB!(cLayer)
    # initVS!(cLayer, model.optimizer)

    N = ndims(Ai)
    Ai = permutedims(Ai, [N,(1:N-1)...])

    NDim = cLayer.dim + 1

    μ = mean(Ai, dims = 1:NDim)
    Ai_μ = Ai .- μ
    Num = prod(size(Ai)[1:NDim])
    Ai_μ_s = Ai_μ .^ 2
    var = sum(Ai_μ_s, dims = 1:NDim) ./ Num
    Z = Ai_μ ./ sqrt.(var .+ cLayer.ϵ)
    Ap = cLayer.W .* Z .+ cLayer.B
    Ao = permutedims(Ap, [(2:N)..., 1])
    cLayer.forwCount += 1
    # Ai_μ = nothing
    cLayer.inputS = cLayer.outputS = size(Ao)
    # Base.GC.gc()
    return Dict(
        :μ => μ,
        :Ai_μ => Ai_μ,
        :Ai_μ_s => Ai_μ_s,
        :var => var,
        :Z => Z,
        :A => Ao,
        :Ap => Ap,
    )
end #function layerForProp(cLayer::BatchNorm)

export layerForProp
