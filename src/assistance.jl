
@doc raw"""
    oneHot(Y; classes = [], numC = 0)

convert array of integer classes into one Hot coding.

# Arguments

- `Y` := a vector of classes as a number

- `classes` := the classes explicity represented (in case not all the classes are present in the labels given)

- `numC` := number of classes as alternative to `classes` variable

# Examples

```julia
Y = rand(0:9, 100); # a 100 item with class of [0-9]
"""
function oneHot(Y; classes = [], numC = 0)
    if numC > 0
        c = numC
        Cs = sort(unique(Y))
    elseif length(classes) > 0
        Cs = sort(classes)
        c = length(Cs)
    else
        Cs = sort(unique(Y))
        c = length(Cs)
    end
    hotY = BitArray{2}(undef, c, length(Y))
    @simd for i=1:length(Y)
        hotY[:,i] .= (Cs .== Y[i])
    end
    return hotY
end #oneHot

export oneHot


@doc raw"""
    resetCount!(outLayer::Layer, cnt::Symbol)

to reset a counter in all layers under `outLayer`.

# Arguments

- `outLayer::Layer` := the layer from start reseting the counter

- `cnt::Symbol` := the counter to be reseted

# Examples

```julia
X_train = rand(128, 100);

X_Input = Input(X_train);
X = FCLayer(50, :relu)(X_Input);
X_out = FCLayer(10, :softmax)(X);

FCache = chainForProp(X_train, X_Input);

# Now to reset the forwCount in all layers

resetCount!(X_out, :forwCount)
```

"""
function resetCount!(outLayer::Layer,
                     cnt::Symbol)
    prevLayer = outLayer.prevLayer
    if outLayer isa Input
        # if outLayer.forwCount != 0
            eval(:($outLayer.$cnt = 0))
        # end #if outLayer.forwCount != 0
    elseif isa(outLayer, MILayer) #if prevLayer == nothing
        for prevLayer in outLayer.prevLayer
            resetCount!(prevLayer, cnt)
        end #for
        eval(:($outLayer.$cnt = 0))
    else #if prevLayer == nothing
        resetCount!(prevLayer, cnt)
        eval(:($outLayer.$cnt = 0))
    end #if prevLayer == nothing
    return nothing
end #function resetForwCount

export resetCount!


### to extend the getindex fun

@doc raw"""
    getindex(it, key; default) = haskey(it, key) ? it[key] : default

# Examples

```julia
D = Dict(:A=>"A", :B=>"B")

A = getindex(D, :A)

## this will return an error
#C = getindex(D: :C)

#instead
C = getindex(D, :C; default="C")
#this will return the `String` C
```
"""
Base.getindex(it, key; default) = haskey(it, key) ? it[key] : default


#### getLayerSlice

export getLayerSlice

function getLayerSlice(cLayer::Layer, nextLayer::ConcatLayer, BCache::Dict{Layer, Dict{Symbol, AbstractArray}})
    N = ndims(BCache[nextLayer][:dA])
    fAx = axes(BCache[nextLayer][:dA])[1:end-2]
    lAx = axes(BCache[nextLayer][:dA])[end]
    LSlice = nextLayer.LSlice[cLayer]
    return BCache[nextLayer][:dA][fAx...,LSlice,lAx]
end #function getLayerSlice(cLayer::Layer, nextLayer::ConcatLayer

"""
    getLayerSlice(cLayer::Layer, nextLayer::Layer, BCache::Dict{Layer, Dict{Symbol, AbstractArray}})

Fall back method for  `Layer`s other than `ConcatLayer`

"""
function getLayerSlice(cLayer::Layer, nextLayer::Layer, BCache::Dict{Layer, Dict{Symbol, AbstractArray}})
    return BCache[nextLayer][:dA]
end #function getLayerSlice(cLayer::Layer, nextLayer::Layer


###

export chParamType

function chParamType!(cLayer::Layer, T::DataType, cnt::Integer = -1)
    if cnt < 0
        cnt = cLayer.updateCount + 1
    end

    if length(cLayer.nextLayers) == 0
        if cLayer.updateCount < cnt
            try
                cLayer.W = T.(cLayer.W)
                cLayer.B = T.(cLayer.B)
                cLayer.updateCount += 1
            catch

            end
        end #if cLayer.forwCount < cnt
        return nothing
    elseif isa(cLayer, MILayer) #if typeof(cLayer)==AddLayer
        if all(
            i -> (i.updateCount == cLayer.prevLayer[1].udpateCount),
            cLayer.prevLayer,
        )
            try
                cLayer.W = T.(cLayer.W)
                cLayer.B = T.(cLayer.B)
                cLayer.updateCount += 1
            catch

            end
            for nextLayer in cLayer.nextLayers
                chParamType!(nextLayer, T, cnt)
            end
        end #if all

        return FCache
    else #if cLayer.prevLayer==nothing
        if cLayer.forwCount < cnt
            if cLayer isa Input
                try
                    cLayer.W = T.(cLayer.W)
                    cLayer.B = T.(cLayer.B)
                    cLayer.updateCount += 1
                catch

                end
            else
                try
                    cLayer.W = T.(cLayer.W)
                    cLayer.B = T.(cLayer.B)
                    cLayer.updateCount += 1
                catch

                end
            end
            for nextLayer in cLayer.nextLayers
                chParamType!(nextLayer, T, cnt)
            end

        end #if cLayer.forwCount < cnt
        return nothing
    end #if cLayer.prevLayer!=nothing

    return nothing


end #function chParamType(cLayer::Layer, T::DataType)
