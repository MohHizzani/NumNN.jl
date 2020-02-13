"""
    initialize W and B for layer with inputs of size of (nl_1) and layer size
        of (nl)

    returns:
        W: of size of (nl, nl_1)
"""
function initWB(nl, nl_1,
                p::Type{T}=Float64::Type{Float64};
                He=true,
                coef=0.01,
                zro=false) where {T}
    if He
        coef = sqrt(2/nl_1)
    end
    if zro
        W = zeros(T, (nl,nl_1))
    else
        W = randn(T, nl, nl_1) .* coef
    end
    B = zeros(T, (nl,1))
    return W, B
end #initWB

export initWB

"""
    initialize W's and B's using
        X := is the input of the neural Network
        layers := is 1st rank array contains elements of Layer(s) (hidden and output)

    returns:
        W := 1st rank array contains all the W's for each layer
            #Vector{Matrix{T}}
        B := 1st rank array contains all the B's for each layer
            #Vector{Matrix{T}}
"""
function deepInitWB!(X,
                    outLayer::Layer,
                    cnt = -1;
                    He=true,
                    coef=0.01,
                    zro=false)
    if cnt < 0
        cnt = outLayer.forwCount
    end
    T = eltype(X)
    prevLayer = outLayer.prevLayer
    forwCount = outLayer.forwCount
    if prevLayer == nothing
        if forwCount < cnt
        _w, _b = initWB(outLayer.numNodes, size(X)[1],T; He=He, coef=coef, zro=zro)
        outLayer.W, outLayer.B = _w, _b
        outLayer.dW, outLayer.dB = zeros(T, size(_w)...), zeros(T, size(_b)...)
        outLayer.forwCount += 1
        end #if forwCount < cnt
    elseif isa(outLayer, AddLayer) && forwCount < cnt
        deepInitWB!(X, prevLayer, outLayer.forwCount+1;
                    He=He,
                    coef=coef,
                    zro=zro)
        deepInitWB!(X, outLayer.l2, outLayer.forwCount+1;
                    He=He,
                    coef=coef,
                    zro=zro)
        outLayer.forwCount += 1

    else #if prevLayer == nothing
        deepInitWB!(X, prevLayer, outLayer.forwCount+1;
                    He=He,
                    coef=coef,
                    zro=zro)
        _w, _b = initWB(outLayer.numNodes, prevLayer.numNodes,T; He=He, coef=coef, zro=zro)
        outLayer.W, outLayer.B = _w, _b
        outLayer.dW, outLayer.dB = zeros(T, size(_w)...), zeros(T, size(_b)...)
        outLayer.forwCount += 1

    end #if prevLayer == nothing

    return nothing
end #deepInitWB

export deepInitWB!



function deepInitVS(W::Array{Array{T,2},1},
                    B::Array{Array{T,2},1},
                    optimizer::Symbol) where {T<:Number}

    if optimizer==:adam || optimizer==:momentum
        VdW = Array{Array{eltype(W[1]),2},1}([zeros(size(w)) for w in W])
        VdB = Array{Array{eltype(B[1]),2},1}([zeros(size(b)) for b in B])
        if optimizer==:adam
            SdW = Array{Array{eltype(W[1]),2},1}([zeros(size(w)) for w in W])
            SdB = Array{Array{eltype(B[1]),2},1}([zeros(size(b)) for b in B])
        else
            SdW = Array{Array{eltype(W[1]),2},1}(undef,0)
            SdB = Array{Array{eltype(W[1]),2},1}(undef,0)
        end

    else
        SdW = Array{Array{eltype(W[1]),2},1}(undef,0)
        SdB = Array{Array{eltype(W[1]),2},1}(undef,0)
        VdW = Array{Array{eltype(W[1]),2},1}(undef,0)
        VdB = Array{Array{eltype(W[1]),2},1}(undef,0)
    end

    return Dict(:dw=>VdW, :db=>VdB), Dict(:dw=>SdW, :db=>SdB)
end

export deepInitVS
