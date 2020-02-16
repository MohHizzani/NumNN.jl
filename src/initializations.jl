"""
    initialize W and B for layer with inputs of size of (nl_1) and layer size
        of (nl)

    returns:
        W: of size of (nl, nl_1)
"""
function initWB(
    nl,
    nl_1,
    p::Type{T} = Float64::Type{Float64};
    He = true,
    coef = 0.01,
    zro = false,
) where {T}
    if He
        coef = sqrt(2 / nl_1)
    end
    if zro
        W = zeros(T, (nl, nl_1))
    else
        W = randn(T, nl, nl_1) .* coef
    end
    B = zeros(T, (nl, 1))
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
function deepInitWB!(
    X,
    outLayer::Layer,
    cnt = -1;
    He = true,
    coef = 0.01,
    zro = false,
)
    if cnt < 0
        cnt = outLayer.forwCount + 1
    end
    T = eltype(X)
    prevLayer = outLayer.prevLayer
    forwCount = outLayer.forwCount
    if prevLayer == nothing
        if forwCount < cnt
            outLayer.forwCount += 1
            _w, _b = initWB(
                outLayer.numNodes,
                size(X)[1],
                T;
                He = He,
                coef = coef,
                zro = zro,
            )
            outLayer.W, outLayer.B = _w, _b
            outLayer.dW, outLayer.dB = zeros(T, size(_w)...),
                zeros(T, size(_b)...)

        end #if forwCount < cnt
    elseif isa(outLayer, AddLayer)
        if forwCount < cnt
            outLayer.forwCount += 1
            for prevLayer in outLayer.prevLayer
                    deepInitWB!(
                    X,
                    prevLayer,
                    outLayer.forwCount;
                    He = He,
                    coef = coef,
                    zro = zro,
                )
            end #for prevLayer in outLayer.prevLayer

        end #if forwCount < cnt

    else #if prevLayer == nothing
        if forwCount < cnt
            outLayer.forwCount += 1
            deepInitWB!(
                X,
                prevLayer,
                outLayer.forwCount;
                He = He,
                coef = coef,
                zro = zro,
            )
            _w, _b = initWB(
                outLayer.numNodes,
                prevLayer.numNodes,
                T;
                He = He,
                coef = coef,
                zro = zro,
            )
            outLayer.W, outLayer.B = _w, _b
            outLayer.dW, outLayer.dB = zeros(T, size(_w)...),
                zeros(T, size(_b)...)

        end #if forwCount < cnt

    end #if prevLayer == nothing

    return nothing
end #deepInitWB

export deepInitWB!



function deepInitVS!(outLayer::Layer, optimizer::Symbol, cnt::Integer = -1)

    if cnt < 0
        cnt = outLayer.forwCount + 1
    end

    prevLayer = outLayer.prevLayer

    if optimizer == :adam || optimizer == :momentum
        if prevLayer == nothing
            if outLayer.forwCount < cnt
                outLayer.forwCount += 1
                T = eltype(outLayer.W)
                outLayer.V[:dw] = zeros(T, size(outLayer.W)...)
                outLayer.V[:db] = zeros(T, size(outLayer.B)...)
                if optimizer == :adam
                    outLayer.S[:dw] = zeros(T, size(outLayer.W)...)
                    outLayer.S[:db] = zeros(T, size(outLayer.B)...)
                end #if optimizer == :adam
            end #if outLayer.forwCount < cnt
        elseif isa(outLayer, AddLayer)
            if outLayer.forwCount < cnt
                outLayer.forwCount += 1
                for prevLayer in outLayer.prevLayer
                    deepInitVS!(prevLayer, optimizer, outLayer.forwCount)
                end #for
            end #if outLayer.forwCount < cnt
        else #if prevLayer == nothing
            if outLayer.forwCount < cnt
                outLayer.forwCount += 1
                deepInitVS!(prevLayer, optimizer, outLayer.forwCount)
                T = eltype(outLayer.W)
                outLayer.V[:dw] = zeros(T, size(outLayer.W)...)
                outLayer.V[:db] = zeros(T, size(outLayer.B)...)
                if optimizer == :adam
                    outLayer.S[:dw] = zeros(T, size(outLayer.W)...)
                    outLayer.S[:db] = zeros(T, size(outLayer.B)...)
                end #if optimizer == :adam
            end #if outLayer.forwCount < cnt
        end #if prevLayer == nothing

    end #if optimizer==:adam || optimizer==:momentum

    return nothing
end #function deepInitVS!

export deepInitVS!
