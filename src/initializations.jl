"""
    initialize W and B for layer with inputs of size of (nl_1) and layer size
        of (nl)

    returns:
        W: of size of (nl, nl_1)
"""
function initWB(
    cLayer::FCLayer,
    p::Type{T} = Float64::Type{Float64};
    He = true,
    coef = 0.01,
    zro = false,
) where {T}

    s = (cLayer.numNodes, cLayer.prevLayer.numNodes)
    if He
        coef = sqrt(2 / s[end])
    end
    if zro
        W = zeros(T, s...)
    else
        W = randn(T, s...) .* coef
    end
    B = zeros(T, (s[1], 1))
    return W, B
end #initWB


function initWB(
    cLayer::CL,
    p::Type{T} = Float64::Type{Float64};
    He = true,
    coef = 0.01,
    zro = false,
) where {T,CL<:ConvLayer}


    f = cLayer.f
    cl = cLayer.channels
    cl_1 = cLayer.prevLayer.channels
    if He
        coef = sqrt(2 / cl_1)
    end


    if zro
        W = [zeros(T, f..., cl_1) for i = 1:cl]
    else
        W = [randn(T, f..., cl_1) .* coef for i = 1:cl]
    end
    B = zeros(T, repeat([1], length(f))..., cl)
    return W, B
end #initWB

export initWB

"""
    initialize W's and B's using

    inputs:
        X := is the input of the neural Network
        outLayer := is the output Layer or the current layer
                    of initialization
        cnt := is a counter to determinde the current step
                and its an internal variable


        kwargs:
            He := is a true/false array, whether to use the He **et al.** initialization
                    or not

            coef := when not using He **et al.** initialization use this coef
                    to multiply with the random numbers initialization
            zro := true/false variable whether to initialize W with zeros or not

"""
function deepInitWB!(
    X,
    outLayer::Layer,
    cnt = -1;
    He = true,
    coef = 0.01,
    zro = false,
    dtype = nothing,
)
    if cnt < 0
        cnt = outLayer.forwCount + 1
    end

    if dtype == nothing
        T = eltype(X)
    else
        T = dtype
    end

    prevLayer = outLayer.prevLayer
    forwCount = outLayer.forwCount
    if prevLayer isa Input
        if forwCount < cnt
            outLayer.forwCount += 1
            outLayer.W, outLayer.B =
                initWB(outLayer, T; He = He, coef = coef, zro = zro)

            outLayer.dW, outLayer.dB = initWB(outLayer, T; zro = true)

        end #if forwCount < cnt
    elseif isa(outLayer, AddLayer)
        if forwCount < cnt
            outLayer.forwCount += 1
            for prevLayer in outLayer.prevLayer
                deepInitWB!(
                    X,
                    prevLayer,
                    cnt;
                    He = He,
                    coef = coef,
                    zro = zro,
                    dtype = dtype,
                )
            end #for prevLayer in outLayer.prevLayer

        end #if forwCount < cnt

    else #if prevLayer == nothing
        if forwCount < cnt
            outLayer.forwCount += 1
            deepInitWB!(
                X,
                prevLayer,
                cnt;
                He = He,
                coef = coef,
                zro = zro,
                dtype = dtype,
            )

            outLayer.W, outLayer.B =
                initWB(outLayer, T; He = He, coef = coef, zro = zro)
            outLayer.dW, outLayer.dB = initWB(outLayer, T; zro = zro)

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
        if prevLayer isa Input
            if outLayer.forwCount < cnt
                outLayer.forwCount += 1
                outLayer.V[:dw] = deepcopy(outLayer.dW)
                outLayer.V[:db] = deepcopy(outLayer.dB)
                if optimizer == :adam
                    outLayer.S = deepcopy(outLayer.V)
                end #if optimizer == :adam
            end #if outLayer.forwCount < cnt
        elseif isa(outLayer, AddLayer)
            if outLayer.forwCount < cnt
                outLayer.forwCount += 1
                for prevLayer in outLayer.prevLayer
                    deepInitVS!(prevLayer, optimizer, cnt)
                end #for
            end #if outLayer.forwCount < cnt
        else #if prevLayer == nothing
            if outLayer.forwCount < cnt
                outLayer.forwCount += 1
                deepInitVS!(prevLayer, optimizer, cnt)
                outLayer.V[:dw] = deepcopy(outLayer.dW)
                outLayer.V[:db] = deepcopy(outLayer.dB)
                if optimizer == :adam
                    outLayer.S = deepcopy(outLayer.V)
                end #if optimizer == :adam
            end #if outLayer.forwCount < cnt
        end #if prevLayer == nothing

    end #if optimizer==:adam || optimizer==:momentum

    return nothing
end #function deepInitVS!

export deepInitVS!
