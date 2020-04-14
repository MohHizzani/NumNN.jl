
### single layer initWB

#TODO create initializers dependent for each method

"""
    initialize W and B for layer with inputs of size of (nl_1) and layer size
        of (nl)

    returns:
        W: of size of (nl, nl_1)
"""
function initWB!(
    cLayer::FCLayer,
    p::Type{T} = Float64::Type{Float64};
    He = true,
    coef = 0.01,
    zro = false,
) where {T}

    s = (cLayer.channels, cLayer.prevLayer.channels)
    if He
        coef = sqrt(2 / s[end])
    end
    if zro
        W = zeros(T, s...)
    else
        W = T.(randn(s...) .* coef)
    end
    B = zeros(T, (s[1], 1))

    cLayer.W, cLayer.B = W, B
    cLayer.dW = zeros(T, s...)
    cLayer.dB = deepcopy(B)
    return nothing
end #initWB

function initWB!(
    cLayer::Activation,
    p::Type{T} = Float64::Type{Float64};
    He = true,
    coef = 0.01,
    zro = false,
) where {T}

    return nothing
end


function initWB!(
    cLayer::Flatten,
    p::Type{T} = Float64::Type{Float64};
    He = true,
    coef = 0.01,
    zro = false,
) where {T}

    return nothing
end

function initWB!(
    cLayer::ConcatLayer,
    p::Type{T} = Float64::Type{Float64};
    He = true,
    coef = 0.01,
    zro = false,
) where {T}

    return nothing
end

function initWB!(
    cLayer::Input,
    p::Type{T} = Float64::Type{Float64};
    He = true,
    coef = 0.01,
    zro = false,
) where {T}

    # cLayer.inputS = cLayer.outputS = size(cLayer.A)
    return nothing
end


function initWB!(
    cLayer::BatchNorm,
    p::Type{T} = Float64::Type{Float64};
    He = true,
    coef = 0.01,
    zro = false,
) where {T}


    cn = cLayer.prevLayer.channels

    if He
        coef = sqrt(2 / cn)
    end
    N = length(cLayer.prevLayer.inputS)
    normDim = cLayer.dim
    #bring the batch size into front
    S = (cLayer.prevLayer.outputS[end], cLayer.prevLayer.outputS[1:end-1]...)
    paramS = Array{Integer,1}([1])
    for i=2:N
        if S[i] < 1 || (i-1) <= normDim
            push!(paramS, 1)
        else
            push!(paramS, S[i])
        end #if S[i] < 1
    end #for

    if !zro
        W = T.(randn(paramS...) .* coef)
    else
        W = zeros(T, paramS...)
    end

    if !(size(cLayer.W) == size(W))
        cLayer.W = W
        cLayer.dW = zeros(T, paramS...)
        cLayer.B = zeros(T, paramS...)
        cLayer.dB = zeros(T, paramS...)

    end #if !(size(cLayer.W) == size(W))
    return nothing
end #function initWB!(cLayer::BatchNorm


export initWB!



### deepInitWB!

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
    outLayer::Layer,
    cnt = -1;
    He = true,
    coef = 0.01,
    zro = false,
    dtype = Float64,
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
    if outLayer isa Input
        if forwCount < cnt
            outLayer.forwCount += 1
            initWB!(outLayer, T; He = He, coef = coef, zro = zro)
        end #if forwCount < cnt
    elseif isa(outLayer, MILayer)
        if forwCount < cnt
            outLayer.forwCount += 1
            for prevLayer in outLayer.prevLayer
                deepInitWB!(
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
                prevLayer,
                cnt;
                He = He,
                coef = coef,
                zro = zro,
                dtype = dtype,
            )

            initWB!(outLayer, T; He = He, coef = coef, zro = zro)
        end #if forwCount < cnt

    end #if prevLayer == nothing

    return nothing
end #deepInitWB

export deepInitWB!


### single layer initVS


function initVS!(
    cLayer::FCLayer,
    optimizer::Symbol
    )

    cLayer.V[:dw] = deepcopy(cLayer.dW)
    cLayer.V[:db] = deepcopy(cLayer.dB)
    if optimizer == :adam
        cLayer.S = deepcopy(cLayer.V)
    end #if optimizer == :adam

    return nothing
end #function initVS!

function initVS!(
    cLayer::IoA,
    optimizer::Symbol
    ) where {IoA <: Union{Input, Activation, Flatten, ConcatLayer}}

    return nothing
end #

function initVS!(
    cLayer::BatchNorm,
    optimizer::Symbol,
    )

    T = typeof(cLayer.W)
    if (! haskey(cLayer.V, :dw)) || (size(cLayer.V[:dw]) != size(cLayer.dW))
        cLayer.V = Dict(:dw=>deepcopy(cLayer.dW), :db=>deepcopy(cLayer.dB))
        cLayer.S = deepcopy(cLayer.V)
    end


    return nothing
end

export initVS!

### deepInitVS!

function deepInitVS!(outLayer::Layer, optimizer::Symbol, cnt::Integer = -1)

    if cnt < 0
        cnt = outLayer.forwCount + 1
    end

    prevLayer = outLayer.prevLayer

    if optimizer == :adam || optimizer == :momentum
        if outLayer isa Input
            if outLayer.forwCount < cnt
                outLayer.forwCount += 1
                initVS!(outLayer, optimizer)
            end #if outLayer.forwCount < cnt
        elseif isa(outLayer, MILayer)
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
                initVS!(outLayer, optimizer)
            end #if outLayer.forwCount < cnt
        end #if prevLayer == nothing

    end #if optimizer==:adam || optimizer==:momentum

    return nothing
end #function deepInitVS!

export deepInitVS!
