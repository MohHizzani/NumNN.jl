
### single layer initWB

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

    cLayer.W, cLayer.B = W, B
    cLayer.dW = zeros(T, s...)
    cLayer.dB = deepcopy(B)
    return nothing
end #initWB


function initWB!(
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

    cLayer.W, cLayer.B = W, B
    cLayer.dW = [zeros(T, f..., cl_1) for i = 1:cl]
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
    cLayer::Input,
    p::Type{T} = Float64::Type{Float64};
    He = true,
    coef = 0.01,
    zro = false,
) where {T}

    return nothing
end


function initWB!(
    cLayer::BatchNorm,
    p::Type{T} = Float64::Type{Float64};
    He = true,
    coef = 0.01,
    zro = false,
) where {T}

    cn = 0
    try
        cn = cLayer.prevLayer.channels
        if cn == 0
            try
                cn = cLayer.prevLayer.numNodes
            catch e1

            end
        end #if cn ==0
    catch e
        cn = cLayer.prevLayer.numNodes
    end #try/catch
    if He
        coef = sqrt(2 / cn)
    end
    N = ndims(cLayer.prevLayer.A)
    normDim = cLayer.dim
    S = size(cLayer.prevLayer.A)
    paramS = Array{Integer,1}(undef,0)
    for i=1:N
        if S[i] < 1 || i <= normDim
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

function initWB!(
    cLayer::P,
    p::Type{T} = Float64::Type{Float64};
    He = true,
    coef = 0.01,
    zro = false,
) where {T, P <: PoolLayer}

    return nothing
end

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
    if outLayer isa Input
        if forwCount < cnt
            outLayer.forwCount += 1
            initWB!(outLayer, T; He = He, coef = coef, zro = zro)
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

            initWB!(outLayer, T; He = He, coef = coef, zro = zro)
        end #if forwCount < cnt

    end #if prevLayer == nothing

    return nothing
end #deepInitWB

export deepInitWB!


### single layer initVS


function initVS!(
    cLayer::FoC,
    optimizer::Symbol
    ) where {FoC <: Union{FCLayer, <:ConvLayer}}

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
    ) where {IoA <: Union{Input, Activation}}

    return nothing
end #

function initVS!(
    cLayer::BatchNorm,
    optimizer::Symbol,
    )

    T = typeof(cLayer.W)
    cLayer.V = Dict(:dw=>deepcopy(cLayer.dW), :db=>deepcopy(cLayer.dB))
    cLayer.S = deepcopy(cLayer.V)

    return nothing
end

initVS!(cLayer::P, optimizer::Symbol) where {P <: PoolLayer} = nothing

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
                initVS!(outLayer, optimizer)
            end #if outLayer.forwCount < cnt
        end #if prevLayer == nothing

    end #if optimizer==:adam || optimizer==:momentum

    return nothing
end #function deepInitVS!

export deepInitVS!
