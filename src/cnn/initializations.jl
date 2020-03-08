

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
        W = zeros(T, f..., cl_1, cl)
    else
        W = T.(randn(f..., cl_1, cl) .* coef)
    end
    B = zeros(T, repeat([1], length(f)+1)..., cl)

    cLayer.W, cLayer.B = W, B
    cLayer.dW = zeros(T, f..., cl_1, cl)
    cLayer.dB = deepcopy(B)
    return nothing
end #initWB


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


### initVS!

function initVS!(
    cLayer::ConvLayer,
    optimizer::Symbol
    )

    cLayer.V[:dw] = deepcopy(cLayer.dW)
    cLayer.V[:db] = deepcopy(cLayer.dB)
    if optimizer == :adam
        cLayer.S = deepcopy(cLayer.V)
    end #if optimizer == :adam

    return nothing
end #function initVS!

initVS!(cLayer::P, optimizer::Symbol) where {P <: PoolLayer} = nothing


export initVS!
