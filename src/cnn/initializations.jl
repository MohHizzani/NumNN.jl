

function initWB!(
    cLayer::Conv1D,
    p::Type{T} = Float64::Type{Float64};
    He::Bool = true,
    coef::AbstractFloat = 0.01,
    zro::Bool = false,
) where {T}


    f = cLayer.f
    cl = cLayer.channels
    cl_1 = cLayer.prevLayer.channels

    inputS = cLayer.inputS
    s_H = cLayer.s
    f_H = cLayer.f

    # ## note this is the input to the matmult after padding
    # if cLayer.padding == :same
    #     p_H = Integer(ceil((s_H * (n_Hi - 1) - n_Hi + f_H) / 2))
    #     n_H = n_Hi + 2p_H
    # elseif cLayer.padding == :valid
    #     n_H = n_Hi
    # end

    if He
        coef = sqrt(2 / cl_1)
    end


    if zro
        W = zeros(T, f..., cl_1, cl)
    else
        W = T.(randn(f..., cl_1, cl) .* coef)
    end
    B = zeros(T, repeat([1], length(f) + 1)..., cl)

    cLayer.W, cLayer.B = W, B
    cLayer.K = Matrix{T}(undef,0,0)
    #unroll(cLayer, (n_H, ci, m))
    cLayer.dW = zeros(T, f..., cl_1, cl)
    cLayer.dK = Matrix{T}(undef,0,0)
    #unroll(cLayer, (n_H, ci, m), :dW)
    cLayer.dB = deepcopy(B)
    return nothing
end #initWB

function initWB!(
    cLayer::Conv2D,
    p::Type{T} = Float64::Type{Float64};
    He::Bool = true,
    coef::AbstractFloat = 0.01,
    zro::Bool = false,
) where {T}


    f = cLayer.f
    cl = cLayer.channels
    cl_1 = cLayer.prevLayer.channels

    inputS = cLayer.inputS
    s_H, s_W = cLayer.s
    f_H, f_W = cLayer.f

    ## note this is the input to the matmult after padding
    # if cLayer.padding == :same
    #     p_H = Integer(ceil((s_H * (n_Hi - 1) - n_Hi + f_H) / 2))
    #     p_W = Integer(ceil((s_W * (n_Wi - 1) - n_Wi + f_W) / 2))
    #     n_H, n_W = n_Hi + 2p_H, n_Wi + 2p_W
    # elseif cLayer.padding == :valid
    #     n_H = n_Hi
    #     n_W = n_Wi
    # end

    if He
        coef = sqrt(2 / cl_1)
    end


    if zro
        W = zeros(T, f..., cl_1, cl)
    else
        W = T.(randn(f..., cl_1, cl) .* coef)
    end
    B = zeros(T, repeat([1], length(f) + 1)..., cl)

    cLayer.W, cLayer.B = W, B
    cLayer.K = Matrix{T}(undef,0,0)
    #unroll(cLayer, (n_H, n_W, ci, m))
    cLayer.dW = zeros(T, f..., cl_1, cl)
    cLayer.dK = Matrix{T}(undef,0,0)
    #unroll(cLayer, (n_H, n_W, ci, m), :dW)
    cLayer.dB = deepcopy(B)
    return nothing
end #initWB

function initWB!(
    cLayer::Conv3D,
    p::Type{T} = Float64::Type{Float64};
    He::Bool= true,
    coef::AbstractFloat = 0.01,
    zro::Bool = false,
) where {T}


    f = cLayer.f
    cl = cLayer.channels
    cl_1 = cLayer.prevLayer.channels

    inputS = cLayer.inputS
    s_H, s_W, s_D = cLayer.s
    f_H, f_W, f_D = cLayer.f

    ## note this is the input to the matmult after padding
    # if cLayer.padding == :same
    #     p_H = Integer(ceil((s_H * (n_Hi - 1) - n_Hi + f_H) / 2))
    #     p_W = Integer(ceil((s_W * (n_Wi - 1) - n_Wi + f_W) / 2))
    #     p_D = Integer(ceil((s_D * (n_Di - 1) - n_Di + f_D) / 2))
    #     n_H, n_W, n_D = n_Hi + 2p_H, n_Wi + 2p_W, n_Di + 2p_D
    # elseif cLayer.padding == :valid
    #     n_H = n_Hi
    #     n_W = n_Wi
    #     n_D = n_Di
    # end

    if He
        coef = sqrt(2 / cl_1)
    end


    if zro
        W = zeros(T, f..., cl_1, cl)
    else
        W = T.(randn(f..., cl_1, cl) .* coef)
    end
    B = zeros(T, repeat([1], length(f) + 1)..., cl)

    cLayer.W, cLayer.B = W, B
    cLayer.K = Matrix{T}(undef,0,0)
    #unroll(cLayer, (n_H, n_W, n_D, ci, m))
    cLayer.dW = zeros(T, f..., cl_1, cl)
    cLayer.dK = Matrix{T}(undef,0,0)
    #unroll(cLayer, (n_H, n_W, n_D, ci, m), :dW)
    cLayer.dB = deepcopy(B)
    return nothing
end #initWB


###Pooling Layers

function initWB!(
    cLayer::P,
    p::Type{T} = Float64::Type{Float64};
    He::Bool = true,
    coef::AbstractFloat = 0.01,
    zro::Bool = false,
) where {T,P<:PoolLayer}

    return nothing
end

export initWB!


### initVS!

function initVS!(cLayer::ConvLayer, optimizer::Symbol)

    if optimizer == :adam || optimizer == :momentum
        cLayer.V[:dw] = deepcopy(cLayer.dW)
        cLayer.V̂dk = deepcopy(cLayer.dK)
        cLayer.V[:db] = deepcopy(cLayer.dB)
        if optimizer == :adam
            cLayer.S = deepcopy(cLayer.V)
            cLayer.Ŝdk = deepcopy(cLayer.V̂dk)
        end #if optimizer == :adam
    end #if optimizer == :adam || optimizer == :momentum

    return nothing
end #function initVS!

initVS!(cLayer::P, optimizer::Symbol) where {P<:PoolLayer} = nothing


export initVS!
