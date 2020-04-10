using PaddedViews

###
export paddedSize

function paddedSize(cLayer::PL, Ai::AoN=nothing) where {PL<:PaddableLayer, AoN <: Union{AbstractArray, Nothing}}
    if Ai == nothing
        Ai = cLayer.prevLayer.A
    end

    return paddedSize(cLayer, size(Ai))
end #function paddedSize(cLayer::PL, Ai::AoN=nothing) where {PL<:PaddableLayer, AoN <: Union{AbstractArray, Nothing}}

function paddedSize(cLayer::PL, AiS::Tuple) where {PL<:PaddableLayer}
    padS = paddingSize(cLayer, AiS)

    outPS = [i for i in AiS]
    for i=1:(length(AiS)-2)
        outPS[i] += padS[2i-1] + padS[2i]
    end

    return Tuple(outPS)
end #function paddedSize(cLayer::PL, AiS::Tuple) where {PL<:PaddableLayer}

###
export paddingSize

function paddingSize(cLayer::PL, Ai::AoN=nothing) where {PL<:PaddableLayer, AoN <: Union{AbstractArray, Nothing}}
    if Ai == nothing
        Ai = cLayer.prevLayer.A
    end

    return paddingSize(cLayer, size(Ai))
end

"""
    function paddingSize(cLayer::PL, Ai::AbstractArray) where {PL<:PaddableLayer}

Helping function that returns the p_H_hi, p_H_lo, and (in case 2D Conv), p_W_hi, p_W_lo, and so on
"""
function paddingSize(cLayer::PL, AiS::Tuple) where {PL<:PaddableLayer}



    ndim = length(AiS)

    if cLayer.padding == :valid
        return Tuple(repeat([0], (ndim-2)*2))
    end

    if ndim == 3
        n_Hi, ci, m = AiS
        s_H = cLayer.s
        f_H = cLayer.f
        if cLayer.padding == :same
            p_H = s_H * (n_Hi - 1) - n_Hi + f_H

            p_H_hi = p_H ÷ 2 + ((p_H % 2 == 0) ? 0 : 1)
            p_H_lo = p_H ÷ 2

            n_H = n_Hi
            return (p_H_hi, p_H_lo)
        end
    elseif ndim == 4
        n_Hi, n_Wi, ci, m = AiS
        (s_H, s_W) = cLayer.s
        (f_H, f_W) = cLayer.f
        if cLayer.padding == :same
            p_H = (s_H * (n_Hi - 1) - n_Hi + f_H)
            p_W = (s_W * (n_Wi - 1) - n_Wi + f_W)

            p_H_hi = p_H ÷ 2 + ((p_H % 2 == 0) ? 0 : 1)
            p_H_lo = p_H ÷ 2
            p_W_hi = p_W ÷ 2 + ((p_W % 2 == 0) ? 0 : 1)
            p_W_lo = p_W ÷ 2

            n_H, n_W = n_Hi, n_Wi
            return (p_H_hi, p_H_lo, p_W_hi, p_W_lo)
        end

    elseif ndim == 5
        n_Hi, n_Wi, n_Di, ci, m = AiS
        s_H, s_W, s_D = cLayer.s
        f_H, f_W, f_D = cLayer.f
        if cLayer.padding == :same
            p_H = (s_H * (n_Hi - 1) - n_Hi + f_H)
            p_W = (s_W * (n_Wi - 1) - n_Wi + f_W)
            p_D = (s_D * (n_Di - 1) - n_Di + f_D)

            p_H_hi = p_H ÷ 2 + ((p_H % 2 == 0) ? 0 : 1)
            p_H_lo = p_H ÷ 2
            p_W_hi = p_W ÷ 2 + ((p_W % 2 == 0) ? 0 : 1)
            p_W_lo = p_W ÷ 2
            p_D_hi = p_D ÷ 2 + ((p_D % 2 == 0) ? 0 : 1)
            p_D_lo = p_D ÷ 2

            n_H, n_W, n_D = n_Hi, n_Wi, n_Di
            return (p_H_hi, p_H_lo, p_W_hi, p_W_lo, p_D_hi, p_D_lo)
        end #if cLayer.padding == :same
    end

end #function paddingSize(cLayer::CL, AiS::Tuple)

###Padding functions

function padding(
    cLayer::P,
    Ai::AoN = nothing,
) where {P<:PaddableLayer,AoN<:Union{AbstractArray,Nothing}}

    ndim = Ai

    if Ai == nothing
        Ai = cLayer.prevLayer.A
    end

    if cLayer.padding == :same
        Ai = padding(Ai, paddingSize(cLayer, Ai)...)
    elseif cLayer.padding == :valid
        Ai = Ai
    end

    return Ai

end #function padding(cLayer::P) where {P <: PaddableLayer}




"""
    function padding(Ai::AbstractArray{T,4},
                     p_H::Integer,
                     p_W::Integer=-1) where {T}

pad zeros to the Array Ai with amount of p values

inputs:
    Ai := Array of type T and dimension N
    p  := integer determinde the amount of zeros padding
          i.e.
          if Ai is a 3-dimensional array the padding will be for the first
              dimension
          if Ai is a 4-dimensional array the padding will be for the first 2
              dimensions
          if Ai is a 5-dimensional array the padding will be for the first 3
              dimensions

output:
    PaddinView array where it contains the padded values and the original
        data without copying it
"""
function padding(
    Ai::AbstractArray{T,4},
    p_H_hi::Integer,
    p_H_lo::Integer,
    p_W_hi::Integer,
    p_W_lo::Integer,
) where {T}

    n_H, n_W, c, m = size(Ai)
    return PaddedView(
        0,
        Ai,
        (p_H_hi + p_H_lo + n_H, p_W_hi + p_W_lo + n_W, c, m),
        (1 + p_H_hi, 1 + p_W_hi, 1, 1),
    )
end #function padding(Ai::Array{T,4}


function padding(
    Ai::AbstractArray{T,3},
    p_H_hi::Integer,
    p_H_lo::Integer,
) where {T}

    n_H, c, m = size(Ai)
    return PaddedView(0, Ai, (p_H_hi + p_H_lo + n_H, c, m), (1 + p_H_hi, 1, 1))
end #function padding(Ai::Array{T,3}


function padding(
    Ai::AbstractArray{T,5},
    p_H_hi::Integer,
    p_H_lo::Integer,
    p_W_hi::Integer,
    p_W_lo::Integer,
    p_D_hi::Integer,
    p_D_lo::Integer,
) where {T}

    n_H, n_W, n_D, c, m = size(Ai)
    return PaddedView(
        0,
        Ai,
        (
            p_H_hi + p_H_lo + n_H,
            p_W_hi + p_W_lo + n_W,
            p_D_hi + p_D_lo + n_D,
            c,
            m,
        ),
        (1 + p_H_hi, 1 + p_W_hi, 1 + p_D_hi, 1, 1),
    )
end #function padding(Ai::Array{T,5}

export padding
