###Padding functions

function padding(
                 cLayer::P,
                 Ai::AoN=nothing,
                 ) where {P <: PaddableLayer, AoN <: Union{AbstractArray, Nothing}}

    ndim = ndims(cLayer.A)

    if Ai == nothing
        Ai = cLayer.prevLayer.A
    end

    if ndim == 3
        n_Hi, ci, m = size(Ai)
        s_H = cLayer.s
        f_H = cLayer.f
        if cLayer.padding == :same
            p_H = Integer(ceil((s_H*(n_Hi-1)-n_Hi+f_H)/2))
            n_H = n_Hi
            Ai = padding(Ai, p_H)
        elseif cLayer.padding == :valid
            Ai = Ai
        end
    elseif ndim == 4
        n_Hi, n_Wi, ci, m = size(Ai)
        s_H, s_W = cLayer.s
        f_H, f_W = cLayer.f
        if cLayer.padding == :same
            p_H = Integer(ceil((s_H*(n_Hi-1)-n_Hi+f_H)/2))
            p_W = Integer(ceil((s_W*(n_Wi-1)-n_Wi+f_W)/2))
            n_H, n_W = n_Hi, n_Wi
            Ai = padding(Ai, p_H, p_W)
        elseif cLayer.padding == :valid
            Ai = Ai
        end

    elseif ndim == 5
        n_Hi, n_Wi, n_Di, ci, m = size(Ai)
        s_H, s_W, s_D = cLayer.s
        f_H, f_W, f_D = cLayer.f
        if cLayer.padding == :same
            p_H = Integer(ceil((s_H*(n_Hi-1)-n_Hi+f_H)/2))
            p_W = Integer(ceil((s_W*(n_Wi-1)-n_Wi+f_W)/2))
            p_D = Integer(ceil((s_D*(n_Di-1)-n_Di+f_D)/2))
            n_H, n_W, n_D = n_Hi, n_Wi, n_D
            Ai = padding(Ai, p_H, p_W, p_D)
        elseif cLayer.padding == :valid
            Ai = Ai
        end #if cLayer.padding == :same
    end
    return Ai

end #function padding(cLayer::P) where {P <: PaddableLayer}




"""
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
function padding(Ai::Array{T,4},
                 p_H::Integer,
                 p_W::Integer=-1) where {T}

    if p_W < 0
        p_W = p_H
    end
    n_H, n_W, c, m = size(Ai)
    return PaddedView(0, Ai, (2*p_H+n_H, 2*p_W+n_W, c, m), (1+p_H, 1+p_W, 1, 1))
end #function padding(Ai::Array{T,4}


function padding(Ai::Array{T,3},
                 p::Integer) where {T}

    n_H, c, m = size(Ai)
    return PaddedView(0, Ai, (2*p+n_H, c, m), (1+p, 1, 1))
end #function padding(Ai::Array{T,3}


function padding(Ai::Array{T,5},
                 p_H::Integer,
                 p_W::Integer=-1,
                 p_D::Integer=-1) where {T}

    if p_W < 0
        p_W = p_H
    end
    if p_D < 0
        p_D = p_H
    end
    n_H, n_W, n_D, c, m = size(Ai)
    return PaddedView(0, Ai,
                        (2*p_H+n_H, 2*p_W+n_W, 2*p_D+n_D, c, m),
                        (1+p_H, 1+p_W, 1+p_D, 1, 1))
end #function padding(Ai::Array{T,5}

export padding
