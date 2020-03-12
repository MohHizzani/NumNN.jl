###img2vec


function img2vec(A::Array{T,3}) where {T}
    S = size(A)
    vs = prod(S[1:end-1])
    m = S[end]
    return reshape(A, vs,m)
end #function img2vec(A::Array{T,3})

function img2vec(A::Array{T,4}) where {T}
    S = size(A)
    vs = prod(S[1:end-1])
    m = S[end]
    return reshape(permutedims(A, [2,1,3,4]), vs,m)
end #function img2vec(A::Array{T,4})

function img2vec(A::Array{T,5}) where {T}
    S = size(A)
    vs = prod(S[1:end-1])
    m = S[end]
    return reshape(permutedims(A, [2,1,3,4,5]), vs,m)
end #function img2vec(A::Array{T,5})

export img2vec

### vec2img

function vec2img1D(Av::AbstractArray{T,2}, c::Integer) where{T}
    S = size(Av)
    H = S[1]÷c
    return reshape(Av, (H,c,S[2]))
end #function vec2img(Av::Array{T,2}, c::Integer)

function vec2img2D(Av::AbstractArray{T,2}, c::Integer; H::Integer=-1, W::Integer=-1) where{T}
    S = size(Av)
    WH = S[1]÷c
    if H <0 && W<0
        H = W = Integer(floor(sqrt(WH)))
    elseif H < 0
        H = WH ÷ W
    elseif W < 0
        W = WH ÷ H
    end
    return permutedims(reshape(Av, (W,H,c,S[2])), [2,1,3,4])
end #function vec2img(Av::Array{T,2}, c::Integer; H::Integer=-1, W::Integer=-1)

function vec2img3D(Av::AbstractArray{T,2}, c::Integer; H::Integer=-1, W::Integer=-1, D::Integer=-1) where{T}
    S = size(Av)
    HWD = S[1]÷c
    if H <0 && W<0 && D<0
        H = W = D = Integer(floor((HWD)^(1/3)))
    elseif H < 0 && W < 0
        H = W = Integer(floor(sqrt(HWD÷D)))
    elseif H < 0 && D < 0
        H = D = Integer(floor(sqrt(HWD÷W)))
    elseif W < 0 && D < 0
        W = D = Integer(floor(sqrt(HWD÷H)))
    elseif H < 0
        H = HWD ÷ (W*D)
    elseif W < 0
        W = HWD ÷ (H*D)
    elseif D < 0
        D = HWD ÷ (H*W)
    end
    return permutedims(reshape(Av, (W,H,D,c,S[2])), [2,1,3,4,5])
end #function vec2img(Av::Array{T,2}, c::Integer; H::Integer=-1, W::Integer=-1, D::Integer=-1)


export vec2img
