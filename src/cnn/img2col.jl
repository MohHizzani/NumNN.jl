###img2col

function img2col(A::AbstractArray{T,3}) where {T}
    S = size(A)
    vs = prod(S[1:end-1])
    m = S[end]
    return reshape(A, vs,m)
end #function img2col(A::Array{T,3})

function img2col(A::AbstractArray{T,4}) where {T}
    S = size(A)
    vs = prod(S[1:end-1])
    m = S[end]
    return reshape(permutedims(A, [2,1,3,4]), vs,m)
end #function img2col(A::Array{T,4})

function img2row(A::AbstractArray{T,4}) where {T}
    S = size(A)
    vs = prod(S[1:end-1])
    m = S[end]
    return reshape(A, vs, m)
end #function img2col(A::Array{T,4})

function img2col(A::AbstractArray{T,5}) where {T}
    S = size(A)
    vs = prod(S[1:end-1])
    m = S[end]
    return reshape(permutedims(A, [2,1,3,4,5]), vs,m)
end #function img2col(A::Array{T,5})

export img2col

### col2img

function col2img1D(Av::AbstractArray{T,2}, outputS::Tuple{Integer,Integer,Integer}) where{T}
    # S = size(Av)
    # if H < 0
    #     H = S[1]÷c
    # end #if H < 0
    return reshape(Av, outputS)
end #function col2img(Av::Array{T,2}, c::Integer)

function col2img2D(Av::AbstractArray{T,2}, outputS::Tuple{Integer,Integer,Integer,Integer}) where{T}
    H,W,c,m = outputS
    # if H <0 && W<0
    #     H = W = Integer(floor(sqrt(WH)))
    # elseif H < 0
    #     H = WH ÷ W
    # elseif W < 0
    #     W = WH ÷ H
    # end
    return permutedims(reshape(Av, (W,H,c,m)), [2,1,3,4])
end #function col2img(Av::Array{T,2}, c::Integer; H::Integer=-1, W::Integer=-1)

function col2img3D(Av::AbstractArray{T,2}, outputS::Tuple{Integer,Integer,Integer,Integer,Integer}) where{T}
    H,W,D,c,m = outputS
    # S = size(Av)
    # HWD = S[1]÷c
    # if H <0 && W<0 && D<0
    #     H = W = D = Integer(floor((HWD)^(1/3)))
    # elseif H < 0 && W < 0
    #     H = W = Integer(floor(sqrt(HWD÷D)))
    # elseif H < 0 && D < 0
    #     H = D = Integer(floor(sqrt(HWD÷W)))
    # elseif W < 0 && D < 0
    #     W = D = Integer(floor(sqrt(HWD÷H)))
    # elseif H < 0
    #     H = HWD ÷ (W*D)
    # elseif W < 0
    #     W = HWD ÷ (H*D)
    # elseif D < 0
    #     D = HWD ÷ (H*W)
    # end
    return permutedims(reshape(Av, (W,H,D,c,m)), [2,1,3,4,5])
end #function col2img(Av::Array{T,2}, c::Integer; H::Integer=-1, W::Integer=-1, D::Integer=-1)


export col2img1D, col2img2D, col2img3D
