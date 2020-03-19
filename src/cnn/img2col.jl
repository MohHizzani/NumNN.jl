###img2col

function img2col(A::AbstractArray{T,3})::AbstractArray{T,2} where {T}
    S = size(A)
    vs = prod(S[1:end-1])
    m = S[end]
    return reshape(A, vs, m)
end #function img2col(A::Array{T,3})

function img2col(A::AbstractArray{T,4})::AbstractArray{T,2} where {T}
    S = size(A)
    vs = prod(S[1:end-1])
    m = S[end]
    return reshape(permutedims(A, [2, 1, 3, 4]), vs, m)
end #function img2col(A::Array{T,4})

function img2row(A::AbstractArray{T,4})::AbstractArray{T,2} where {T}
    S = size(A)
    vs = prod(S[1:end-1])
    m = S[end]
    return reshape(A, vs, m)
end #function img2col(A::Array{T,4})

function img2col(A::AbstractArray{T,5})::AbstractArray{T,2} where {T}
    S = size(A)
    vs = prod(S[1:end-1])
    m = S[end]
    return reshape(permutedims(A, [2, 1, 3, 4, 5]), vs, m)
end #function img2col(A::Array{T,5})

export img2col

### col2img

function col2img1D(
    Av::AbstractArray{T,2},
    outputS::Tuple{Integer,Integer,Integer},
)::AbstractArray{T,3} where {T}
    return reshape(Av, outputS)
end #function col2img(Av::Array{T,2}, c::Integer)

function col2img2D(
    Av::AbstractArray{T,2},
    outputS::Tuple{Integer,Integer,Integer,Integer},
)::AbstractArray{T,4} where {T}
    H, W, c, m = outputS
    return permutedims(reshape(Av, (W, H, c, m)), [2, 1, 3, 4])
end #function col2img(Av::Array{T,2}, c::Integer; H::Integer=-1, W::Integer=-1)

function col2img3D(
    Av::AbstractArray{T,2},
    outputS::Tuple{Integer,Integer,Integer,Integer,Integer},
)::AbstractArray{T,5} where {T}
    H, W, D, c, m = outputS
    return permutedims(reshape(Av, (W, H, D, c, m)), [2, 1, 3, 4, 5])
end #function col2img(Av::Array{T,2}, c::Integer; H::Integer=-1, W::Integer=-1, D::Integer=-1)


export col2img1D, col2img2D, col2img3D
