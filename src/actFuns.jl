"""
    return the Sigmoid output
    inputs must be matices
"""
σ(X,W,B) = 1 ./ (1 .+ exp.(.-(W*X .+ B)))
σ(Z)  = 1 ./ (1 .+ exp.(.-Z))

export σ

"""
    return the derivative of Sigmoid function
"""
dσ(Z) = σ(Z) .* (1 .- σ(Z))

export dσ

"""
    return the ReLU output
"""
function relu(Z::Array{T,N}) where {T,N}
    max.(zero(T), Z)
end #function relu(Z::Array{T,N}) where {T,N}

export relu

"""
    return the derivative of ReLU function
"""
function drelu(Z::Array{T,N}) where {T,N}
    return T.(Z .> zero(T))
end #function drelu(z::Array{T,N}) where {T,N}

export drelu


"""
    compute the softmax function

"""
function softmax(Ŷ, dim=1)
    Ŷ_exp = exp.(Ŷ)
    sumofexp = sum(Ŷ_exp, dims=dim)
    return Ŷ_exp./sumofexp
end #softmax

function dsoftmax(Ŷ,dim=1)
    sŶ = softmax(Ŷ)
    T = eltype(Ŷ)
    softMat = Array{T,3}(undef,0,0,0)
    sSize = size(Ŷ)[dim]
    for c in eachcol(sŶ)
        tmpMat = zeros(T, sSize, sSize)
        for i=1:length(c)
            for j=1:length(c)
                if i==j
                    tmpMat[i,j] = c[i] * (1-c[j])
                else
                    tmpMat[i,j] = -c[i]*c[j]
                end
            end
        end
        softMat = cat(softMat, tmpMat, dims=3)
    end

    return softMat
end #dsoftmax

export softmax, dsoftmax


Base.tanh(Z::Array{T,N}) where {T,N} = tanh.(Z)

dtanh(Z::Array{T,N}) where {T,N} = 1 .- (tanh.(Z)).^2

export dtanh, tanh


function noAct(Z)
    return Z
end

function dnoAct(Z)
    return ones(eltype(Z), size(Z)...)
end

export noAct, dnoAct
