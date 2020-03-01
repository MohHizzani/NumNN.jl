
abstract type actFun end

export actFun






### sigmoid

abstract type σ <: actFun end

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


function predict(
    actFun::Type{σ},
    probs::Array{T, N},
    labels::Aa=nothing
    ) where {Aa <: Union{<:AbstractArray, Nothing}, T, N}

    s = size{probs}
    Ŷ_bool = probs .> T(0.5)
    if labels isa AbstractArray
        acc = sum(Ŷ_bool .== labels)/(s[end-1]*s[end])
        println("Accuracy = $acc")
    end
    return Ŷ_bool
end #predictpredict(probs::Array{T, 2},


### relu

abstract type relu <: actFun end


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



### softmax


abstract type softmaxFamily <: actFun end

abstract type softmax <: softmaxFamily end

"""
    compute the softmax function

"""
function softmax(Ŷ::AbstractArray{T, N}) where {T,N}
    Ŷ_exp = exp.(Ŷ)
    sumofexp = sum(Ŷ_exp, dims=N-1)
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

Base.eachslice(A::AbstractArray, B::AbstractArray; dims::Integer) = (eachslice(A, dims=dims), eachslice(B,dims=dims))

function predict(
    actFun::Type{S},
    probs::Array{T, N};
    labels=nothing,
    ) where {T, N, S <: softmaxFamily}

    maximums = maximum(probs, dims=N-1)
    Ŷ_bool = probs .== maximums
    if labels isa AbstractArray
        acc = 0
        bool_labels = Bool.(labels)
        for (lab, pred) in eachslice(labels, Ŷ_bool; dims=N)
            acc += (lab == pred) ? 1 : 0
        end
        acc /= size(labels)[end]
        println("Accuracy = $acc")
    end


    return Ŷ_bool
end #predictpredict(probs::Array{T, 2}, :softmax)

export predict

### tanh

abstract type tanh <: actFun end


Base.tanh(Z::Array{T,N}) where {T,N} = tanh.(Z)

dtanh(Z::Array{T,N}) where {T,N} = 1 .- (tanh.(Z)).^2

export dtanh, tanh


### noAct


abstract type noAct <: actFun end

function noAct(Z)
    return Z
end

function dnoAct(Z)
    return ones(eltype(Z), size(Z)...)
end

export noAct, dnoAct
