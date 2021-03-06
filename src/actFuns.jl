
abstract type actFun end

export actFun



export probToValue


### sigmoid

abstract type σ <: actFun end

"""
    return the Sigmoid output
    inputs must be matices
"""
σ(X, W, B) = 1 ./ (1 .+ exp.(.-(W * X .+ B)))
σ(Z) = 1 ./ (1 .+ exp.(.-Z))

export σ

"""
    return the derivative of Sigmoid function
"""
dσ(Z) = σ(Z) .* (1 .- σ(Z))

export dσ

@doc raw"""
    function probToValue(
        actFun::Type{σ},
        probs::AbstractArray{T,N},
        labels::Aa = nothing;
        evalConst = 0.5,
    ) where {Aa<:Union{<:AbstractArray,Nothing},T,N}

Convert the probabilities return out of sigmoid function to Bool value (i.e. 0,1) values based on comparing on a threshold value `evalConst`

# Return

- `Ŷ_bool` := Boolean valuse of the probabilites

- `acc` := Accuracy when `labels` provided
"""
function probToValue(
    actFun::Type{σ},
    probs::AbstractArray{T,N},
    labels::Aa = nothing;
    evalConst = 0.5,
) where {Aa<:Union{<:AbstractArray,Nothing},T,N}

    s = size{probs}
    Ŷ_bool = probs .> T(evalConst)
    acc = nothing
    if labels isa AbstractArray
        acc = sum(Ŷ_bool .== labels) / (s[end-1] * s[end])
        # println("Accuracy = $acc")
    end
    return Ŷ_bool, acc
end #predictpredict(probs::AbstractArray{T, 2},


### relu

abstract type relu <: actFun end


"""
    return the ReLU output
"""
function relu(Z::AbstractArray{T,N}) where {T,N}
    max.(zero(T), Z)
end #function relu(Z::AbstractArray{T,N}) where {T,N}

export relu

"""
    return the derivative of ReLU function
"""
function drelu(Z::AbstractArray{T,N}) where {T,N}
    return T.(Z .> zero(T))
end #function drelu(z::AbstractArray{T,N}) where {T,N}

export drelu



### softmax


abstract type softmaxFamily <: actFun end

abstract type softmax <: softmaxFamily end

"""
    compute the softmax function

"""
function softmax(Ŷ::AbstractArray{T,N}) where {T,N}
    Ŷ_exp = exp.(Ŷ)
    sumofexp = sum(Ŷ_exp, dims = N - 1)
    return Ŷ_exp ./ sumofexp
end #softmax

function dsoftmax(Ŷ, dim = 1)
    sŶ = softmax(Ŷ)
    T = eltype(Ŷ)
    softMat = AbstractArray{T,3}(undef, 0, 0, 0)
    sSize = size(Ŷ)[dim]
    for c in eachcol(sŶ)
        tmpMat = zeros(T, sSize, sSize)
        for i = 1:length(c)
            for j = 1:length(c)
                if i == j
                    tmpMat[i, j] = c[i] * (1 - c[j])
                else
                    tmpMat[i, j] = -c[i] * c[j]
                end
            end
        end
        softMat = cat(softMat, tmpMat, dims = 3)
    end

    return softMat
end #dsoftmax

export softmax, dsoftmax

@doc raw"""
    function probToValue(
        actFun::Type{S},
        probs::AbstractArray{T,N};
        labels = nothing,
    ) where {T,N,S<:softmaxFamily}

convert the probabilites out of `softmax` or softmax-like functions into `Bool` values, where the max value gets 1 and the other get zeros

# Return

- `Ŷ_bool` := Boolean valuse of the probabilites

- `acc` := Accuracy when `labels` provided
"""
function probToValue(
    actFun::Type{S},
    probs::AbstractArray{T,N};
    labels = nothing,
) where {T,N,S<:softmaxFamily}

    maximums = maximum(probs, dims = N - 1)
    Ŷ_bool = probs .== maximums
    acc = nothing
    if labels isa AbstractArray
        acc = 0
        bool_labels = Bool.(labels)
        ax = axes(bool_labels)[1:end-1]
        endax = axes(bool_labels)[end]
        trueFalse = Array{Bool,1}(undef, length(endax))
        @simd for i in endax
            lab = view(bool_labels, ax..., i)
            pred = view(Ŷ_bool, ax..., i)
            trueFalse[i] = (lab == pred)
        end
        acc = mean(trueFalse)
        # println("Accuracy = $acc")
    end


    return Ŷ_bool, acc
end #predictpredict(probs::AbstractArray{T, 2}, :softmax)


### tanh

abstract type tanh <: actFun end


Base.tanh(Z::AbstractArray{T,N}) where {T,N} = Base.tanh.(Z)

tanh(Z::AbstractArray{T,N}) where {T,N} = Base.tanh.(Z)

dtanh(Z::AbstractArray{T,N}) where {T,N} = 1 .- (Base.tanh.(Z)) .^ 2

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
