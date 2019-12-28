"""
    return the Sigmoid output
"""
σ(x,w,b) = 1/(1+exp(-(w*x+b)))
σ(z)  = 1/(1+exp(-z))

export σ

"""
    return the derivative of Sigmoid function
"""
dσ(z) = σ(z) * (1-σ(z))

export dσ

"""
    return the ReLU output
"""
relu(z::T) where {T} = max(zero(T), z)

export relu

"""
    return the derivative of ReLU function
"""
drelu(z::T) where {T} = z > zero(T) ? one(T) : zero(T)

export drelu


"""
    compute the softmax function

"""
function softmax(Ŷ)
    Ŷ_exp = exp.(Ŷ)
    sumofexp = sum(Ŷ)
    return Ŷ./sumofexp
end #softmax

function dsoftmax(Ŷ)
    return dσ.(Ŷ)
end

export softmax, dsoftmax
