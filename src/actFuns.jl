"""
    return the Sigmoid output
"""
σ(x,w,b) = 1/(1+exp(-(w*x+b)))
σ(z)  = 1/(1+exp(-z))

"""
    return the derivative of Sigmoid function
"""
dσ(z) = σ(z) * (1-σ(z))


"""
    return the ReLU output
"""
relu(z::T) where {T} = max(zero(T), z)


"""
    return the derivative of ReLU function
"""
drelu(z::T) where {T} = z > zero(T) ? one(T) : zero(T)
