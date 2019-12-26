"""
    input:
        Ŷ := (1,m) matrix of predicted labels
        Y := (1,m) matrix of true      labels

    output:
        the average of the cross entropy loss function
"""
# function crossentropy(Ŷ, Y)
#     m = length(Y)
#     newŶ = copy(Ŷ)
#
#     """
#         return previous float if x == 1 and nextfloat if x == 0
#     """
#     prevnextfloat(x) = x==0 ? nextfloat(x) : x==1 ? prevfloat(x) : x
#
#     newŶ = prevnextfloat.(newŶ)
#
#     J = -sum(Y .* log.(newŶ) .+ (1 .- Y) .* log.(1 .- newŶ))/m
#     return J
# end #crossentropy

function crossentropy(a, y)
    aNew = prevnextfloat(a)
    J = -(y * log(aNew) + (1-y) * log(1-aNew))
    return J
end #crossentropy

export crossentropy

"""
    compute the drivative of cross-entropy loss function
"""
function dcrossentropy(a, y)

    """
        return previous float if x == 1 and nextfloat if x == 0
    """
    prevnextfloat(x) = x==0 ? nextfloat(x) : x==1 ? prevfloat(x) : x

    aNew = prevnextfloat(a)
    dJ = -(y/aNew + (1-y)/(1-aNew))
    return dJ
end #dcrossentropy

export dcrossentropy

"""
    return previous float if x == 1 and nextfloat if x == 0
"""
prevnextfloat(x) = x==0 ? nextfloat(x) : x==1 ? prevfloat(x) : x
