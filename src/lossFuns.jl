
abstract type lossFun end

export lossFun

### binaryCrossentropy

"""
    return the average cross entropy loss over vector of labels and predictions

    input:
        a := (c,1) matrix of predicted values, where c is the number of classes
        y := (c,1) matrix of predicted values, where c is the number of classes

        Note: in case the number of classes is one (1) it is okay to have
              a scaler values for a and y

    output:
        J := scaler value of the cross entropy loss
"""

abstract type binaryCrossentropy <: lossFun end

function binaryCrossentropy(a, y)

    aNew = prevnextfloat.(a)
    J = .-(y .* log.(aNew) .+ (1 .- y) .* log.(1 .- aNew))
    return J
end #binaryCrossentropy

export binaryCrossentropy

"""
    compute the drivative of cross-entropy loss function to the input of the
    layer dZ
"""
function dbinaryCrossentropy(a, y)
    dJ = a .- y
    return dJ
end #dbinaryCrossentropy

export dbinaryCrossentropy


### categoricalCrossentropy

abstract type categoricalCrossentropy <: lossFun end

function categoricalCrossentropy(a, y)
    aNew = prevnextfloat.(a)
    J = .-(y .* log.(aNew))
    return J
end


function dcategoricalCrossentropy(a, y)
    dJ = a .- y
    return dJ
end

export categoricalCrossentropy, dcategoricalCrossentropy


"""
    return previous float if x == 1 and nextfloat if x == 0
"""
prevnextfloat(x) = x==0 ? nextfloat(x) : x==1 ? prevfloat(x) : x

export prevnextfloat


### cost function

function cost(
    loss::Type{categoricalCrossentropy},
    A::AbstractArray{T1,N},
    Y::AbstractArray{T2,N},
) where {T1, T2, N}

    c, m = size(A)[N-1:N]
    costs = sum(loss(A,Y)) / m
    return costs

end #function cost

function cost(
    loss::Type{binaryCrossentropy},
    A::AbstractArray{T1,N},
    Y::AbstractArray{T2,N},
) where {T1, T2, N}

    c, m = size(A)[N-1:N]
    costs = sum(loss(A,Y)) / (c*m)
    return costs

end #function cost
