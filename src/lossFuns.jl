
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
function crossentropy(a, y)

    aNew = prevnextfloat.(a)
    J = sum(.-(y .* log.(aNew) .+ (1 .- y) .* log.(1 .- aNew)))
    return J
end #crossentropy

export crossentropy

"""
    compute the drivative of cross-entropy loss function
"""
function dcrossentropy(a, y)
    aNew = prevnextfloat.(a)
    dJ = .-(y ./ aNew .- (1 .- y) ./ (1 .- aNew))
    return dJ
end #dcrossentropy

export dcrossentropy

"""
    return previous float if x == 1 and nextfloat if x == 0
"""
prevnextfloat(x) = x==0 ? nextfloat(x) : x==1 ? prevfloat(x) : x

export prevnextfloat
