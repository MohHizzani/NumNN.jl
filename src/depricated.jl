
"""
    flatten 2D matrix into (m*n, 1) matrix

        mainly used for images to flatten images

    inputs:
        x := 3D (rgp, m, n) matrix

    outputs:
        y := 2D (rgp*m*n, 1) matrix
"""
function flatten(x)
    rgp, n, m = size(x)
    return reshape(x, (rgp*m*n, 1))
end



"""
    perform the forward propagation using

    input:
        x := (n0, m) matrix
        y := (c,  m) matrix where c is the number of classes

    return cache of A, Z, Yhat, Cost
"""
function forwardProp(X::Matrix{T},
                     Y::Matrix{T},
                     model::Model) where {T}

    W::AbstractArray{Matrix{T},1},
    B::AbstractArray{Matrix{T},1},
    layers::AbstractArray{Layer,1},
    lossFun = model.W, model.B, model.layers, model.lossFun
    regulization = model.regulization
    λ = model.λ
    m = size(X)[2]
    c = size(Y)[1]
    L = length(layers)
    A = Vector{Matrix{eltype(X)}}()
    Z = Vector{Matrix{eltype(X)}}()
    push!(Z, W[1]*X .+ B[1])
    actFun = layers[1].actFun
    push!(A, eval(:($actFun.($Z[1]))))
    for l=2:L
        push!(Z, W[l]*A[l-1] .+ B[l])
        actFun = layers[l].actFun

        if isequal(actFun, :softmax)
            a = Matrix{eltype(X)}(undef, c, 0)
            for i=1:m
                zCol = Z[l][:,i]
                a = hcat(a, eval(:($(actFun)($zCol))))
            end
            push!(A, a)
        else
            push!(A, eval(:($(actFun).($Z[$l]))))
        end
    end

    if isequal(lossFun, :categoricalCrossentropy)
        cost = sum(eval(:($lossFun.($A[$L], $Y))))/ m
    else
        cost = sum(eval(:($lossFun.($A[$L], $Y)))) / (m*c)
    end

    if regulization > 0
        cost += (λ/2m) * sum([norm(w, regulization) for w in W])
    end

    return Dict("A"=>A,
                "Z"=>Z,
                "Yhat"=>A[L],
                "Cost"=>cost)
end #forwardProp

# export forwardProp


"""
    predict Y using the model and the input X and the labels Y
    inputs:
        model := the trained model
        X := the input matrix
        Y := the input labels to compare with

    output:
        a Dict of
        "Yhat" := the predicted values
        "Yhat_bools" := the predicted labels
        "accuracy" := the accuracy of the predicted labels
"""
function predict(model::Model, X, Y)
    Ŷ = forwardProp(X, Y, model)["Yhat"]
    T = eltype(Ŷ)
    layers = model.layers
    c, m = size(Y)
    # if isbool(Y)
    acc = 0
    if isequal(layers[end].actFun, :softmax)
        Ŷ_bool = BitArray(undef, c, 0)
        p = Progress(size(Ŷ)[2], 0.01)
        for v in eachcol(Ŷ)
            Ŷ_bool = hcat(Ŷ_bool, v .== maximum(v))
            next!(p)
        end

        acc = sum([Ŷ_bool[:,i] == Y[:,i] for i=1:size(Y)[2]])/m
        println("Accuracy is = $acc")
    end

    if isequal(layers[end].actFun, :σ)
        Ŷ_bool = BitArray(undef, c, 0)
        p = Progress(size(Ŷ)[2], 0.01)
        for v in eachcol(Ŷ)
            Ŷ_bool = hcat(Ŷ_bool, v .> T(0.5))
            next!(p)
        end

        acc = sum(Ŷ_bool .== Y)/(c*m)
        println("Accuracy is = $acc")
    end
    return Dict("Yhat"=>Ŷ,
                "Yhat_bool"=>Ŷ_bool,
                "accuracy"=>acc)
end #predict



function backProp(X,Y,
                  model::Model,
                  cache::Dict{})

    layers::AbstractArray{Layer, 1} = model.layers
    lossFun = model.lossFun
    m = size(X)[2]
    L = length(layers)
    A, Z = cache["A"], cache["Z"]
    W, B, regulization, λ = model.W, model.B, model.regulization, model.λ

    D = [rand(size(A[i])...) .< layers[i].keepProb for i=1:L]

    if layers[L].keepProb < 1.0 #to save time of multiplication in case keepProb was one
        A = [A[i] .* D[i] for i=1:L]
        A = [A[i] ./ layers[i].keepProb for i=1:L]
    end
    # init all Arrays
    dA = Vector{Matrix{eltype(A[1])}}([similar(mat) for mat in A])
    dZ = Vector{Matrix{eltype(Z[1])}}([similar(mat) for mat in Z])
    dW = Vector{Matrix{eltype(W[1])}}([similar(mat) for mat in W])
    dB = Vector{Matrix{eltype(B[1])}}([similar(mat) for mat in B])

    dlossFun = Symbol("d",lossFun)
    actFun = layers[L].actFun
    if ! isequal(actFun, :softmax) && !(actFun==:σ &&
                                        lossFun==:binaryCrossentropy)
        dA[L] = eval(:($dlossFun.($A[$L], $Y))) #.* eval(:($dActFun.(Z[L])))
    end
    if layers[L].keepProb < 1.0 #to save time of multiplication in case keepProb was one
        dA[L] = dA[L] .* D[L]
        dA[L] = dA[L] ./ layers[L].keepProb
    end
    for l=L:-1:2
        actFun = layers[l].actFun
        dActFun = Symbol("d",actFun)
        if l==L && (isequal(actFun, :softmax) ||
                    (actFun==:σ && lossFun==:binaryCrossentropy))
            dZ[l] = A[l] .- Y
        else
            dZ[l] = dA[l] .* eval(:($dActFun.($Z[$l])))
        end

        dW[l] = dZ[l]*A[l-1]' ./m

        if regulization > 0
            if regulization==1
                dW[l] .+= (λ/2m)
            else
                dW[l] .+= (λ/m) .* W[l]
            end
        end
        dB[l] = 1/m .* sum(dZ[l], dims=2)
        dA[l-1] = W[l]'dZ[l]
        if layers[l-1].keepProb < 1.0 #to save time of multiplication in case keepProb was one
            dA[l-1] = dA[l-1] .* D[l-1]
            dA[l-1] = dA[l-1] ./ layers[l-1].keepProb
        end
    end

    l=1 #shortcut cause I just copied from the for loop
    actFun = layers[l].actFun
    dActFun = Symbol("d",actFun)
    dZ[l] = dA[l] .* eval(:($dActFun.($Z[$l])))

    dW[l] = 1/m .* dZ[l]*X' #where X = A[0]

    if regulization > 0
        if regulization==1
            dW[l] .+= (λ/2m)
        else
            dW[l] .+= (λ/m) .* W[l]
        end
    end
    dB[l] = 1/m .* sum(dZ[l], dims=2)

    grads = Dict("dW"=>dW,
                 "dB"=>dB)
    return grads
end #backProp


function updateParams(model::Model, grads::Dict, tMiniBatch::Integer)
    W, B, V, S, layers = model.W, model.B, model.V, model.S, model.layers
    optimizer = model.optimizer
    dW, dB = grads["dW"], grads["dB"]
    L = length(layers)
    α = model.α
    β1, β2, ϵAdam = model.β1, model.β2, model.ϵAdam

    #initialize the needed variables to hold the corrected values
    #it is being init here cause these are not needed elsewhere
    VCorrected, SCorrected = deepInitVS(W,B,optimizer)
    if optimizer==:adam || optimizer==:momentum
        for i=1:L
            V[:dw][i] .= β1 .* V[:dw][i] .+ (1-β1) .* dW[i]
            V[:db][i] .= β1 .* V[:db][i] .+ (1-β1) .* dB[i]

            ##correcting
            VCorrected[:dw][i] .= V[:dw][i] ./ (1-β1^tMiniBatch)
            VCorrected[:db][i] .= V[:db][i] ./ (1-β1^tMiniBatch)

            if optimizer==:adam
                S[:dw][i] .= β2 .* S[:dw][i] .+ (1-β2) .* (dW[i].^2)
                S[:db][i] .= β2 .* S[:db][i] .+ (1-β2) .* (dB[i].^2)

                ##correcting
                SCorrected[:dw][i] .= S[:dw][i] ./ (1-β2^tMiniBatch)
                SCorrected[:db][i] .= S[:db][i] ./ (1-β2^tMiniBatch)

                ##update parameters with adam
                W[i] .-= (α .* (VCorrected[:dw][i] ./ (sqrt.(SCorrected[:dw][i]) .+ ϵAdam)))
                B[i] .-= (α .* (VCorrected[:db][i] ./ (sqrt.(SCorrected[:db][i]) .+ ϵAdam)))
            else#if optimizer==:momentum
                W[i] .-= (α .* VCorrected[:dw][i])
                B[i] .-= (α .* VCorrected[:db][i])
            end #if optimizer==:adam
        end #for i=1:L
    else
        W .-= (α .* dW)
        B .-= (α .* dB)
    end #if optimizer==:adam || optimizer==:momentum
    return W, B
end #updateParams



function updateParams!(model::Model, grads::Dict, tMiniBatch::Integer)
    W, B, V, S, layers = model.W, model.B, model.V, model.S, model.layers
    optimizer = model.optimizer
    dW, dB = grads["dW"], grads["dB"]
    L = length(layers)
    α = model.α
    β1, β2, ϵAdam = model.β1, model.β2, model.ϵAdam

    #initialize the needed variables to hold the corrected values
    #it is being init here cause these are not needed elsewhere
    VCorrected, SCorrected = deepInitVS(W,B,optimizer)
    if optimizer==:adam || optimizer==:momentum
        for i=1:L
            V[:dw][i] .= β1 .* V[:dw][i] .+ (1-β1) .* dW[i]
            V[:db][i] .= β1 .* V[:db][i] .+ (1-β1) .* dB[i]

            ##correcting
            VCorrected[:dw][i] .= V[:dw][i] ./ (1-β1^tMiniBatch)
            VCorrected[:db][i] .= V[:db][i] ./ (1-β1^tMiniBatch)

            if optimizer==:adam
                S[:dw][i] .= β2 .* S[:dw][i] .+ (1-β2) .* (dW[i].^2)
                S[:db][i] .= β2 .* S[:db][i] .+ (1-β2) .* (dB[i].^2)

                ##correcting
                SCorrected[:dw][i] .= S[:dw][i] ./ (1-β2^tMiniBatch)
                SCorrected[:db][i] .= S[:db][i] ./ (1-β2^tMiniBatch)

                ##update parameters with adam
                W[i] .-= (α .* (VCorrected[:dw][i] ./ (sqrt.(SCorrected[:dw][i]) .+ ϵAdam)))
                B[i] .-= (α .* (VCorrected[:db][i] ./ (sqrt.(SCorrected[:db][i]) .+ ϵAdam)))
            else#if optimizer==:momentum
                W[i] .-= (α .* VCorrected[:dw][i])
                B[i] .-= (α .* VCorrected[:db][i])
            end #if optimizer==:adam
        end #for i=1:L
    else
        W .-= (α .* dW)
        B .-= (α .* dB)
    end #if optimizer==:adam || optimizer==:momentum
    model.W, model.B = W, B
    return
end #updateParams!


function updateParams!(model::Model,
                       cLayer::Layer,
                       cnt::Integer = -1;
                       tMiniBatch::Integer = 1)

    optimizer = model.optimizer
    α = model.α
    β1, β2, ϵAdam = model.β1, model.β2, model.ϵAdam

    if cLayer.updateCount >= cnt
        return nothing
    end #if cLayer.updateCount >= cnt

    cLayer.updateCount += 1
    #initialize the needed variables to hold the corrected values
    #it is being init here cause these are not needed elsewhere
    VCorrected = Dict(:dw=>similar(cLayer.dW), :db=>similar(cLayer.dB))
    SCorrected = Dict(:dw=>similar(cLayer.dW), :db=>similar(cLayer.dB))
    if optimizer==:adam || optimizer==:momentum

        cLayer.V[:dw] .= β1 .* cLayer.V[:dw] .+ (1-β1) .* cLayer.dW
        cLayer.V[:db] .= β1 .* cLayer.V[:db] .+ (1-β1) .* cLayer.dB

        ##correcting
        VCorrected[:dw] .= cLayer.V[:dw] ./ (1-β1^tMiniBatch)
        VCorrected[:db] .= cLayer.V[:db] ./ (1-β1^tMiniBatch)

        if optimizer==:adam
            cLayer.S[:dw] .= β2 .* cLayer.S[:dw] .+ (1-β2) .* (cLayer.dW.^2)
            cLayer.S[:db] .= β2 .* cLayer.S[:db] .+ (1-β2) .* (cLayer.dB.^2)

            ##correcting
            SCorrected[:dw] .= cLayer.S[:dw] ./ (1-β2^tMiniBatch)
            SCorrected[:db] .= cLayer.S[:db] ./ (1-β2^tMiniBatch)

            ##update parameters with adam
            cLayer.W .-= (α .* (VCorrected[:dw] ./ (sqrt.(SCorrected[:dw]) .+ ϵAdam)))
            cLayer.B .-= (α .* (VCorrected[:db] ./ (sqrt.(SCorrected[:db]) .+ ϵAdam)))

        else#if optimizer==:momentum

            cLayer.W .-= (α .* VCorrected[:dw])
            cLayer.B .-= (α .* VCorrected[:db])

        end #if optimizer==:adam
    else
        cLayer.W .-= (α .* cLayer.dW)
        cLayer.B .-= (α .* cLayer.dB)
    end #if optimizer==:adam || optimizer==:momentum

    return nothing
end #updateParams!
