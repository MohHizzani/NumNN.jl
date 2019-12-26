
using LinearAlgebra
using ProgressMeter


struct Layer
    numNodes::Integer
    actFun::Symbol

    """
        drop-out keeping node probability
    """
    keepProb::AbstractFloat
    # W::AbstractArray{<:AbstractFloat,2}
    # B::AbstractArray{<:AbstractFloat,2}
    function Layer(numNodes,actFun;keepProb=1.0)
        # W, B
        new(numNodes, actFun, keepProb)
    end #Layer
end #struct Layer

mutable struct Model
    layers::AbstractArray{Layer,1}
    lossFun::Symbol

    """
        regulization type
            0 : mean no regulization
            1 : L1 regulization
            2 : L2 regulization
    """
    regulization::Integer

    """
        regulization constant
    """
    λ::AbstractFloat

    """
        learning rate
    """
    α::AbstractFloat
    W::AbstractArray{AbstractArray{AbstractFloat,2},1}
    B::AbstractArray{AbstractArray{AbstractFloat,2},1}
    function Model(X, Y, layers, α; regulization = 0, λ = 1.0, lossFun = :crossentropy)
        W, B = deepInitWB(X, Y, layers)
        @assert regulization in [0, 1, 2]
        return new(layers, lossFun, regulization, λ, α, W, B)
    end #inner-constructor
end #Model


"""
    return the Sigmoid output
"""
σ(x,w,b) = 1/(1+exp(-(w*x+b)))
σ(z)  = 1/(1+exp(-z))

dσ(z) = σ(z) * (1-σ(z))


"""
    return the ReLU output
"""
relu(z::T) where {T} = max(zero(T), z)

drelu(z::T) where {T} = z > zero(T) ? one(T) : zero(T)

"""
    initialize W and B for layer with inputs of size of (nl_1) and layer size
        of (nl)

    returns:
        W: of size of (nl, nl_1)
"""
function initWB(nl, nl_1,
                p::Type{T}=Float64::Type{Float64};
                He=true,
                coef=0.01,
                zro=false) where {T}
    if He
        coef = sqrt(2/nl_1)
    end
    if zro
        W = zeros(T, (nl,nl_1))
    else
        W = randn(T, nl, nl_1) .* coef
    end
    B = zeros(T, (nl,1))
    return W, B
end #initWB



"""
    initialize W's and B's using
        X := is the input of the neural Network
        Y := is the labels
        layers := is 1st rank array contains elements of Layer(s) (hidden and output)

    returns:
        W := 1st rank array contains all the W's for each layer
            #Vector{Matrix{T}}
        B := 1st rank array contains all the B's for each layer
            #Vector{Matrix{T}}
"""
function deepInitWB(X, Y,
                    layers,
                    p::Type{T}=Float64::Type{Float64};
                    He=true,
                    coef=0.01,
                    zro=false) where {T}

    W = Array{Matrix{T},1}()
    B = Array{Matrix{T},1}()
    _w, _b = initWB(layers[1],size(X)[1],T; He=true, coef=0.01, zro=false)
    push!(W, _w)
    push!(B, _b)
    for i=2:length(layers)
        _w, _b = initWB(layers[i].numNodes,
                        layers[i-1].numNodes,
                        T;
                        He=true,
                        coef=0.01,
                        zro=false)
        push!(W, _w)
        push!(B, _b)
    end
    return W,B
end #deepInitWB





"""
    predict Y using the model
"""
function predict(model::Model, X)
    W, B, layers = model.W, model.B, model.layers
    Ŷ = forwardProp(X, W, B, layers)["Yhat"]
    return Ŷ
end #predict

"""
    perform the forward propagation using

    input:
        x :=
"""
function forwardProp(X::Matrix{T},
                     model::Model) where {T}

    W::AbstractArray{Matrix{T},1},
    B::AbstractArray{Matrix{T},1},
    layers::AbstractArray{Layer,1};
    costFun = model.W, model.B, model.layers, model.lossFun
    regulization = model.regulization
    λ = model.λ
    m = size(X)[2]
    L = length(layers)
    A = Vector{Matrix{eltype(X)}}()
    Z = Vector{Matrix{eltype(X)}}()
    push!(Z, W[1]*X .+ B[1])
    push!(A, eval(:($layers[1].actFun.(Z[1]))))
    for l=2:L
        push!(Z, W[l]*A[l-1] .+ B[l])
        actFun = layers[l].actFun
        push!(A, eval(:($(actFun).(Z[l]))))
    end

    cost = eval(:($costFun(A[L], Y)))

    #add the cost of regulization
    if regulization > 0
        cost += (λ/2m) * sum([norm(w, regulization) for w in W])
    end

    return Dict("A"=>A,
                "Z"=>Z,
                "Yhat"=>A[L],
                "Cost"=>cost)
end #forwardProp

"""
    input:
        Ŷ := (1,m) matrix of predicted labels
        Y := (1,m) matrix of true      labels

    output:
        the average of the cross entropy loss function
"""
function crossentropy(Ŷ, Y)
    m = length(Y)
    newŶ = copy(Ŷ)

    """
        return previous float if x == 1 and nextfloat if x == 0
    """
    prevnextfloat(x) = x==0 ? nextfloat(x) : x==1 ? prevfloat(x) : x

    newŶ = prevnextfloat.(newŶ)

    J = -sum(Y .* log.(newŶ) .+ (1 .- Y) .* log.(1 .- newŶ))/m
    return J
end #crossentropy



"""
    compute the drivative of cross-entropy loss function
"""
function dcrossentropy(a, y)

    """
        return previous float if x == 1 and nextfloat if x == 0
    """
    prevnextfloat(x) = x==0 ? nextfloat(x) : x==1 ? prevfloat(x) : x

    aNew = prevnextfloat(a)
    dJ = -(y/a + (1-y)/(1-a))
    return dJ
end #dcrossentropy


"""
    compute the softmax function

"""
function softmax(Ŷ)
    Ŷ_exp = exp.(Ŷ)
    sumofexp = sum(Ŷ)
    return Ŷ./sumofexp
end #softmax



"""

    inputs:
    X := is a (nx, m) matrix
    model contains these parameters
        W := is a Vector of Matrices of (n[l], n[l-1])
        B := is a Vector of Matrices of (n[l], 1)

"""
function backprop(X,Y,
                  model::Model,
                  cache::Dict{})

    layers::AbstractArray{Layer, 1} = model.layers
    lossFun = model.lossFun
    m = size(X)[2]
    L = length(layers)
    A, Z = cache["A"], cache["Z"]
    W, B, regulization, λ = model.W, model.B, model.regulization, model.λ

    D = Vector{BitArray}([rand(size(X[i])...) .< layers[i].keepProb for i=1:L])

    A = [A[i] .* D[i] for i=1:L]
    A = [A[i] ./ layers[i].keepProb for i=1:L]
    # init all Arrays
    dA = Vector{Matrix{eltype(A)}}([similar(mat) for mat in A])
    dZ = Vector{Matrix{eltype(Z)}}([similar(mat) for mat in Z])
    dW = Vector{Matrix{eltype(W)}}([similar(mat) for mat in W])
    dB = Vector{Matrix{eltype(B)}}([similar(mat) for mat in B])

    dlossFun = Symbol("d",lossFun)
    dA[L] = eval(:($dlossFun.(A[L], Y))) #.* eval(:($dActFun.(Z[L])))

    dA[L] = dA[L] .* D[L]
    dA[L] = dA[L] ./ layers[L].keepProb

    for l=L:-1:1
        actFun = layers[l].actFun
        dActFun = Symbol("d",actFun)
        dZ[l] = dA[l] .* eval(:($dActFun.(Z[l])))

        dW[l] = 1/m .* dZ[l]*A[l-1]'

        if regulization > 0
            if regulization==1
                dW[l] .+= (λ/m)
            else
                dW[l] .+= (λ/m) .* W[l]
            end
        end
        dB[l] = 1/m .* sum(dZ[l], dims=2)
        dA[l-1] = W[l]'dZ[l]
        A[l-1] = dA[l-1] .* D[l-1]
        dA[l-1] = dA[l-1] ./ layers[l-1].keepProb
    end
    grads = Dict("dW"=>dW,
                 "dB"=>dB)
    return grads
end #backprop


function updateParams(W, B, grads::Dict, α)
    dW, dB = grads["dW"], grads["dB"]
    W .-= α .* dW
    B .-= α .* dB
    return W, B
end


"""
    Repeat the trainging for a single preceptron


    returns:
        W := Array of matrices of size (n[l], n[l-1])
        B := Array of matrices of size (n[l], 1)
"""
function train(X,Y,model::Model, epochs; ϵ=10^-6)
    layers, lossFun, α, W, B = model.layers, model.lossFun, model.α, model.W, model.B
    Costs = []
#     p = Progress(epochs, 1)
    for i=1:epochs
        cache = forwardProp(X,
                            model)
        push!(Costs, cache["Cost"])
        grads = backprop(X,Y,
                         model,
                         cache)

        W, B = updateParams(W, B, grads, α)
        println("N = $i, Cost = $cost\r")
#         next!(p)
    end

    model.W, model.B = W, B
    return Dict("model"=>model,
                "Costs"=>Costs)
end



function pic_model(X_train,
                   Y_train,
                   X_test,
                   Y_test,
                   ActFun=σ,
                   num_interations = 2000,
                   α = 0.5,
                   print_cost = false)

    W, B = initWB(X_train)

    W, B, train_costs, Ŷs, Ws, Bs = multiBackprop(X_train,
                                                   Y_train,
                                                   W,
                                                   B,
                                                   α,
                                                   ActFun,
                                                   num_interations)

    Ŷ_train = broadcast((x)-> x>0.5 ? 1 : 0, predict(X_train, W, B, ActFun))
    Ŷ_test = broadcast((x)-> x>0.5 ? 1 : 0, predict(X_test, W, B, ActFun))

    training_acc = sum(abs.(Ŷ_train .- Y_train))/length(Ŷ_train)
    testing_acc = sum(abs.(Ŷ_test .- Y_test))/length(Y_test)

    println("Accuracy of traingin data is: $training_acc")
    println("Accuracy of testing data is : $testing_acc")

    d = Dict("W"=>W,
             "B"=>B,
             "train_costs"=>train_costs,
             "yhat_train"=>Ŷ_train,
             "yhat_test"=>Ŷ_test,
             "yhats"=>Ŷs,
             "Ws"=>Ws,
             "Bs"=>Bs)

    return d



end
