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

export initWB

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
    _w, _b = initWB(layers[1].numNodes,size(X)[1],T; He=true, coef=0.01, zro=false)
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

export deepInitWB
