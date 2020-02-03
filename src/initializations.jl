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
                    outLayer::Layer;
                    He=true,
                    coef=0.01,
                    zro=false)

    T = eltype(X)
    W = Array{Matrix{T},1}()
    B = Array{Matrix{T},1}()
    prevLayer = outLayer.prevLayer
    _w, _b = initWB(outLayer.numNodes, prevLayer.numNodes,T; He=true, coef=0.01, zro=false)
    outLayer.W, outLayer.B = _w, _b
    push!(W, _w)
    push!(B, _b)

    # for i=2:length(layers)
    while (! isequal(prevLayer, nothing))
        if (! isa(prevLayer, AddLayer))
            if (! isequal(prevLayer.prevLayer, nothing))
                _w, _b = initWB(prevLayer.numNodes,
                                prevLayer.prevLayer.numNodes,
                                T;
                                He=true,
                                coef=0.01,
                                zro=false)
            else
                _w, _b = initWB(prevLayer.numNodes,
                                size(X)[1],
                                T;
                                He=true,
                                coef=0.01,
                                zro=false)
            end #if (! isequal(prevLayer.prevLayer, nothing))
            prevLayer.W, prevLayer.B = _w, _b
            push!(W, _w)
            push!(B, _b)
            prevLayer = prevLayer.prevLayer

        else #if it's an AddLayer
            _w = Matrix{T}(undef, 0,0)
            _b = Matrix{T}(undef, 0,0)
            push!(W, _w)
            push!(B, _b)
            prevLayer = prevLayer.prevLayer
        end #if (! isequal(prevLayer, AddLayer))


    end #while (! isequal(prevLayer, nothing))

    return W[end:-1:1],B[end:-1:1]
end #deepInitWB

export deepInitWB



function deepInitVS(W::Array{Array{T,2},1},
                    B::Array{Array{T,2},1},
                    optimizer::Symbol) where {T<:Number}

    if optimizer==:adam || optimizer==:momentum
        VdW = Array{Array{eltype(W[1]),2},1}([zeros(size(w)) for w in W])
        VdB = Array{Array{eltype(B[1]),2},1}([zeros(size(b)) for b in B])
        if optimizer==:adam
            SdW = Array{Array{eltype(W[1]),2},1}([zeros(size(w)) for w in W])
            SdB = Array{Array{eltype(B[1]),2},1}([zeros(size(b)) for b in B])
        else
            SdW = Array{Array{eltype(W[1]),2},1}(undef,0)
            SdB = Array{Array{eltype(W[1]),2},1}(undef,0)
        end

    else
        SdW = Array{Array{eltype(W[1]),2},1}(undef,0)
        SdB = Array{Array{eltype(W[1]),2},1}(undef,0)
        VdW = Array{Array{eltype(W[1]),2},1}(undef,0)
        VdB = Array{Array{eltype(W[1]),2},1}(undef,0)
    end

    return Dict(:dw=>VdW, :db=>VdB), Dict(:dw=>SdW, :db=>SdB)
end

export deepInitVS
