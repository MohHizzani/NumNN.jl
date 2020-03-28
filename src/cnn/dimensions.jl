
export outDims

function outDims(cLayer::PL, Ai::AbstractArray{T,N}) where {PL <: PaddableLayer, N, T}
    return outDims(cLayer, size(Ai))
end

function outDims(cLayer::PL, AiS::Tuple) where {PL <: PaddableLayer}

    f = cLayer.f
    s = cLayer.s
    inputS = cLayer.inputS[1:end-2]
    paddedS = paddedSize(cLayer, AiS)[1:end-2]
    ci, m = cLayer.inputS[end-1:end]
    co = cLayer.channels
    outputS = []
    for i=1:length(f)
        n = (paddedS[i] - f[i]) ÷ s[i] + 1
        push!(outputS, n)
    end

    return (outputS..., co, m)

end
