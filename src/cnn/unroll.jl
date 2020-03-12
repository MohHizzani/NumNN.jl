using ToeplitzMatrices
using PaddedViews

function unroll(cLayer::Conv2D, Ai)
    prevLayer = cLayer.prevLayer
    n_Hi, n_Wi, ci, m = size(Ai)
    HWi = n_Hi*n_Wi
    f_H, f_W = cLayer.f
    fHW = f_H * f_W
    s_W, s_H = cLayer.s
    c = cLayer.channels
    n_H = (n_Hi - f_H) ÷ s_H + 1
    n_W = (n_Wi - f_W) ÷ s_W + 1
    W = cLayer.W
    T = eltype(W)
    B = cLayer.B
    K = nothing
    for ch=1:size(W)[end]
        w3d = W[:,:,:,ch]
        k2d = nothing
        k3d = nothing
        k = nothing
        for i=1:n_W
            w3d = PaddedView(0, w3d, (n_Wi, n_Wi,ci), (i,1,1))
            for w=1:size(w3d)[3]
                for h=1:size(w3d)[1]
                    if h==1
                        k2d = Toeplitz([Float64.(w3d[h,1,w]), repeat([0],n_H-1)...], w3d[h,:,w])
                    else
                        k2d = hcat(k2d, Toeplitz([w3d[h,1,w], repeat([0], n_H-1)...], w3d[h,:,w]))
                    end #if h==1
                end #for h=1:size(w)[1]
                if w==1
                    k3d = k2d
                else
                    k3d = hcat(k3d, k2d)
                end #if w==1
            end #for w=1:size(w3d)[3]

            if i==1
                k = k3d
            else
                k = vcat(k, k3d)
            end #if i==1
        end #for i=1:n_W

        if ch==1
            K = k
        else
            K = vcat(K, k)
        end #if ch==1
    end #for ch=1:size(W)[end]

    return T.(K)

end #function unroll(cLayer::Conv2D)
