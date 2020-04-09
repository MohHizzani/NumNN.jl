using PaddedViews

#TODO use parallel processing to speed up the unrolling process

#TODO build a reroll function to extract W from K

### unroll conv1d
@doc raw"""
    unroll(cLayer::Conv1D, AiS::Tuple, param::Symbol=:W)

unroll the `param` of `Conv1D` into 2D matrix

# Arguments

- `cLayer` := the layer of the paramters to unroll

- `AiS` := the `padded` input to determinde the size and shape of the output of `unroll`

- `param` := `Conv1D` parameter to be `unroll`ed

# Return

- `K` := 2D `Matrix` of the `param`
"""
function unroll(cLayer::Conv1D, AiS::Tuple, param::Symbol=:W)
    prevLayer = cLayer.prevLayer
    n_Hi, ci, m = AiS
    f_H = cLayer.f
    s_H = cLayer.s
    c = cLayer.channels
    n_H = (n_Hi - f_H) ÷ s_H + 1
    W = eval(:($cLayer.$param))
    T = eltype(W)
    B = cLayer.B
    K = nothing
    for ch=1:size(W)[end]
        k2d = nothing
        w3d = W[:,:,ch]
        k = nothing
        for i=range(1, step=s_H, length=n_H)
            w2d = PaddedView(0, w3d, (n_Hi,ci), (i,1))
            for w=1:size(w2d)[2]
                    if w==1
                        k2d = w2d[:,w]
                    else
                        k2d = vcat(k2d, w2d[:,w])
                    end #if h==1
            end #for w=1:size(w3d)[3]

            if i==1
                k = k2d
            else
                k = hcat(k, k2d)
            end #if i==1
        end #for i=1:n_W

        if ch == 1
            K = k
        else
            K = hcat(K, k)
        end #if ch==1
    end #for ch=1:size(W)[end]

    return transpose(K)

end #function unroll(cLayer::Conv2D)

###unroll conv2d
@doc raw"""
    unroll(cLayer::Conv2D, AiS::Tuple, param::Symbol=:W)

unroll the `param` of `Conv1D` into 2D matrix

# Arguments

- `cLayer` := the layer of the paramters to unroll

- `AiS` := the `padded` input to determinde the size and shape of the output of `unroll`

- `param` := `Conv1D` parameter to be `unroll`ed

# Return

- `K` := 2D `Matrix` of the `param`
"""
function unroll(cLayer::Conv2D, AiS::Tuple, param::Symbol=:W)
    prevLayer = cLayer.prevLayer
    n_Hi, n_Wi, ci, m = AiS
    HWi = n_Hi*n_Wi
    f_H, f_W = cLayer.f
    fHW = f_H * f_W
    s_H, s_W = cLayer.s
    c = cLayer.channels
    n_H = (n_Hi - f_H) ÷ s_H + 1
    n_W = (n_Wi - f_W) ÷ s_W + 1
    W = eval(:($cLayer.$param))
    T = eltype(W)
    B = cLayer.B
    K = nothing
    for ch=1:size(W)[end]
        w3da = W[:,:,:,ch]
        k2d = nothing
        k3d = nothing
        k4d = nothing
        k = nothing
        for i=range(1, step=s_H, length=n_H)
            w3db = PaddedView(0, w3da, (n_Hi, f_W,ci), (i,1,1))
            for j=range(1, step=s_W, length=n_W)
                w3d = PaddedView(0, w3db, (n_Hi, n_Wi,ci), (1,j,1))
                for w=1:size(w3d)[3]
                    for h=1:size(w3d)[1]
                        if h==1
                            k2d = w3d[h,:,w]
                        else
                            k2d = vcat(k2d, w3d[h,:,w])
                        end #if h==1
                    end #for h=1:size(w)[1]
                    if w==1
                        k3d = k2d
                    else
                        k3d = vcat(k3d, k2d)
                    end #if w==1
                end #for w=1:size(w3d)[3]

                if j==1
                    k4d = k3d
                else
                    k4d = hcat(k4d, k3d)
                end
            end #for j=1:n_H

            if i==1
                k = k4d
            else
                k = hcat(k, k4d)
            end #if i==1
        end #for i=1:n_W

        if ch==1
            K = k
        else
            K = hcat(K, k)
        end #if ch==1
    end #for ch=1:size(W)[end]

    return transpose(K)

end #function unroll(cLayer::Conv2D)



###unroll conv3d

@doc raw"""
    unroll(cLayer::Conv3D, AiS::Tuple, param::Symbol=:W)

unroll the `param` of `Conv3D` into 2D matrix

# Arguments

- `cLayer` := the layer of the paramters to unroll

- `AiS` := the `padded` input to determinde the size and shape of the output of `unroll`

- `param` := `Conv1D` parameter to be `unroll`ed

# Return

- `K` := 2D `Matrix` of the `param`
"""
function unroll(cLayer::Conv3D, AiS::Tuple, param::Symbol=:W)
    prevLayer = cLayer.prevLayer
    n_Hi, n_Wi, n_Di, ci, m = AiS
    HWDi = n_Hi*n_Wi*n_Di
    f_H, f_W, f_D = cLayer.f
    fHW = f_H * f_W * f_D
    s_H, s_W, s_D = cLayer.s
    c = cLayer.channels
    n_H = (n_Hi - f_H) ÷ s_H + 1
    n_W = (n_Wi - f_W) ÷ s_W + 1
    n_D = (n_Di - f_D) ÷ s_D + 1
    W = eval(:($cLayer.$param))
    T = eltype(W)
    B = cLayer.B
    K = nothing
    for ch=1:size(W)[end]
        w4da = W[:,:,:,:,ch]
        k2d = nothing
        k3d = nothing
        k4d = nothing
        k5d = nothing
        k = nothing
        k1 = nothing
        for l=range(1, step=s_D, length=n_D)
            w4db = PaddedView(0, w4da, (f_H, f_W, n_Di, ci), (1,1,l,1))
            for i=range(1, step=s_H, length=n_H)
                w4dc = PaddedView(0, w4db, (n_Hi, f_W, n_Di, ci), (i,1,1,1))
                for j=range(1, step=s_W, length=n_W)
                    w4d = PaddedView(0, w4dc, (n_Hi, n_Wi, n_Di, ci), (1,j,1,1))
                    for w=1:size(w4d)[4]
                        for d=1:size(w4d)[3]
                            for h=1:size(w4d)[1]
                                if h==1
                                    k2d = w4d[h,:,d,w]
                                else
                                    k2d = vcat(k2d, w4d[h,:,d,w])
                                end #if h==1
                            end #for h=1:size(w)[1]
                            if d==1
                                k3d = deepcopy(k2d)
                            else
                                k3d = vcat(k3d, k2d)
                            end #if w==1
                        end #for w=1:size(w3d)[3]
                        if w==1
                            k4d = deepcopy(k3d)
                        else
                            k4d = vcat(k4d, k3d)
                        end #if d==1
                    end #for d=1:size(w4d)[4]
                    if j==1
                        k5d = deepcopy(k4d)
                    else
                        k5d = hcat(k5d, k4d)
                    end
                end #for j=1:n_H

                if i==1
                    k = deepcopy(k5d)
                else
                    k = hcat(k, k5d)
                end #if i==1
            end #for i=1:n_W
            if l==1
                k1 = deepcopy(k)
            else
                k1 = hcat(k1, k)
            end #if l==1
        end #for l=range(1, step=s_D, length=n_D)
        if ch==1
            K = k1
        else
            K = hcat(K, k1)
        end #if ch==1
    end #for ch=1:size(W)[end]

    return transpose(K)

end #function unroll(cLayer::Conv2D)


export unroll



function unrollcol(cLayer::Conv2D, AiS::Tuple, param::Symbol=:W)
    prevLayer = cLayer.prevLayer
    n_Hi, n_Wi, ci, m = AiS
    HWi = n_Hi*n_Wi
    f_H, f_W = cLayer.f
    fHW = f_H * f_W
    s_H, s_W = cLayer.s
    c = cLayer.channels
    n_H = (n_Hi - f_H) ÷ s_H + 1
    n_W = (n_Wi - f_W) ÷ s_W + 1
    W = eval(:($cLayer.$param))
    T = eltype(W)
    B = cLayer.B
    K = nothing
    for ch=1:size(W)[end]
        w3da = W[:,:,:,ch]
        k2d = nothing
        k3d = nothing
        k4d = nothing
        k = nothing
        for i=range(1, step=s_W, length=n_W)
            w3db = PaddedView(0, w3da, (f_H, n_Wi,ci), (1,i,1))
            for j=range(1, step=s_H, length=n_H)
                w3d = PaddedView(0, w3db, (n_Hi, n_Wi,ci), (j,1,1))
                for cj=1:size(w3d)[3]
                    for w=1:size(w3d)[2]
                        if w==1
                            k2d = w3d[:,w,cj]
                        else
                            k2d = vcat(k2d, w3d[:,w,cj])
                        end #if h==1
                    end #for h=1:size(w)[1]
                    if cj==1
                        k3d = k2d
                        k2d = nothing
                    else
                        k3d = vcat(k3d, k2d)
                        k2d = nothing
                    end #if w==1
                end #for w=1:size(w3d)[3]

                if j==1
                    k4d = k3d
                    k3d = nothing
                else
                    k4d = hcat(k4d, k3d)
                    k3d = nothing
                end
            end #for j=1:n_H

            if i==1
                k = k4d
                k4d = nothing
            else
                k = hcat(k, k4d)
                k4d = nothing
            end #if i==1
        end #for i=1:n_W

        if ch==1
            K = k
            k = nothing
            Base.GC.gc()
        else
            K = hcat(K, k)
            k2d = nothing
            Base.GC.gc()
        end #if ch==1
    end #for ch=1:size(W)[end]

    return transpose(K)

end #function unroll(cLayer::Conv2D)
